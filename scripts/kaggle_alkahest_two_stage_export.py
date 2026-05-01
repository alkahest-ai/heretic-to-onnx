#!/usr/bin/env python3
"""Kaggle post-training pipeline for Alkahest 0.8B two-stage SFT candidates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.qwen3_5_opt_transplant import build_optimized_qwen35_package
from tools.heretic_to_onnx.validate_repo import validate_package
from scripts.alkahest_rp_scorecard import (
    CandidateScore,
    SMOKE_PROMPTS,
    PromotionDecision,
    promotion_decision as _promotion_decision,
    score_responses as _score_responses,
    select_passing as _select,
)


BASE_MODEL_ID = "thomasjvu/alkahest-0.8b-heretic-merged"
TEMPLATE_MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX-OPT"
QWEN_BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
EXPECTED_ONNX_FILES = [
    "onnx/vision_encoder_fp16.onnx",
    "onnx/embed_tokens_q4.onnx",
    "onnx/decoder_model_merged_q4.onnx",
]
TEMPLATE_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "onnx/*",
]
@dataclass(slots=True)
class CandidateSpec:
    name: str
    stage_a_scale: float
    stage_b_scale: float
    source: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="", help="Directory containing the completed two-stage SFT output.")
    parser.add_argument("--artifact-name", default="alkahest-08b-two-stage-sft")
    parser.add_argument("--work-dir", default="/kaggle/working/alkahest-08b-two-stage-export")
    parser.add_argument("--repo-prefix", default="thomasjvu/alkahest-0.8b-heretic-rp-sft-two-stage")
    parser.add_argument("--source-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--template-model-id", default=TEMPLATE_MODEL_ID)
    parser.add_argument("--qwen-base-model-id", default=QWEN_BASE_MODEL_ID)
    parser.add_argument("--max-selected", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--candidate-names",
        default="",
        help="Comma-separated candidate names to score. Empty scores the full 10/25/50/75/100 ladder.",
    )
    parser.add_argument("--baseline-min-total", type=float, default=0.70)
    parser.add_argument("--baseline-min-margin", type=float, default=0.05)
    parser.add_argument("--compare-baseline", action="store_true", default=True)
    parser.add_argument("--no-compare-baseline", dest="compare_baseline", action="store_false")
    parser.add_argument(
        "--selected-candidates",
        default="",
        help="Comma-separated candidate names to export directly, skipping local text smoke selection.",
    )
    parser.add_argument("--export", action="store_true", default=True)
    parser.add_argument("--no-export", dest="export", action="store_false")
    parser.add_argument("--upload", action="store_true", default=True)
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--no-private", dest="private", action="store_false")
    return parser


def _disk(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_gb": round(usage.free / 1024**3, 2),
        "total_gb": round(usage.total / 1024**3, 2),
    }


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _find_artifacts(explicit: str, artifact_name: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    input_slug_candidates = {
        artifact_name,
        artifact_name.replace("08b", "0-8b"),
        artifact_name.replace("0.8b", "0-8b"),
    }
    candidates.extend(
        [
            *[Path("/kaggle/input") / slug / artifact_name for slug in sorted(input_slug_candidates)],
            Path("/tmp/kaggle-alkahest-two-stage-sft-v1") / artifact_name,
        ]
    )
    candidates.extend(Path("/kaggle/input").glob(f"**/{artifact_name}"))
    for candidate in candidates:
        if (
            (candidate / "stage-a-adapter" / "adapter_model.safetensors").exists()
            and (candidate / "stage-b-adapter" / "adapter_model.safetensors").exists()
            and (candidate / "stage-a-merged" / "model.safetensors").exists()
            and (candidate / "stage-ab-merged" / "model.safetensors").exists()
        ):
            return candidate.resolve()
    checked = "\n".join(str(path) for path in candidates[:20])
    raise FileNotFoundError(
        "Could not locate completed two-stage artifacts. Add the training notebook output as a Kaggle "
        f"kernel source or pass --artifact-dir. Checked:\n{checked}"
    )


def _snapshot_download(repo_id: str, local_dir: Path, *, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
    )


def _merge(base_dir: Path, adapter_dir: Path, output_dir: Path, scale: float) -> None:
    _rm(output_dir)
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT_DIR / "scripts/merge_lora_scaled.py"),
            "--base-dir",
            str(base_dir),
            "--adapter-dir",
            str(adapter_dir),
            "--output-dir",
            str(output_dir),
            "--scale",
            str(scale),
        ]
    )


def _candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("a100-b100", 1.0, 1.0, "stage-ab-merged"),
        CandidateSpec("a100-b75", 1.0, 0.75, "merge"),
        CandidateSpec("a100-b50", 1.0, 0.5, "merge"),
        CandidateSpec("a100-b25", 1.0, 0.25, "merge"),
        CandidateSpec("a100-b10", 1.0, 0.10, "merge"),
        CandidateSpec("a75-b100", 0.75, 1.0, "merge"),
        CandidateSpec("a75-b75", 0.75, 0.75, "merge"),
        CandidateSpec("a50-b100", 0.5, 1.0, "merge"),
        CandidateSpec("a50-b50", 0.5, 0.5, "merge"),
        CandidateSpec("a25-b100", 0.25, 1.0, "merge"),
        CandidateSpec("a25-b25", 0.25, 0.25, "merge"),
        CandidateSpec("a10-b100", 0.10, 1.0, "merge"),
    ]


def _filtered_candidate_specs(value: str) -> list[CandidateSpec]:
    names = _selected_candidate_names(value)
    specs = _candidate_specs()
    if not names:
        return specs
    by_name = {spec.name: spec for spec in specs}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown candidate names: {', '.join(unknown)}")
    return [by_name[name] for name in names]


def _materialize_candidate(spec: CandidateSpec, artifacts: Path, base_dir: Path, work_dir: Path) -> tuple[Path, bool]:
    if spec.source == "stage-ab-merged":
        return artifacts / "stage-ab-merged", False
    if spec.stage_a_scale == 1.0:
        stage_a_dir = artifacts / "stage-a-merged"
        cleanup_stage_a = None
    else:
        stage_a_dir = work_dir / f"{spec.name}-stage-a"
        _merge(base_dir, artifacts / "stage-a-adapter", stage_a_dir, spec.stage_a_scale)
        cleanup_stage_a = stage_a_dir

    out = work_dir / f"{spec.name}-merged"
    _merge(stage_a_dir, artifacts / "stage-b-adapter", out, spec.stage_b_scale)
    if cleanup_stage_a is not None:
        _rm(cleanup_stage_a)
    return out, True


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def _generate_responses(model_dir: Path, *, max_new_tokens: int, temperature: float) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    use_cuda = torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if use_cuda:
        model_kwargs["dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        # CPU fallback is slower, but it lets Kaggle continue the selection/export
        # path after weekly GPU quota is exhausted.
        model_kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    model_device = next(model.parameters()).device
    responses: dict[str, str] = {}
    for name, prompt in SMOKE_PROMPTS.items():
        text = _apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        responses[name] = generated.strip()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def _score_candidate(spec: CandidateSpec, path: Path, *, max_new_tokens: int, temperature: float) -> CandidateScore:
    responses = _generate_responses(path, max_new_tokens=max_new_tokens, temperature=temperature)
    return _score_responses(spec.name, path, responses)


def _score_baseline(path: Path, *, max_new_tokens: int, temperature: float) -> CandidateScore:
    responses = _generate_responses(path, max_new_tokens=max_new_tokens, temperature=temperature)
    return _score_responses("direct-heretic", path, responses)


def _select_promoted(
    scores: list[CandidateScore],
    baseline: CandidateScore,
    *,
    max_selected: int,
    min_total: float,
    min_margin: float,
) -> tuple[list[CandidateScore], list[PromotionDecision]]:
    decisions = [
        _promotion_decision(
            baseline,
            score,
            min_total=min_total,
            min_margin=min_margin,
        )
        for score in scores
    ]
    decisions_by_name = {decision.rp: decision for decision in decisions}
    promoted = [
        score
        for score in scores
        if decisions_by_name[score.name].promoted
    ]
    return sorted(promoted, key=lambda item: item.total, reverse=True)[:max_selected], decisions


def _selected_candidate_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_readme(package_dir: Path, *, repo_id: str, score: CandidateScore) -> None:
    package_dir.joinpath("README.md").write_text(
        "\n".join(
            [
                "---",
                "license: other",
                "library_name: transformers.js",
                "pipeline_tag: text-generation",
                "private: true",
                "---",
                f"# {repo_id}",
                "",
                "Private experimental q4 WebGPU ONNX package for Alkahest Qwen two-stage SFT.",
                "",
                f"Candidate: `{score.name}`",
                f"Local Kaggle score: `{score.total}`",
                "",
                "Text sessions use official-style q4 artifacts:",
                "- `onnx/embed_tokens_q4.onnx`",
                "- `onnx/decoder_model_merged_q4.onnx`",
                "",
                "This package still requires manual browser-chat smoke before promotion.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _manifest(repo_id: str, source_id: str, base_model_id: str, manifest_dir: Path) -> Manifest:
    return Manifest(
        source_model_id=source_id,
        base_model_id=base_model_id,
        architecture="qwen3_5_conditional_generation",
        target_repo_id=repo_id,
        target_dtype="q4",
        target_device="webgpu",
        modalities=["text", "image"],
        inherit_assets=InheritAssets(),
        expected_architecture="Qwen3_5ForConditionalGeneration",
        expected_onnx_files=list(EXPECTED_ONNX_FILES),
        validation=ValidationConfig(browser_loader_class="Qwen3_5ForConditionalGeneration"),
        manifest_path=manifest_dir / "manifest.yaml",
    )


def _upload(package_dir: Path, repo_id: str, *, private: bool) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {"ok": False, "skipped": True, "error": "HF_TOKEN is not set"}

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(package_dir),
        commit_message="Upload Alkahest Qwen two-stage SFT q4 ONNX candidate",
    )
    return {"ok": True, "repo_id": repo_id, "private": private}


def _export_candidate(
    score: CandidateScore,
    *,
    repo_prefix: str,
    template_dir: Path,
    qwen_base_model_id: str,
    export_root: Path,
    private: bool,
    upload: bool,
) -> dict[str, Any]:
    repo_id = f"{repo_prefix}-{score.name}-q4-onnx"
    package_dir = export_root / f"{score.name}-q4-onnx"
    _rm(package_dir)
    report = build_optimized_qwen35_package(
        source_dir=score.path,
        template_dir=template_dir,
        output_dir=package_dir,
        decoder_dtype="q4",
    )
    _write_readme(package_dir, repo_id=repo_id, score=score)
    manifest = _manifest(repo_id, score.path, qwen_base_model_id, package_dir)
    validation = validate_package(manifest, package_dir, strict_onnx=True, runtime_smoke=True)
    upload_report = _upload(package_dir, repo_id, private=private) if upload and validation.ok else {
        "ok": False,
        "skipped": True,
        "error": "upload disabled or validation failed",
    }
    return {
        "candidate": score.name,
        "repo_id": repo_id,
        "package_dir": str(package_dir),
        "export": report.to_dict(),
        "validate": validation.to_dict(),
        "upload": upload_report,
    }


def main() -> int:
    args = _parser().parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = work_dir / "candidates"
    export_root = work_dir / "packages"
    cache_root = work_dir / "hf-cache"
    candidate_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "artifact_dir": "",
        "source_model_id": args.source_model_id,
        "template_model_id": args.template_model_id,
        "qwen_base_model_id": args.qwen_base_model_id,
        "candidate_specs": [],
        "baseline_score": None,
        "promotion_decisions": [],
        "scores": [],
        "selected": [],
        "exports": [],
        "disk": {"start": _disk(work_dir)},
    }

    artifacts = _find_artifacts(args.artifact_dir, args.artifact_name)
    report["artifact_dir"] = str(artifacts)
    template_dir = _snapshot_download(
        args.template_model_id,
        cache_root / "qwen35-onnx-template",
        allow_patterns=TEMPLATE_ALLOW_PATTERNS,
    )
    report["disk"]["after_downloads"] = _disk(work_dir)

    materialized: dict[str, tuple[Path, bool]] = {}
    scores: list[CandidateScore] = []
    specs = {spec.name: spec for spec in _candidate_specs()}
    candidate_specs = _filtered_candidate_specs(args.candidate_names)
    report["candidate_specs"] = [asdict(spec) for spec in candidate_specs]
    direct_selected = _selected_candidate_names(args.selected_candidates)
    if direct_selected:
        base_dir = None
        selected: list[CandidateScore] = []
        for name in direct_selected:
            if name not in specs:
                raise ValueError(f"unknown selected candidate: {name}")
            spec = specs[name]
            if spec.source != "stage-ab-merged" and base_dir is None:
                base_dir = _snapshot_download(args.source_model_id, cache_root / "base-heretic-merged")
            path, should_delete = _materialize_candidate(
                spec,
                artifacts,
                base_dir or artifacts / "stage-ab-merged",
                candidate_root,
            )
            materialized[spec.name] = (path, should_delete)
            selected.append(
                CandidateScore(
                    name=spec.name,
                    path=str(path),
                    total=0.0,
                    passed=True,
                    scores={"package_only": 1.0},
                    responses={},
                    errors=[],
                )
            )
        report["scores"] = [asdict(item) for item in selected]
        (work_dir / "score-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        base_dir = _snapshot_download(args.source_model_id, cache_root / "base-heretic-merged")
        baseline_score = None
        if args.compare_baseline:
            baseline_score = _score_baseline(
                base_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            report["baseline_score"] = asdict(baseline_score)
            (work_dir / "score-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for spec in candidate_specs:
            path, should_delete = _materialize_candidate(spec, artifacts, base_dir, candidate_root)
            materialized[spec.name] = (path, should_delete)
            score = _score_candidate(
                spec,
                path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            scores.append(score)
            report["scores"] = [asdict(item) for item in scores]
            (work_dir / "score-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        if baseline_score is not None:
            selected, decisions = _select_promoted(
                scores,
                baseline_score,
                max_selected=args.max_selected,
                min_total=args.baseline_min_total,
                min_margin=args.baseline_min_margin,
            )
            report["promotion_decisions"] = [asdict(item) for item in decisions]
        else:
            selected = _select(scores, args.max_selected)
    selected_names = {score.name for score in selected}
    for name, (path, should_delete) in materialized.items():
        if should_delete and name not in selected_names:
            _rm(path)

    report["selected"] = [asdict(item) for item in selected]
    report["disk"]["after_scoring"] = _disk(work_dir)

    if args.export:
        if not selected:
            report["errors"] = ["no candidates passed the required smoke score; refusing ONNX export"]
            (work_dir / "post-training-export-report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1
        for score in selected:
            export_report = _export_candidate(
                score,
                repo_prefix=args.repo_prefix,
                template_dir=template_dir,
                qwen_base_model_id=args.qwen_base_model_id,
                export_root=export_root,
                private=args.private,
                upload=args.upload,
            )
            report["exports"].append(export_report)
            report["disk"][f"after_export_{score.name}"] = _disk(work_dir)

    report["ok"] = bool(selected) and all(item.get("validate", {}).get("ok", False) for item in report["exports"])
    (work_dir / "post-training-export-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
