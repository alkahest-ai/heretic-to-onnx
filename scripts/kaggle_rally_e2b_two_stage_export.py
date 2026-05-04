#!/usr/bin/env python3
"""Kaggle export/publish pipeline for Rally/Gemma E2B direct and RP browser packages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class PackageTarget:
    name: str
    template: str
    source_model_id: str
    repo_id: str
    full_export: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="", help="Directory containing the completed Rally E2B SFT output.")
    parser.add_argument("--artifact-name", default="rally-e2b-two-stage-sft")
    parser.add_argument("--work-dir", default="/kaggle/working/rally-e2b-browser-export")
    parser.add_argument("--direct-source-model-id", default="p-e-w/gemma-4-E2B-it-heretic-ara")
    parser.add_argument("--base-model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--direct-full-repo", default="thomasjvu/rally-2b")
    parser.add_argument("--direct-text-repo", default="thomasjvu/rally-2b-text")
    parser.add_argument("--rp-merged-repo", default="thomasjvu/rally-2b-rp-a100-b75-merged")
    parser.add_argument("--rp-full-repo", default="thomasjvu/rally-2b-rp")
    parser.add_argument("--rp-text-repo", default="thomasjvu/rally-2b-rp-text")
    parser.add_argument("--candidate-name", default="a100-b75")
    parser.add_argument("--stage-b-scale", type=float, default=0.75)
    parser.add_argument("--export-device", default="cuda")
    parser.add_argument("--export-torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-rp", action="store_true")
    parser.add_argument("--skip-full-packages", action="store_true")
    parser.add_argument("--score", action="store_true", default=True)
    parser.add_argument("--no-score", dest="score", action="store_false")
    parser.add_argument("--require-score", action="store_true")
    parser.add_argument("--upload", action="store_true", default=True)
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--no-private", dest="private", action="store_false")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
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


def _has_merged_checkpoint(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()


def _find_artifacts(explicit: str, artifact_name: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path("/kaggle/input") / artifact_name,
            Path("/kaggle/input") / artifact_name / artifact_name,
            Path("/kaggle/working") / artifact_name,
        ]
    )
    candidates.extend(Path("/kaggle/input").glob(f"**/{artifact_name}"))

    for candidate in candidates:
        if (
            (candidate / "stage-a-adapter" / "adapter_model.safetensors").exists()
            and (candidate / "stage-b-adapter" / "adapter_model.safetensors").exists()
            and _has_merged_checkpoint(candidate / "stage-a-merged")
        ):
            return candidate.resolve()
    checked = "\n".join(str(path) for path in candidates[:30])
    raise FileNotFoundError(
        "Could not locate completed Rally E2B two-stage artifacts. Add the training notebook output as a "
        f"Kaggle kernel source or pass --artifact-dir. Checked:\n{checked}"
    )


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=str(cwd))


def _merge_scaled(stage_a_merged: Path, stage_b_adapter: Path, output_dir: Path, scale: float) -> None:
    _rm(output_dir)
    _run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts/merge_lora_scaled.py"),
            "--base-dir",
            str(stage_a_merged),
            "--adapter-dir",
            str(stage_b_adapter),
            "--output-dir",
            str(output_dir),
            "--scale",
            str(scale),
        ],
        cwd=ROOT_DIR,
    )


def _render_manifest(target: PackageTarget, manifest_dir: Path, base_model_id: str) -> Path:
    manifest_path = manifest_dir / f"{target.name}.yaml"
    _run(
        [
            sys.executable,
            "-m",
            "tools.heretic_to_onnx",
            "render-manifest",
            "--template",
            str(ROOT_DIR / target.template),
            "--output",
            str(manifest_path),
            "--source-model-id",
            target.source_model_id,
            "--base-model-id",
            base_model_id,
            "--target-repo-id",
            target.repo_id,
        ],
        cwd=ROOT_DIR,
    )
    return manifest_path


def _convert_full(target: PackageTarget, manifest_path: Path, work_dir: Path, package_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    _run(
        [
            sys.executable,
            "-m",
            "tools.heretic_to_onnx",
            "convert",
            "--config",
            str(manifest_path),
            "--work-dir",
            str(work_dir),
            "--output-dir",
            str(package_dir),
            "--force",
            "--strict-onnx",
            "--export-mode",
            "execute",
            "--quantize-mode",
            "execute",
            "--python-exec",
            sys.executable,
            "--export-device",
            args.export_device,
            "--export-torch-dtype",
            args.export_torch_dtype,
        ],
        cwd=ROOT_DIR,
    )
    return {"work_dir": str(work_dir), "package_dir": str(package_dir), "quantized_dir": str(work_dir / "export/quantized")}


def _package_text_from_quantized(
    target: PackageTarget,
    manifest_path: Path,
    work_dir: Path,
    package_dir: Path,
    quantized_dir: Path,
) -> dict[str, str]:
    _run(
        [
            sys.executable,
            "-m",
            "tools.heretic_to_onnx",
            "convert",
            "--config",
            str(manifest_path),
            "--work-dir",
            str(work_dir),
            "--output-dir",
            str(package_dir),
            "--force",
            "--strict-onnx",
            "--export-mode",
            "plan",
            "--quantize-mode",
            "plan",
            "--onnx-source-dir",
            str(quantized_dir),
        ],
        cwd=ROOT_DIR,
    )
    return {"work_dir": str(work_dir), "package_dir": str(package_dir), "quantized_dir": str(quantized_dir)}


def _publish(manifest_path: Path, package_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not args.upload:
        return {"ok": False, "skipped": True, "reason": "upload disabled"}
    command = [
        sys.executable,
        "-m",
        "tools.heretic_to_onnx",
        "publish-hf",
        "--config",
        str(manifest_path),
        "--package-dir",
        str(package_dir),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.private:
        command.append("--private")
    _run(command, cwd=ROOT_DIR)
    return {"ok": True, "package_dir": str(package_dir)}


def _upload_merged(merged_dir: Path, repo_id: str, args: argparse.Namespace) -> dict[str, Any]:
    if not args.upload:
        return {"ok": False, "skipped": True, "reason": "upload disabled"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {"ok": False, "skipped": True, "reason": "HF_TOKEN is not set"}
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(merged_dir),
        commit_message="Upload Rally E2B two-stage RP A100/B75 merged checkpoint",
    )
    return {"ok": True, "repo_id": repo_id, "private": args.private}


def _score_models(direct_model_id: str, rp_model_dir: Path, *, max_new_tokens: int = 96, temperature: float = 0.2) -> dict[str, Any]:
    from scripts.alkahest_rp_scorecard import promotion_decision, score_responses

    def generate(model_spec: str | Path) -> dict[str, str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.alkahest_rp_scorecard import SMOKE_PROMPTS

        tokenizer = AutoTokenizer.from_pretrained(model_spec, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_spec, **model_kwargs)
        device = next(model.parameters()).device
        responses: dict[str, str] = {}
        for name, prompt in SMOKE_PROMPTS.items():
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            responses[name] = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            ).strip()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return responses

    direct = score_responses("direct-rally-e2b", direct_model_id, generate(direct_model_id))
    rp = score_responses("rally-e2b-rp-a100-b75", rp_model_dir, generate(rp_model_dir))
    decision = promotion_decision(direct, rp)
    return {
        "direct": asdict(direct),
        "rp": asdict(rp),
        "promotion_decision": asdict(decision),
    }


def _process_pair(
    *,
    label: str,
    source_model_id: str,
    full_repo: str,
    text_repo: str,
    full_template: str,
    text_template: str,
    work_dir: Path,
    args: argparse.Namespace,
    base_model_id: str,
) -> list[dict[str, Any]]:
    manifests_dir = work_dir / "manifests"
    packages_dir = work_dir / "packages"
    full_target = PackageTarget(label + "-full", full_template, source_model_id, full_repo, True)
    text_target = PackageTarget(label + "-text", text_template, source_model_id, text_repo, False)
    results: list[dict[str, Any]] = []

    if args.skip_full_packages:
        text_manifest = _render_manifest(text_target, manifests_dir, base_model_id)
        text_paths = _convert_full(
            text_target,
            text_manifest,
            work_dir / "work" / text_target.name,
            packages_dir / text_target.name,
            args,
        )
        text_publish = _publish(text_manifest, Path(text_paths["package_dir"]), args)
        results.append({"target": asdict(text_target), "paths": text_paths, "publish": text_publish})
        return results

    full_manifest = _render_manifest(full_target, manifests_dir, base_model_id)
    full_paths = _convert_full(
        full_target,
        full_manifest,
        work_dir / "work" / full_target.name,
        packages_dir / full_target.name,
        args,
    )
    full_publish = _publish(full_manifest, Path(full_paths["package_dir"]), args)

    text_manifest = _render_manifest(text_target, manifests_dir, base_model_id)
    text_paths = _package_text_from_quantized(
        text_target,
        text_manifest,
        work_dir / "work" / text_target.name,
        packages_dir / text_target.name,
        Path(full_paths["quantized_dir"]),
    )
    text_publish = _publish(text_manifest, Path(text_paths["package_dir"]), args)
    results.extend(
        [
            {"target": asdict(full_target), "paths": full_paths, "publish": full_publish},
            {"target": asdict(text_target), "paths": text_paths, "publish": text_publish},
        ]
    )
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "work_dir": str(work_dir),
        "artifact_dir": "",
        "candidate_name": args.candidate_name,
        "stage_b_scale": args.stage_b_scale,
        "exports": [],
        "merged_upload": {},
        "scorecard": {},
        "disk": {"start": _disk(work_dir)},
        "warnings": [],
    }

    artifacts = _find_artifacts(args.artifact_dir, args.artifact_name)
    report["artifact_dir"] = str(artifacts)
    selected_merged = work_dir / f"{args.candidate_name}-merged"
    _merge_scaled(artifacts / "stage-a-merged", artifacts / "stage-b-adapter", selected_merged, args.stage_b_scale)
    report["selected_merged"] = str(selected_merged)
    report["disk"]["after_merge"] = _disk(work_dir)

    if args.score:
        try:
            report["scorecard"] = _score_models(args.direct_source_model_id, selected_merged)
        except Exception as exc:
            report["warnings"].append(f"scorecard failed: {type(exc).__name__}: {exc}")
            if args.require_score:
                raise

    report["merged_upload"] = _upload_merged(selected_merged, args.rp_merged_repo, args)
    report["disk"]["after_merged_upload"] = _disk(work_dir)

    if not args.skip_direct:
        report["exports"].extend(
            _process_pair(
                label="direct",
                source_model_id=args.direct_source_model_id,
                full_repo=args.direct_full_repo,
                text_repo=args.direct_text_repo,
                full_template="configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml",
                text_template="configs/heretic-to-onnx.gemma4-e2b-heretic-ara-text.yaml",
                work_dir=work_dir / "direct",
                args=args,
                base_model_id=args.base_model_id,
            )
        )
        report["disk"]["after_direct"] = _disk(work_dir)
        if not args.keep_artifacts:
            _rm(work_dir / "direct" / "work")
            report["disk"]["after_direct_work_cleanup"] = _disk(work_dir)

    if not args.skip_rp:
        report["exports"].extend(
            _process_pair(
                label="rp-a100-b75",
                source_model_id=str(selected_merged),
                full_repo=args.rp_full_repo,
                text_repo=args.rp_text_repo,
                full_template="configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml",
                text_template="configs/heretic-to-onnx.gemma4-e2b-rp-text.yaml",
                work_dir=work_dir / "rp",
                args=args,
                base_model_id=args.base_model_id,
            )
        )
        report["disk"]["after_rp"] = _disk(work_dir)
        if not args.keep_artifacts:
            _rm(work_dir / "rp" / "work")
            report["disk"]["after_rp_work_cleanup"] = _disk(work_dir)

    report["ok"] = all(item.get("publish", {}).get("ok", False) for item in report["exports"]) if args.upload else True
    report_path = work_dir / "rally-e2b-browser-export-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
