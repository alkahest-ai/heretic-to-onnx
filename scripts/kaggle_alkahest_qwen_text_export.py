#!/usr/bin/env python3
"""Build and optionally upload a text-only Qwen3.5 q4 browser ONNX package.

This is intended for Kaggle. It keeps the official-style Qwen text artifacts
but omits the fp16 vision encoder to reduce cold browser download size.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


EXPECTED_TEXT_ONNX_FILES = [
    "onnx/embed_tokens_q4.onnx",
    "onnx/embed_tokens_q4.onnx_data",
    "onnx/decoder_model_merged_q4.onnx",
    "onnx/decoder_model_merged_q4.onnx_data",
]


@dataclass(slots=True)
class TextExportReport:
    ok: bool
    source_repo_id: str
    template_model_id: str
    base_model_id: str
    target_repo_id: str
    package_dir: str
    package_size_gb: float
    transplant: dict[str, Any]
    validation: dict[str, Any]
    upload: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", default="thomasjvu/alkahest-2b-heretic-merged")
    parser.add_argument("--template-model-id", default="onnx-community/Qwen3.5-2B-ONNX-OPT")
    parser.add_argument("--base-model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--target-repo-id", default="thomasjvu/alkahest-2b-heretic-q4-onnx-text")
    parser.add_argument("--work-dir", default="/kaggle/working/alkahest-qwen-text-export")
    parser.add_argument("--package-dir", default="")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--runtime-smoke", action="store_true", default=True)
    parser.add_argument("--no-runtime-smoke", dest="runtime_smoke", action="store_false")
    parser.add_argument("--upload", action="store_true", default=True)
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--no-private", dest="private", action="store_false")
    return parser


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _folder_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _snapshot_download(repo_id: str, local_dir: Path, *, allow_patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    _rm(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    return local_dir


def _write_readme(package_dir: Path, *, repo_id: str, source_repo_id: str) -> None:
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
                "Text-only q4 WebGPU ONNX package for Alkahest Qwen3.5.",
                "",
                f"Source checkpoint: `{source_repo_id}`",
                "",
                "Included ONNX sessions:",
                "- `onnx/embed_tokens_q4.onnx`",
                "- `onnx/decoder_model_merged_q4.onnx`",
                "",
                "The fp16 vision encoder is intentionally omitted to reduce browser cold-load size.",
                "Use a multimodal Alkahest package when image support is required.",
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
        modalities=["text"],
        inherit_assets=InheritAssets(),
        expected_architecture="Qwen3_5ForConditionalGeneration",
        expected_onnx_files=list(EXPECTED_TEXT_ONNX_FILES),
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
        commit_message="Upload text-only Alkahest Qwen q4 ONNX package",
    )
    return {"ok": True, "repo_id": repo_id, "private": private}


def main() -> int:
    args = _parser().parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    source_dir = work_dir / "source"
    template_dir = work_dir / "template"
    package_dir = (
        Path(args.package_dir).expanduser().resolve()
        if args.package_dir
        else work_dir / "package" / args.target_repo_id.replace("/", "__")
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    _rm(package_dir)

    source = _snapshot_download(
        args.source_repo_id,
        source_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "model.safetensors",
        ],
    )
    template = _snapshot_download(
        args.template_model_id,
        template_dir,
        allow_patterns=[
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "onnx/embed_tokens_q4.onnx",
            "onnx/embed_tokens_q4.onnx_data",
            "onnx/decoder_model_merged_q4.onnx",
            "onnx/decoder_model_merged_q4.onnx_data",
        ],
    )

    transplant = build_optimized_qwen35_package(
        source_dir=source,
        template_dir=template,
        output_dir=package_dir,
        block_size=args.block_size,
        decoder_dtype="q4",
        include_vision=False,
    )
    _write_readme(package_dir, repo_id=args.target_repo_id, source_repo_id=args.source_repo_id)

    manifest = _manifest(args.target_repo_id, args.source_repo_id, args.base_model_id, package_dir)
    validation = validate_package(manifest, package_dir, strict_onnx=True, runtime_smoke=args.runtime_smoke)
    upload_report = _upload(package_dir, args.target_repo_id, private=args.private) if args.upload and validation.ok else {
        "ok": False,
        "skipped": True,
        "error": "upload disabled or validation failed",
    }

    report = TextExportReport(
        ok=bool(transplant.ok and validation.ok and (upload_report.get("ok") or not args.upload)),
        source_repo_id=args.source_repo_id,
        template_model_id=args.template_model_id,
        base_model_id=args.base_model_id,
        target_repo_id=args.target_repo_id,
        package_dir=str(package_dir),
        package_size_gb=round(_folder_size(package_dir) / 1024**3, 3),
        transplant=transplant.to_dict(),
        validation=asdict(validation),
        upload=upload_report,
    )
    package_dir.joinpath("text-export-report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
