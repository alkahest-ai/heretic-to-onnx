from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest
from .runtime import resolve_hf_command, run_command


@dataclass(slots=True)
class PublishHFReport:
    ok: bool
    repo_id: str
    package_dir: str
    model_card_path: str
    commands: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_model_card(manifest: Manifest, repo_id: str) -> str:
    title = repo_id.split("/")[-1]
    architecture_label = manifest.expected_architecture
    sessions = [f"- `{Path(path).name}`" for path in manifest.expected_onnx_files if path.endswith(".onnx")]
    if manifest.architecture == "gemma4_conditional_generation":
        summary = "Browser-oriented ONNX export of a Gemma 4 Heretic checkpoint packaged for WebGPU / Transformers.js."
    elif manifest.architecture == "qwen3_5_conditional_generation":
        summary = "Browser-oriented ONNX export of a Qwen3.5 Heretic checkpoint packaged for WebGPU / Transformers.js."
    else:
        summary = "Browser-oriented ONNX export packaged for WebGPU / Transformers.js."
    return "\n".join(
        [
            f"# {title}",
            "",
            summary,
            "",
            "## Provenance",
            f"- Source model: `{manifest.source_model_id}`",
            f"- Base model for inherited processor assets: `{manifest.base_model_id}`",
            f"- Architecture family: `{manifest.architecture}`",
            f"- Expected architecture: `{architecture_label}`",
            f"- Target dtype: `{manifest.target_dtype}`",
            f"- Target device: `{manifest.target_device}`",
            "",
            "## Expected ONNX Sessions",
            *sessions,
            "",
            "## Usage",
            "Load this repo with Transformers.js using the model's `transformers.js_config` metadata and WebGPU backend.",
            "",
        ]
    )


def _ensure_model_card(manifest: Manifest, package_dir: Path, repo_id: str) -> Path:
    model_card_path = package_dir / "README.md"
    if not model_card_path.exists():
        model_card_path.write_text(_default_model_card(manifest, repo_id), encoding="utf-8")
    return model_card_path


def publish_hf(
    manifest: Manifest,
    *,
    package_dir: str | Path,
    repo_id: str | None = None,
    private: bool = False,
    num_workers: int = 8,
) -> PublishHFReport:
    resolved_package_dir = Path(package_dir).expanduser().resolve()
    if not resolved_package_dir.exists():
        raise FileNotFoundError(f"package directory does not exist: {resolved_package_dir}")

    resolved_repo_id = repo_id or manifest.target_repo_id
    model_card_path = _ensure_model_card(manifest, resolved_package_dir, resolved_repo_id)
    hf_command = resolve_hf_command()

    create_command = [*hf_command, "repos", "create", resolved_repo_id, "--type", "model", "--exist-ok"]
    if private:
        create_command.append("--private")

    upload_command = [
        *hf_command,
        "upload-large-folder",
        resolved_repo_id,
        str(resolved_package_dir),
        "--type",
        "model",
        "--num-workers",
        str(num_workers),
        "--no-bars",
    ]

    report = PublishHFReport(
        ok=True,
        repo_id=resolved_repo_id,
        package_dir=str(resolved_package_dir),
        model_card_path=str(model_card_path),
        commands=[create_command, upload_command],
    )

    if not os.environ.get("HF_TOKEN"):
        report.ok = False
        report.warnings.append("HF_TOKEN is not set; Hugging Face upload will fail")
        return report

    run_command(create_command, cwd=manifest.manifest_dir)
    report.notes.append(f"ensured Hugging Face model repo exists: {resolved_repo_id}")

    run_command(upload_command, cwd=manifest.manifest_dir)
    report.notes.append(f"uploaded package directory to Hugging Face: {resolved_package_dir}")

    return report
