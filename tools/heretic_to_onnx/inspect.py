from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest
from .repo import RepoAccessError, RepoHandle


SUPPORTED_ARCHITECTURES = {
    "gemma4_conditional_generation",
    "qwen3_5_conditional_generation",
}


@dataclass(slots=True)
class InspectionReport:
    ok: bool
    source_repo: str
    base_repo: str
    expected_architecture: str
    detected_architecture: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_assets: dict[str, bool] = field(default_factory=dict)
    inherited_assets: dict[str, str] = field(default_factory=dict)
    has_safetensors_weights: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_architecture(source_repo: RepoHandle) -> tuple[str | None, dict[str, Any] | None]:
    if not source_repo.exists("config.json"):
        return None, None
    config = source_repo.read_json("config.json")
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0]), config
    return None, config


def _can_synthesize_preprocessor(repo: RepoHandle) -> bool:
    if not repo.exists("processor_config.json"):
        return False
    processor_config = repo.read_json("processor_config.json")
    image_processor = processor_config.get("image_processor")
    return isinstance(image_processor, dict) and bool(image_processor)


def _can_synthesize_video_preprocessor(repo: RepoHandle) -> bool:
    if not repo.exists("processor_config.json"):
        return False
    processor_config = repo.read_json("processor_config.json")
    video_processor = processor_config.get("video_processor")
    return isinstance(video_processor, dict) and bool(video_processor)


def inspect_manifest(
    manifest: Manifest,
    *,
    source_spec: str | Path | None = None,
    base_spec: str | Path | None = None,
    require_weights: bool = True,
) -> InspectionReport:
    source_repo = RepoHandle(str(source_spec or manifest.source_model_id), manifest.manifest_dir)
    base_repo = RepoHandle(str(base_spec or manifest.base_model_id), manifest.manifest_dir)

    report = InspectionReport(
        ok=True,
        source_repo=source_repo.descriptor,
        base_repo=base_repo.descriptor,
        expected_architecture=manifest.expected_architecture,
    )

    try:
        detected_architecture, source_config = _detect_architecture(source_repo)
        report.detected_architecture = detected_architecture
        if detected_architecture != manifest.expected_architecture:
            report.ok = False
            report.errors.append(
                f"expected architecture {manifest.expected_architecture}, got {detected_architecture!r}"
            )

        report.has_safetensors_weights = source_repo.exists("model.safetensors") or source_repo.exists(
            "model.safetensors.index.json"
        )
        if require_weights and not report.has_safetensors_weights:
            report.ok = False
            report.errors.append("source repo is missing model.safetensors or model.safetensors.index.json")
        elif not require_weights and not report.has_safetensors_weights:
            report.warnings.append("source weight file presence was not verified in metadata-only mode")

        for asset in manifest.inherit_assets.from_source:
            exists = source_repo.exists(asset)
            report.source_assets[asset] = exists
            if not exists:
                report.ok = False
                report.errors.append(f"source repo is missing required asset: {asset}")

        for asset in manifest.inherit_assets.from_base_if_missing:
            if source_repo.exists(asset):
                report.inherited_assets[asset] = source_repo.descriptor
            elif base_repo.exists(asset):
                report.inherited_assets[asset] = base_repo.descriptor
            elif asset == "preprocessor_config.json" and (
                _can_synthesize_preprocessor(source_repo) or _can_synthesize_preprocessor(base_repo)
            ):
                descriptor = source_repo.descriptor if _can_synthesize_preprocessor(source_repo) else base_repo.descriptor
                report.inherited_assets[asset] = f"synthetic-from:{descriptor}/processor_config.json"
                report.warnings.append("preprocessor_config.json will be synthesized from processor_config.json")
            elif asset == "video_preprocessor_config.json" and (
                _can_synthesize_video_preprocessor(source_repo) or _can_synthesize_video_preprocessor(base_repo)
            ):
                descriptor = (
                    source_repo.descriptor
                    if _can_synthesize_video_preprocessor(source_repo)
                    else base_repo.descriptor
                )
                report.inherited_assets[asset] = f"synthetic-from:{descriptor}/processor_config.json"
                report.warnings.append(
                    "video_preprocessor_config.json will be synthesized from processor_config.json"
                )
            else:
                report.ok = False
                report.errors.append(f"source/base repos are both missing required asset: {asset}")

        if source_config and "auto_map" in source_config:
            report.warnings.append("source config contains auto_map; verify trust_remote_code is not required")

        if manifest.architecture not in SUPPORTED_ARCHITECTURES:
            report.warnings.append(
                f"architecture family {manifest.architecture!r} is not implemented yet beyond packaging"
            )
    except RepoAccessError as error:
        report.ok = False
        report.errors.append(str(error))

    return report
