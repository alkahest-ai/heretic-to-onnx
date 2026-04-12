from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class InheritAssets:
    from_source: list[str] = field(default_factory=list)
    from_base_if_missing: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationConfig:
    smoke_prompt: str = ""
    browser_loader_class: str = ""
    processor_class: str = ""


@dataclass(slots=True)
class Manifest:
    source_model_id: str
    base_model_id: str
    architecture: str
    target_repo_id: str
    target_dtype: str
    target_device: str
    modalities: list[str]
    inherit_assets: InheritAssets
    expected_architecture: str
    expected_onnx_files: list[str]
    validation: ValidationConfig
    manifest_path: Path

    @property
    def manifest_dir(self) -> Path:
        return self.manifest_path.parent

    @property
    def slug(self) -> str:
        value = self.target_repo_id.replace("/", "-").replace(":", "-")
        return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


def _expect_mapping(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a mapping")
    return data


def _expect_list(data: Any, label: str) -> list[str]:
    if data is None:
        return []
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{label} must be a list of strings")
    return data


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data = _expect_mapping(raw, "manifest")

    inherit_raw = _expect_mapping(data.get("inherit_assets", {}), "inherit_assets")
    validation_raw = _expect_mapping(data.get("validation", {}), "validation")

    required_fields = [
        "source_model_id",
        "base_model_id",
        "architecture",
        "target_repo_id",
        "target_dtype",
        "target_device",
        "expected_architecture",
    ]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        raise ValueError(f"manifest is missing required fields: {', '.join(missing)}")

    return Manifest(
        source_model_id=str(data["source_model_id"]),
        base_model_id=str(data["base_model_id"]),
        architecture=str(data["architecture"]),
        target_repo_id=str(data["target_repo_id"]),
        target_dtype=str(data["target_dtype"]),
        target_device=str(data["target_device"]),
        modalities=_expect_list(data.get("modalities", []), "modalities"),
        inherit_assets=InheritAssets(
            from_source=_expect_list(inherit_raw.get("from_source", []), "inherit_assets.from_source"),
            from_base_if_missing=_expect_list(
                inherit_raw.get("from_base_if_missing", []),
                "inherit_assets.from_base_if_missing",
            ),
        ),
        expected_architecture=str(data["expected_architecture"]),
        expected_onnx_files=_expect_list(data.get("expected_onnx_files", []), "expected_onnx_files"),
        validation=ValidationConfig(
            smoke_prompt=str(validation_raw.get("smoke_prompt", "")),
            browser_loader_class=str(validation_raw.get("browser_loader_class", "")),
            processor_class=str(validation_raw.get("processor_class", "")),
        ),
        manifest_path=manifest_path,
    )

