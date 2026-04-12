from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest


@dataclass(slots=True)
class ValidationReport:
    ok: bool
    package_dir: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_package(
    manifest: Manifest,
    package_dir: str | Path,
    *,
    strict_onnx: bool = False,
) -> ValidationReport:
    package_path = Path(package_dir).expanduser().resolve()
    report = ValidationReport(ok=True, package_dir=str(package_path))

    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
    ]
    for relative_path in required_files:
        if not (package_path / relative_path).exists():
            report.ok = False
            report.errors.append(f"missing required package file: {relative_path}")

    config_path = package_path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architectures = config.get("architectures", [])
        if architectures != [manifest.expected_architecture]:
            report.ok = False
            report.errors.append(
                f"config.json architectures must be [{manifest.expected_architecture!r}], got {architectures!r}"
            )

        transformers_js_config = config.get("transformers.js_config")
        if not isinstance(transformers_js_config, dict):
            report.ok = False
            report.errors.append("config.json is missing transformers.js_config")
        else:
            if "use_external_data_format" not in transformers_js_config:
                report.ok = False
                report.errors.append("config.json is missing transformers.js_config.use_external_data_format")
            if manifest.target_dtype in {"q4f16", "fp16"}:
                kv_cache_dtype = transformers_js_config.get("kv_cache_dtype", {})
                if kv_cache_dtype.get(manifest.target_dtype) != "float16":
                    report.ok = False
                    report.errors.append(
                        f"config.json is missing kv_cache_dtype mapping for {manifest.target_dtype}"
                    )

    missing_onnx = [
        relative_path for relative_path in manifest.expected_onnx_files if not (package_path / relative_path).exists()
    ]
    if missing_onnx:
        if strict_onnx:
            report.ok = False
            report.errors.extend(f"missing ONNX artifact: {relative_path}" for relative_path in missing_onnx)
        else:
            report.warnings.extend(f"missing ONNX artifact: {relative_path}" for relative_path in missing_onnx)

    return report

