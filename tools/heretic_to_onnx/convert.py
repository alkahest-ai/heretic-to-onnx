from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Manifest
from .export_gemma4 import export_gemma4
from .export_qwen3_5 import export_qwen3_5
from .inspect import inspect_manifest
from .package_repo import package_repo
from .prepare import prepare_repos
from .quantize_gemma4 import quantize_gemma4
from .quantize_qwen3_5 import quantize_qwen3_5
from .validate_repo import validate_package
from .workdir import resolve_work_dir


@dataclass(slots=True)
class ConvertReport:
    ok: bool
    output_dir: str
    inspect: dict[str, Any]
    prepare: dict[str, Any]
    export: dict[str, Any]
    quantize: dict[str, Any]
    package: dict[str, Any]
    validate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw_export_dir(layout, manifest: Manifest) -> Path:
    if manifest.architecture == "gemma4_conditional_generation":
        return layout.export_raw / "gemma4"
    if manifest.architecture == "qwen3_5_conditional_generation":
        return layout.export_raw / "qwen3_5"
    raise ValueError(f"unsupported architecture family: {manifest.architecture}")


def _run_export(
    manifest: Manifest,
    layout,
    *,
    mode: str,
    python_exec: str,
    device: str,
    torch_dtype: str,
    opset_version: int,
):
    if manifest.architecture == "gemma4_conditional_generation":
        return export_gemma4(
            manifest,
            layout.root,
            mode=mode,
            python_exec=python_exec,
            device=device,
            torch_dtype=torch_dtype,
            opset_version=opset_version,
        )
    if manifest.architecture == "qwen3_5_conditional_generation":
        return export_qwen3_5(
            manifest,
            layout.root,
            mode=mode,
            python_exec=python_exec,
            device=device,
            torch_dtype=torch_dtype,
            opset_version=opset_version,
        )
    raise ValueError(f"unsupported architecture family: {manifest.architecture}")


def _run_quantize(
    manifest: Manifest,
    layout,
    *,
    mode: str,
    python_exec: str,
    raw_onnx_dir: Path,
    output_dir: Path,
    block_size: int,
):
    if manifest.architecture == "gemma4_conditional_generation":
        return quantize_gemma4(
            manifest,
            layout.root,
            mode=mode,
            python_exec=python_exec,
            raw_onnx_dir=raw_onnx_dir,
            output_dir=output_dir,
            block_size=block_size,
        )
    if manifest.architecture == "qwen3_5_conditional_generation":
        return quantize_qwen3_5(
            manifest,
            layout.root,
            mode=mode,
            python_exec=python_exec,
            raw_onnx_dir=raw_onnx_dir,
            output_dir=output_dir,
            block_size=block_size,
        )
    raise ValueError(f"unsupported architecture family: {manifest.architecture}")


def run_convert(
    manifest: Manifest,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    strict_onnx: bool = False,
    runtime_smoke: bool | None = None,
    work_dir: str | Path | None = None,
    onnx_source_dir: str | Path | None = None,
    export_mode: str = "plan",
    quantize_mode: str = "plan",
    python_exec: str = "python3",
    export_device: str = "cpu",
    export_torch_dtype: str = "auto",
    opset_version: int = 17,
    block_size: int = 32,
) -> ConvertReport:
    resolved_runtime_smoke = (
        runtime_smoke
        if runtime_smoke is not None
        else strict_onnx or export_mode == "execute" or quantize_mode == "execute"
    )
    layout = resolve_work_dir(manifest, work_dir).ensure()
    prepare_report = prepare_repos(manifest, layout.root, source_mode="metadata")
    inspect_report = inspect_manifest(
        manifest,
        source_spec=prepare_report.source_path,
        base_spec=prepare_report.base_path,
        require_weights=False,
    )
    if not inspect_report.ok:
        return ConvertReport(
            ok=False,
            output_dir=str(Path(output_dir).resolve()) if output_dir else "",
            inspect=inspect_report.to_dict(),
            prepare=prepare_report.to_dict(),
            export={},
            quantize={},
            package={},
            validate={},
        )

    export_report = _run_export(
        manifest,
        layout,
        mode=export_mode,
        python_exec=python_exec,
        device=export_device,
        torch_dtype=export_torch_dtype,
        opset_version=opset_version,
    )
    quantize_report = _run_quantize(
        manifest,
        layout,
        mode=quantize_mode,
        python_exec=python_exec,
        raw_onnx_dir=_raw_export_dir(layout, manifest),
        output_dir=layout.export_quantized,
        block_size=block_size,
    )

    resolved_onnx_source_dir = (
        Path(onnx_source_dir).expanduser().resolve()
        if onnx_source_dir
        else layout.export_quantized
    )

    package_report = package_repo(
        manifest,
        output_dir=output_dir or layout.package_dir,
        force=force,
        allow_missing_onnx=not strict_onnx,
        onnx_source_dir=resolved_onnx_source_dir,
        source_spec=prepare_report.source_path,
        base_spec=prepare_report.base_path,
    )
    validation_report = validate_package(
        manifest,
        package_report.output_dir,
        strict_onnx=strict_onnx,
        runtime_smoke=resolved_runtime_smoke,
    )

    return ConvertReport(
        ok=export_report.ok and quantize_report.ok and package_report.ok and validation_report.ok,
        output_dir=package_report.output_dir,
        inspect=inspect_report.to_dict(),
        prepare=prepare_report.to_dict(),
        export=export_report.to_dict(),
        quantize=quantize_report.to_dict(),
        package=package_report.to_dict(),
        validate=validation_report.to_dict(),
    )
