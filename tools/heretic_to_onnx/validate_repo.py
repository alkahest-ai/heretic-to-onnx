from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest

_BROWSER_FLOAT16_METADATA_TARGETS = {"q4f16", "fp16", "q4"}
_QWEN_Q4_08B_EMBED_EXTERNAL_DATA_MAX_BYTES = 350 * 1024 * 1024
_QWEN_WEBGPU_DECODER_MAX_NODES = 10_000
_QWEN_WEBGPU_REQUIRED_DECODER_OPS = {
    "CausalConvWithState",
    "LinearAttention",
    "SkipSimplifiedLayerNormalization",
}


@dataclass(slots=True)
class RuntimeSmokeSessionReport:
    onnx_path: str
    ok: bool
    providers: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class ValidationReport:
    ok: bool
    package_dir: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_smoke_enabled: bool = False
    runtime_smoke: list[RuntimeSmokeSessionReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_bfloat16_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.extend(_find_bfloat16_paths(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(_find_bfloat16_paths(child, child_prefix))
    elif value == "bfloat16":
        paths.append(prefix)
    return paths


def _runtime_smoke_onnx_sessions(
    manifest: Manifest,
    package_path: Path,
    report: ValidationReport,
) -> None:
    session_paths = [
        (relative_path, package_path / relative_path)
        for relative_path in manifest.expected_onnx_files
        if relative_path.endswith(".onnx") and (package_path / relative_path).exists()
    ]
    if not session_paths:
        report.warnings.append("runtime smoke skipped: no packaged ONNX sessions were present")
        return

    try:
        import onnxruntime as ort
    except Exception as exc:
        report.ok = False
        report.errors.append(f"runtime smoke requires onnxruntime: {exc}")
        return

    providers = ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    if hasattr(ort, "GraphOptimizationLevel"):
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    for relative_path, onnx_path in session_paths:
        session_report = RuntimeSmokeSessionReport(
            onnx_path=relative_path,
            ok=True,
            providers=list(providers),
        )
        try:
            ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
        except Exception as exc:
            session_report.ok = False
            session_report.error = str(exc)
            report.ok = False
            report.errors.append(f"ONNX session smoke failed for {relative_path}: {exc}")
        report.runtime_smoke.append(session_report)


def _validate_qwen_webgpu_contract(manifest: Manifest, package_path: Path, report: ValidationReport) -> None:
    if manifest.architecture != "qwen3_5_conditional_generation" or manifest.target_dtype != "q4":
        return

    expected = set(manifest.expected_onnx_files)
    forbidden = sorted(path for path in expected if "_q4f16" in Path(path).name)
    if forbidden:
        report.ok = False
        report.errors.append(
            "Qwen WebGPU q4 packages must not use q4f16 session artifacts: " + ", ".join(forbidden)
        )

    required_text_sessions = {
        "onnx/embed_tokens_q4.onnx",
        "onnx/decoder_model_merged_q4.onnx",
    }
    missing_required = sorted(required_text_sessions - expected)
    if missing_required:
        report.ok = False
        report.errors.append(
            "Qwen WebGPU q4 packages must declare official-style text sessions: "
            + ", ".join(missing_required)
        )

    if "image" in manifest.modalities and "onnx/vision_encoder_fp16.onnx" not in expected:
        report.ok = False
        report.errors.append("Qwen WebGPU q4 image packages must declare onnx/vision_encoder_fp16.onnx")

    if "0.8b" in manifest.target_repo_id.lower():
        embed_data = package_path / "onnx" / "embed_tokens_q4.onnx_data"
        if embed_data.exists() and embed_data.stat().st_size > _QWEN_Q4_08B_EMBED_EXTERNAL_DATA_MAX_BYTES:
            report.ok = False
            report.errors.append(
                "embed_tokens_q4.onnx_data is too large for the 0.8B q4 WebGPU contract; "
                f"got {embed_data.stat().st_size} bytes"
            )

    decoder_path = package_path / "onnx" / "decoder_model_merged_q4.onnx"
    if not decoder_path.exists():
        return

    try:
        import onnx
    except Exception as exc:
        report.ok = False
        report.errors.append(f"Qwen WebGPU decoder contract requires onnx graph inspection: {exc}")
        return

    try:
        decoder = onnx.load(str(decoder_path), load_external_data=False)
    except Exception as exc:
        report.ok = False
        report.errors.append(f"Qwen WebGPU decoder graph could not be inspected: {exc}")
        return

    node_count = len(decoder.graph.node)
    if node_count > _QWEN_WEBGPU_DECODER_MAX_NODES:
        report.ok = False
        report.errors.append(
            "Qwen WebGPU decoder graph is not browser-optimized; "
            f"got {node_count} nodes, expected <= {_QWEN_WEBGPU_DECODER_MAX_NODES}"
        )

    custom_ops = {node.op_type for node in decoder.graph.node if node.domain == "com.microsoft"}
    missing_ops = sorted(_QWEN_WEBGPU_REQUIRED_DECODER_OPS - custom_ops)
    if missing_ops:
        report.ok = False
        report.errors.append(
            "Qwen WebGPU decoder graph is missing required optimized custom ops: " + ", ".join(missing_ops)
        )


def validate_package(
    manifest: Manifest,
    package_dir: str | Path,
    *,
    strict_onnx: bool = False,
    runtime_smoke: bool = True,
) -> ValidationReport:
    package_path = Path(package_dir).expanduser().resolve()
    report = ValidationReport(
        ok=True,
        package_dir=str(package_path),
        runtime_smoke_enabled=runtime_smoke,
    )

    required_files = list(dict.fromkeys([
        *manifest.inherit_assets.from_source,
        *manifest.inherit_assets.from_base_if_missing,
    ]))
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
            if manifest.target_dtype in _BROWSER_FLOAT16_METADATA_TARGETS:
                kv_cache_dtype = transformers_js_config.get("kv_cache_dtype", {})
                if kv_cache_dtype.get(manifest.target_dtype) != "float16":
                    report.ok = False
                    report.errors.append(
                        f"config.json is missing kv_cache_dtype mapping for {manifest.target_dtype}"
                    )

                bfloat16_paths = _find_bfloat16_paths(config)
                if bfloat16_paths:
                    report.ok = False
                    report.errors.append(
                        "config.json still contains bfloat16 fields for a browser float16 target: "
                        + ", ".join(bfloat16_paths)
                    )

    _validate_qwen_webgpu_contract(manifest, package_path, report)

    missing_onnx = [
        relative_path for relative_path in manifest.expected_onnx_files if not (package_path / relative_path).exists()
    ]
    if missing_onnx:
        if strict_onnx:
            report.ok = False
            report.errors.extend(f"missing ONNX artifact: {relative_path}" for relative_path in missing_onnx)
        else:
            report.warnings.extend(f"missing ONNX artifact: {relative_path}" for relative_path in missing_onnx)

    if runtime_smoke:
        _runtime_smoke_onnx_sessions(manifest, package_path, report)

    return report
