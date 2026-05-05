from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .qwen3_5_opt_transplant import (
    _make_tensor_like,
    _open_source_tensors,
    _quantize_matmul_weight,
    _replace_initializer,
    _source_checkpoint_path,
    _tensor_to_numpy,
)


_MATMUL_NODE_RE = re.compile(r"^/model/layers\.(\d+)/(attn|mlp|per_layer)/([^/]+)/MatMul_Quant$")
_MATMUL_INIT_RE = re.compile(r"^(.*_MatMul_weight)_(quant|scales|zp)$")
_LAYER_NORM_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
_V_NORM_ONES_RE = re.compile(r"^/model/layers\.(\d+)/attn/v_norm/ones_weight$")


@dataclass(slots=True)
class Gemma4OptimizedPackageReport:
    ok: bool
    package_dir: str
    source_dir: str
    template_dir: str
    decoder_weights_replaced: int = 0
    decoder_plain_initializers_replaced: int = 0
    decoder_generated_initializers: int = 0
    embed_outputs_patched: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_onnx_dir(path: Path) -> Path:
    return path / "onnx" if (path / "onnx").is_dir() else path


def _state_key_candidates_for_matmul(node_name: str) -> list[str]:
    if node_name == "/model/per_layer_projection/MatMul_Quant":
        return ["model.language_model.per_layer_projection.weight"]
    if node_name == "/lm_head/MatMul_Quant":
        return ["lm_head.weight", "model.lm_head.weight", "model.language_model.embed_tokens.weight"]

    match = _MATMUL_NODE_RE.match(node_name)
    if not match:
        raise ValueError(f"unsupported Gemma4 MatMulNBits node name: {node_name}")

    layer, block, proj = match.groups()
    if block == "attn":
        module = "self_attn"
    elif block == "mlp":
        module = "mlp"
    else:
        module = ""

    if block == "per_layer":
        return [f"model.language_model.layers.{layer}.{proj}.weight"]
    return [f"model.language_model.layers.{layer}.{module}.{proj}.weight"]


def _state_key_candidates_for_plain_initializer(name: str) -> list[str]:
    if name == "model.per_layer_projection_norm.weight":
        return [
            "model.language_model.per_layer_projection_norm.weight",
            "model.language_model.per_layer_projection.norm.weight",
        ]
    if name == "model.layers.35.final_norm_layernorm.weight":
        return ["model.language_model.norm.weight"]

    match = _LAYER_NORM_RE.match(name)
    if not match:
        return []

    layer, suffix = match.groups()
    prefix = f"model.language_model.layers.{layer}."
    if suffix.startswith("attn.q_norm.layernorm"):
        return [prefix + suffix.replace("attn.q_norm.layernorm", "self_attn.q_norm") + ".weight"]
    if suffix.startswith("attn.k_norm.layernorm"):
        return [prefix + suffix.replace("attn.k_norm.layernorm", "self_attn.k_norm") + ".weight"]
    return [prefix + suffix + ".weight"]


def _get_first_tensor(safe_file: Any, candidates: list[str]):
    last_error: Exception | None = None
    for key in candidates:
        try:
            return safe_file.get_tensor(key), key
        except Exception as exc:
            last_error = exc
    candidate_text = ", ".join(candidates)
    raise KeyError(candidate_text) from last_error


def _template_initializers_by_name(model: Any) -> dict[str, Any]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _replace_quantized_decoder_weights(
    *,
    model: Any,
    safe_file: Any,
    block_size: int,
    errors: list[str],
) -> int:
    initializers = _template_initializers_by_name(model)
    replaced = 0
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
            continue
        try:
            tensor, _source_key = _get_first_tensor(safe_file, _state_key_candidates_for_matmul(node.name))
            tensors = _quantize_matmul_weight(
                tensor,
                quant_name=node.input[1],
                scales_name=node.input[2],
                zero_points_name=node.input[3],
                template_initializers=initializers,
                block_size=block_size,
            )
            for replacement in tensors:
                _replace_initializer(model, replacement)
            replaced += 1
        except Exception as exc:
            errors.append(f"{node.name}: {type(exc).__name__}: {exc}")
    return replaced


def _rotary_cache(*, positions: int, rotary_dim: int, theta: float):
    import numpy as np

    half_dim = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    steps = np.arange(positions, dtype=np.float32)
    freqs = np.outer(steps, inv_freq)
    return np.cos(freqs).astype(np.float16), np.sin(freqs).astype(np.float16)


def _generated_initializer(name: str, template_initializer: Any, config: dict[str, Any]):
    import numpy as np
    from onnx import numpy_helper

    text_config = config.get("text_config", config)
    if name in {"cos_cache_local", "sin_cache_local", "cos_cache_global", "sin_cache_global"}:
        max_positions = int(text_config.get("max_position_embeddings", 131072))
        rope_parameters = text_config.get("rope_parameters") or {}
        if name.endswith("_local"):
            head_dim = int(text_config.get("head_dim", 256))
            theta = float((rope_parameters.get("sliding_attention") or {}).get("rope_theta", 10000.0))
        else:
            head_dim = int(text_config.get("global_head_dim") or text_config.get("head_dim", 256))
            theta = float((rope_parameters.get("full_attention") or {}).get("rope_theta", 1_000_000.0))
        cos, sin = _rotary_cache(positions=max_positions, rotary_dim=head_dim, theta=theta)
        return numpy_helper.from_array(cos if name.startswith("cos_") else sin, name)

    if _V_NORM_ONES_RE.match(name):
        return numpy_helper.from_array(np.ones(tuple(template_initializer.dims), dtype=np.float16), name)

    return None


def _replace_plain_decoder_initializers(
    *,
    model: Any,
    safe_file: Any,
    config: dict[str, Any],
    errors: list[str],
) -> tuple[int, int]:
    import onnx

    replaced = 0
    generated = 0
    initializers = list(model.graph.initializer)
    for initializer in initializers:
        candidates = _state_key_candidates_for_plain_initializer(initializer.name)
        if candidates:
            try:
                tensor, _source_key = _get_first_tensor(safe_file, candidates)
                _replace_initializer(model, _make_tensor_like(initializer.name, tensor, initializer))
                replaced += 1
            except Exception as exc:
                errors.append(f"{initializer.name}: {type(exc).__name__}: {exc}")
            continue

        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            generated_tensor = _generated_initializer(initializer.name, initializer, config)
            if generated_tensor is None:
                errors.append(f"{initializer.name}: missing Gemma4 initializer mapping")
                continue
            _replace_initializer(model, generated_tensor)
            generated += 1

    return replaced, generated


def _rewrite_embed_outputs_to_float32(path: Path) -> bool:
    import onnx
    from onnx import TensorProto, helper

    if not path.exists():
        return False

    model = onnx.load(str(path), load_external_data=False)
    patched = False
    graph_output_names = [output.name for output in model.graph.output]
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        original_name = output.name
        internal_name = f"{original_name}_fp16"
        for node in model.graph.node:
            for index, node_output in enumerate(node.output):
                if node_output == original_name:
                    node.output[index] = internal_name
        model.graph.node.append(
            helper.make_node(
                "Cast",
                [internal_name],
                [original_name],
                name=f"{original_name}_CastFloat",
                to=TensorProto.FLOAT,
            )
        )
        tensor_type.elem_type = TensorProto.FLOAT
        patched = True

    if not patched:
        return False

    external_data_path = path.with_name(f"{path.name}_data")
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )
    reloaded = onnx.load(str(path), load_external_data=False)
    if [output.name for output in reloaded.graph.output] != graph_output_names:
        raise RuntimeError("Gemma4 embed output names changed while patching float32 casts")
    return True


def build_optimized_gemma4_text_package(
    *,
    source_dir: str | Path,
    template_dir: str | Path,
    package_dir: str | Path,
    block_size: int = 32,
) -> Gemma4OptimizedPackageReport:
    source = Path(source_dir).expanduser().resolve()
    template = Path(template_dir).expanduser().resolve()
    package = Path(package_dir).expanduser().resolve()
    template_onnx_dir = _resolve_onnx_dir(template)
    package_onnx_dir = package / "onnx"
    errors: list[str] = []
    warnings: list[str] = []

    source_safetensors = _source_checkpoint_path(source)
    if source_safetensors is None:
        errors.append(f"missing source safetensors under {source}")

    template_decoder = template_onnx_dir / "decoder_model_merged_q4f16.onnx"
    if not template_decoder.exists():
        errors.append(f"missing template decoder: {template_decoder}")

    config_path = source / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
        errors.append(f"missing source config: {config_path}")

    report = Gemma4OptimizedPackageReport(
        ok=False,
        package_dir=str(package),
        source_dir=str(source),
        template_dir=str(template),
        errors=errors,
        warnings=warnings,
    )
    if errors:
        return report

    import onnx

    package_onnx_dir.mkdir(parents=True, exist_ok=True)
    output_decoder = package_onnx_dir / "decoder_model_merged_q4f16.onnx"
    external_data_path = output_decoder.with_name(f"{output_decoder.name}_data")
    if output_decoder.exists():
        output_decoder.unlink()
    if external_data_path.exists():
        external_data_path.unlink()

    model = onnx.load(str(template_decoder), load_external_data=False)
    with _open_source_tensors(source_safetensors) as safe_file:
        report.decoder_weights_replaced = _replace_quantized_decoder_weights(
            model=model,
            safe_file=safe_file,
            block_size=block_size,
            errors=errors,
        )
        plain_count, generated_count = _replace_plain_decoder_initializers(
            model=model,
            safe_file=safe_file,
            config=config,
            errors=errors,
        )
        report.decoder_plain_initializers_replaced = plain_count
        report.decoder_generated_initializers = generated_count

    if not errors:
        onnx.save_model(
            model,
            str(output_decoder),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=1024,
            convert_attribute=False,
        )
        report.embed_outputs_patched = _rewrite_embed_outputs_to_float32(package_onnx_dir / "embed_tokens_q4f16.onnx")

    report.errors = errors
    report.ok = not errors
    (package / "gemma4-optimized-package-report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
