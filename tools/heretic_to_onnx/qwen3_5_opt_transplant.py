from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LAYER_WEIGHT_RE = re.compile(r"^/model/layers\.(\d+)/(gdn|attn|mlp)/([^/]+)/MatMul_Quant$")
_RAW_LAYER_MATMUL_RE = re.compile(r"^model\.layers\.(\d+)\.(gdn|attn|mlp)\.([^.]*)\.MatMul\.weight$")


@dataclass(slots=True)
class TransplantReport:
    ok: bool
    output_dir: str
    source_dir: str
    template_dir: str
    decoder_weights_replaced: int
    decoder_plain_initializers_replaced: int
    embed_tokens_replaced: bool
    copied_files: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_dir": self.output_dir,
            "source_dir": self.source_dir,
            "template_dir": self.template_dir,
            "decoder_weights_replaced": self.decoder_weights_replaced,
            "decoder_plain_initializers_replaced": self.decoder_plain_initializers_replaced,
            "embed_tokens_replaced": self.embed_tokens_replaced,
            "copied_files": self.copied_files,
            "errors": self.errors,
        }


def _copy_required_file(src: Path, dst: Path, copied: list[str], errors: list[str], *, required: bool = True) -> None:
    if not src.exists():
        if required:
            errors.append(f"missing required file: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def _normalize_browser_config(value):
    if isinstance(value, dict):
        return {key: _normalize_browser_config(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_browser_config(child) for child in value]
    if value == "bfloat16":
        return "float16"
    return value


def _write_browser_config(source_config: Path, output_config: Path, *, include_vision: bool = True) -> None:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config = _normalize_browser_config(config)
    external_data_format = {
        "decoder_model_merged.onnx": 2,
        "embed_tokens": 1,
        "decoder_model_merged": 1,
    }
    if include_vision:
        external_data_format["vision_encoder"] = 1
    config["transformers.js_config"] = {
        "use_external_data_format": external_data_format,
        "kv_cache_dtype": {
            "q4": "float16",
            "q4f16": "float16",
            "fp16": "float16",
        },
    }
    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_q8_browser_config(source_config: Path, output_config: Path, *, include_vision: bool = True) -> None:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config = _normalize_browser_config(config)
    external_data_format = {
        "embed_tokens.onnx": 1,
        "decoder_model_merged.onnx": 2,
        "embed_tokens_quantized.onnx": 1,
        "decoder_model_merged_quantized.onnx": 1,
    }
    if include_vision:
        external_data_format["vision_encoder.onnx"] = 1
        external_data_format["vision_encoder_quantized.onnx"] = 1
    config["transformers.js_config"] = {
        "use_external_data_format": external_data_format,
        "kv_cache_dtype": {
            "q4f16": "float16",
            "fp16": "float16",
        },
    }
    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_vision_dtype(*, decoder_dtype: str, vision_dtype: str) -> str:
    if vision_dtype == "auto":
        return "q8" if decoder_dtype == "q8" else "fp16"
    if vision_dtype not in {"fp16", "q4", "q8"}:
        raise ValueError(f"unsupported vision dtype: {vision_dtype}")
    return vision_dtype


def _vision_artifact_stem(vision_dtype: str) -> str:
    if vision_dtype == "fp16":
        return "vision_encoder_fp16"
    if vision_dtype == "q4":
        return "vision_encoder_q4"
    if vision_dtype == "q8":
        return "vision_encoder_quantized"
    raise ValueError(f"unsupported vision dtype: {vision_dtype}")


class _SourceTensors:
    """Read tensors from either single-file or sharded HF safetensors checkpoints."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._single = checkpoint_path.suffix == ".safetensors"
        self._weight_map: dict[str, str] = {}
        self._contexts: list[Any] = []
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "_SourceTensors":
        from safetensors import safe_open

        if self._single:
            context = safe_open(self.checkpoint_path, framework="np", device="cpu")
            self._contexts.append(context)
            self._handles[""] = context.__enter__()
            return self

        index = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        self._weight_map = dict(index.get("weight_map") or {})
        for shard_name in sorted(set(self._weight_map.values())):
            context = safe_open(self.checkpoint_path.parent / shard_name, framework="np", device="cpu")
            self._contexts.append(context)
            self._handles[shard_name] = context.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for context in reversed(self._contexts):
            context.__exit__(exc_type, exc, traceback)
        self._contexts.clear()
        self._handles.clear()

    def get_tensor(self, key: str):
        if self._single:
            return self._handles[""].get_tensor(key)
        shard_name = self._weight_map.get(key)
        if shard_name is None:
            raise KeyError(key)
        return self._handles[shard_name].get_tensor(key)


def _open_source_tensors(checkpoint_path: Path) -> _SourceTensors:
    return _SourceTensors(checkpoint_path)


def _source_checkpoint_path(source: Path) -> Path | None:
    single = source / "model.safetensors"
    if single.exists():
        return single
    index = source / "model.safetensors.index.json"
    if index.exists():
        return index
    safetensors_files = sorted(source.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]
    return None


def _write_tokenizer_config(
    *,
    template_tokenizer_config: Path,
    output_tokenizer_config: Path,
) -> None:
    tokenizer_config = json.loads(template_tokenizer_config.read_text(encoding="utf-8"))
    output_tokenizer_config.parent.mkdir(parents=True, exist_ok=True)
    output_tokenizer_config.write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pack_int4(values):
    import numpy as np

    if values.shape[-1] % 2:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)])
    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).astype("uint8")


def _replace_initializer(model, tensor) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == tensor.name:
            model.graph.initializer[index].CopyFrom(tensor)
            return
    raise KeyError(f"initializer not found: {tensor.name}")


def _tensor_to_numpy(tensor, *, dtype):
    import numpy as np

    if hasattr(tensor, "detach"):
        array = tensor.detach().cpu().to(dtype).numpy()
    else:
        array = np.asarray(tensor, dtype=dtype)
    return np.ascontiguousarray(array)


def _make_tensor_like(name: str, tensor, template_initializer):
    import numpy as np
    from onnx import TensorProto, numpy_helper

    if template_initializer.data_type == TensorProto.FLOAT16:
        array = _tensor_to_numpy(tensor, dtype=np.float16)
    else:
        array = _tensor_to_numpy(tensor, dtype=np.float32)
    if template_initializer.data_type == TensorProto.FLOAT16:
        array = array.astype(np.float16, copy=False)
    else:
        array = array.astype(np.float32, copy=False)
    array = np.ascontiguousarray(array)
    return numpy_helper.from_array(array, name)


def _cast_quantized_tensor_like(tensor, template_initializer):
    import numpy as np
    from onnx import TensorProto, numpy_helper

    if tensor.data_type == template_initializer.data_type:
        return tensor
    array = numpy_helper.to_array(tensor)
    if template_initializer.data_type == TensorProto.FLOAT:
        array = array.astype(np.float32, copy=False)
    elif template_initializer.data_type == TensorProto.FLOAT16:
        array = array.astype(np.float16, copy=False)
    elif template_initializer.data_type == TensorProto.UINT8:
        array = array.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"unsupported quantized initializer dtype: {template_initializer.data_type}")
    array = np.ascontiguousarray(array)
    return numpy_helper.from_array(array, tensor.name)


def _quantize_matmul_weight(
    weight,
    *,
    quant_name: str,
    scales_name: str,
    zero_points_name: str,
    template_initializers: dict[str, Any],
    block_size: int,
):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    weight_np = _tensor_to_numpy(weight, dtype=np.float32).T
    temporary_dtype = np.float16 if weight_np.size * np.dtype(np.float32).itemsize > 1_000_000_000 else np.float32
    temporary_tensor_type = TensorProto.FLOAT16 if temporary_dtype == np.float16 else TensorProto.FLOAT
    weight_np = np.ascontiguousarray(weight_np.astype(temporary_dtype, copy=False))
    k_dim, n_dim = weight_np.shape
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul",
                ["x", "W"],
                ["y"],
                name="MatMul",
            )
        ],
        "quantize_weight",
        [helper.make_tensor_value_info("x", temporary_tensor_type, [1, k_dim])],
        [helper.make_tensor_value_info("y", temporary_tensor_type, [1, n_dim])],
        [numpy_helper.from_array(weight_np, "W")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=block_size, is_symmetric=False)
    quantizer.process()
    quantized = {initializer.name: initializer for initializer in quantizer.model.model.graph.initializer}

    out = []
    for source_name, target_name in (
        ("W_Q4", quant_name),
        ("W_scales", scales_name),
        ("W_zero_points", zero_points_name),
    ):
        tensor = onnx.TensorProto()
        tensor.CopyFrom(quantized[source_name])
        tensor.name = target_name
        out.append(_cast_quantized_tensor_like(tensor, template_initializers[target_name]))
    return out


def _state_key_for_matmul(node_name: str) -> str:
    if node_name == "/lm_head/MatMul_Quant":
        return "model.language_model.embed_tokens.weight"

    match = _LAYER_WEIGHT_RE.match(node_name)
    if not match:
        raise ValueError(f"unsupported MatMulNBits node name: {node_name}")

    layer, block, proj = match.groups()
    if block == "gdn":
        module = "linear_attn"
    elif block == "attn":
        module = "self_attn"
    else:
        module = "mlp"
    return f"model.language_model.layers.{layer}.{module}.{proj}.weight"


def _state_key_for_plain_initializer(name: str) -> str | None:
    if re.match(r"^model\.layers\.\d+\.final_norm_layernorm\.weight$", name):
        return "model.language_model.norm.weight"

    layer_match = re.match(r"^model\.layers\.(\d+)\.(.+)$", name)
    if not layer_match:
        return None

    layer, suffix = layer_match.groups()
    prefix = f"model.language_model.layers.{layer}."
    if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
        return prefix + suffix
    if suffix.startswith("attn.q_norm.layernorm."):
        return prefix + suffix.replace("attn.q_norm.layernorm.", "self_attn.q_norm.")
    if suffix.startswith("attn.k_norm.layernorm."):
        return prefix + suffix.replace("attn.k_norm.layernorm.", "self_attn.k_norm.")
    if suffix == "gdn.conv1d.weight":
        return prefix + "linear_attn.conv1d.weight"
    if suffix == "gdn.conv1d.weight_3d":
        return prefix + "linear_attn.conv1d.weight"
    if suffix == "gdn.dt_bias":
        return prefix + "linear_attn.dt_bias"
    if suffix == "gdn.norm.weight":
        return prefix + "linear_attn.norm.weight"
    return None


def _needs_qwen35_rmsnorm_offset(name: str) -> bool:
    # Qwen3.5 RMSNorm stores a delta and applies (1 + weight) at runtime.
    # ORT's SimplifiedLayerNormalization initializer is the direct multiplier.
    if re.match(r"^model\.layers\.\d+\.final_norm_layernorm\.weight$", name):
        return True
    return (
        name.endswith(".input_layernorm.weight")
        or name.endswith(".post_attention_layernorm.weight")
        or ".attn.q_norm.layernorm.weight" in name
        or ".attn.k_norm.layernorm.weight" in name
    )


def _state_key_for_raw_matmul_initializer(name: str) -> str | None:
    if name == "lm_head.MatMul.weight":
        return "model.language_model.embed_tokens.weight"

    match = _RAW_LAYER_MATMUL_RE.match(name)
    if not match:
        return None

    layer, block, proj = match.groups()
    if block == "gdn":
        module = "linear_attn"
    elif block == "attn":
        module = "self_attn"
    else:
        module = "mlp"
    return f"model.language_model.layers.{layer}.{module}.{proj}.weight"


def _tensor_for_initializer(name: str, safe_file, initializer):
    import numpy as np

    state_key = _state_key_for_raw_matmul_initializer(name)
    if state_key is not None:
        return _make_tensor_like(name, safe_file.get_tensor(state_key).T, initializer)

    state_key = _state_key_for_plain_initializer(name)
    if state_key is None:
        return None

    tensor = safe_file.get_tensor(state_key)
    if name.endswith(".gdn.conv1d.weight"):
        tensor = tensor.squeeze(1)
    if _needs_qwen35_rmsnorm_offset(name):
        tensor = _tensor_to_numpy(tensor, dtype=np.float32) + 1.0
    return _make_tensor_like(name, tensor, initializer)


def _replace_a_neg_exp(model, safe_file, replaced: list[str]) -> None:
    import numpy as np
    from onnx import numpy_helper

    for initializer in list(model.graph.initializer):
        match = re.match(r"^model\.layers\.(\d+)\.gdn\.A_neg_exp$", initializer.name)
        if not match:
            continue
        state_key = f"model.language_model.layers.{match.group(1)}.linear_attn.A_log"
        value = -np.exp(_tensor_to_numpy(safe_file.get_tensor(state_key), dtype=np.float32))
        _replace_initializer(model, numpy_helper.from_array(value, initializer.name))
        replaced.append(initializer.name)


def _replace_raw_decoder_initializers(
    *,
    template_decoder: Path,
    source_safetensors: Path,
    output_decoder: Path,
) -> tuple[int, int]:
    import numpy as np
    import onnx

    model = onnx.load(str(template_decoder), load_external_data=False)
    matmul_replaced = 0
    plain_replaced: list[str] = []
    missing_external: list[str] = []

    with _open_source_tensors(source_safetensors) as safe_file:
        for initializer in list(model.graph.initializer):
            tensor = _tensor_for_initializer(initializer.name, safe_file, initializer)
            if tensor is None:
                if initializer.data_location == onnx.TensorProto.EXTERNAL:
                    missing_external.append(initializer.name)
                continue
            _replace_initializer(model, tensor)
            if _state_key_for_raw_matmul_initializer(initializer.name) is not None:
                matmul_replaced += 1
            else:
                plain_replaced.append(initializer.name)

        _replace_a_neg_exp(model, safe_file, plain_replaced)

    if missing_external:
        raise ValueError(
            "raw Qwen template contains external initializers without source mappings: "
            + ", ".join(missing_external[:20])
            + (" ..." if len(missing_external) > 20 else "")
        )

    output_decoder.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_decoder.with_name(f"{output_decoder.name}_data")
    if output_decoder.exists():
        output_decoder.unlink()
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        model,
        str(output_decoder),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return matmul_replaced, len(set(plain_replaced))


def _write_raw_embed_tokens(
    *,
    template_embed: Path,
    source_safetensors: Path,
    output_embed: Path,
) -> None:
    import onnx

    model = onnx.load(str(template_embed), load_external_data=False)
    with _open_source_tensors(source_safetensors) as safe_file:
        tensor = _make_tensor_like(
            "model.embed_tokens.weight",
            safe_file.get_tensor("model.language_model.embed_tokens.weight"),
            model.graph.initializer[0],
        )
    _replace_initializer(model, tensor)

    output_embed.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_embed.with_name(f"{output_embed.name}_data")
    if output_embed.exists():
        output_embed.unlink()
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        model,
        str(output_embed),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _quantize_dynamic_q8(input_path: Path, output_path: Path, *, op_types: list[str] | None = None) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_name(f"{output_path.name}_data")
    ort_external_data_path = output_path.with_suffix(f"{output_path.suffix}.data")
    if output_path.exists():
        output_path.unlink()
    if external_data_path.exists():
        external_data_path.unlink()
    if ort_external_data_path.exists():
        ort_external_data_path.unlink()
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        op_types_to_quantize=op_types,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,
    )
    model = onnx.load(str(output_path), load_external_data=True)
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    if ort_external_data_path.exists():
        ort_external_data_path.unlink()


def _replace_decoder_initializers(
    *,
    template_decoder: Path,
    source_safetensors: Path,
    output_decoder: Path,
    block_size: int,
    decoder_dtype: str = "q4",
) -> tuple[int, int]:
    import numpy as np
    import onnx

    model = onnx.load(str(template_decoder), load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    quantized_count = 0
    plain_replaced: list[str] = []

    with _open_source_tensors(source_safetensors) as safe_file:
        if decoder_dtype == "q4":
            for node in model.graph.node:
                if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
                    continue
                state_key = _state_key_for_matmul(node.name)
                tensors = _quantize_matmul_weight(
                    safe_file.get_tensor(state_key),
                    quant_name=node.input[1],
                    scales_name=node.input[2],
                    zero_points_name=node.input[3],
                    template_initializers=initializers,
                    block_size=block_size,
                )
                for tensor in tensors:
                    _replace_initializer(model, tensor)
                quantized_count += 1
        elif decoder_dtype == "fp16":
            from onnx import TensorProto, helper

            fp16_tensors = []
            remove_initializers: set[str] = set()
            rewritten_nodes = []
            for node in model.graph.node:
                if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
                    rewritten_nodes.append(node)
                    continue
                original_input = node.input[0]
                original_output = node.output[0]
                state_key = _state_key_for_matmul(node.name)
                weight_name = node.input[1].removesuffix("_quant") + "_fp16"
                cast_input = f"{node.name}/input_fp16"
                matmul_output = f"{node.name}/output_fp16"
                remove_initializers.update(node.input[1:4])
                fp16_tensors.append(
                    _make_tensor_like(
                        weight_name,
                        safe_file.get_tensor(state_key).T,
                        type("Template", (), {"data_type": onnx.TensorProto.FLOAT16})(),
                    )
                )
                rewritten_nodes.extend(
                    [
                        helper.make_node(
                            "Cast",
                            [original_input],
                            [cast_input],
                            name=f"{node.name}/CastInputFp16",
                            to=TensorProto.FLOAT16,
                        ),
                        helper.make_node(
                            "MatMul",
                            [cast_input, weight_name],
                            [matmul_output],
                            name=node.name,
                        ),
                        helper.make_node(
                            "Cast",
                            [matmul_output],
                            [original_output],
                            name=f"{node.name}/CastOutputFloat",
                            to=TensorProto.FLOAT,
                        ),
                    ]
                )
                quantized_count += 1
            del model.graph.node[:]
            model.graph.node.extend(rewritten_nodes)
            kept_initializers = [
                initializer for initializer in model.graph.initializer if initializer.name not in remove_initializers
            ]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept_initializers)
            model.graph.initializer.extend(fp16_tensors)
            initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        else:
            raise ValueError(f"unsupported decoder dtype: {decoder_dtype}")

        for name, initializer in initializers.items():
            state_key = _state_key_for_plain_initializer(name)
            if state_key is None:
                continue
            tensor = safe_file.get_tensor(state_key)
            if _needs_qwen35_rmsnorm_offset(name):
                tensor = _tensor_to_numpy(tensor, dtype=np.float32) + 1.0
            _replace_initializer(model, _make_tensor_like(name, tensor, initializer))
            plain_replaced.append(name)

        _replace_a_neg_exp(model, safe_file, plain_replaced)

    output_decoder.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_decoder.with_name(f"{output_decoder.name}_data")
    if output_decoder.exists():
        output_decoder.unlink()
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        model,
        str(output_decoder),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return quantized_count, len(set(plain_replaced))


def _write_embed_tokens(
    *,
    template_embed: Path,
    source_safetensors: Path,
    output_embed: Path,
    block_size: int,
) -> None:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(template_embed), load_external_data=True)
    with _open_source_tensors(source_safetensors) as safe_file:
        weight = _tensor_to_numpy(safe_file.get_tensor("model.language_model.embed_tokens.weight"), dtype=np.float32)

    vocab_size, hidden_size = weight.shape
    if hidden_size % block_size:
        raise ValueError(f"embedding hidden size {hidden_size} is not divisible by block size {block_size}")

    blocks = weight.reshape(vocab_size, hidden_size // block_size, block_size)
    mins = blocks.min(axis=-1)
    maxs = blocks.max(axis=-1)
    scales = (maxs - mins) / 15.0
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    zero_points = np.rint(-mins / scales).clip(0, 15).astype(np.uint8)
    quantized = np.rint(blocks / scales[..., None] + zero_points[..., None]).clip(0, 15).astype(np.uint8)

    replacements = {
        "model_embed_tokens_weight_quant": numpy_helper.from_array(
            _pack_int4(quantized.reshape(vocab_size, hidden_size)),
            "model_embed_tokens_weight_quant",
        ),
        "model_embed_tokens_weight_scales": numpy_helper.from_array(scales, "model_embed_tokens_weight_scales"),
        "model_embed_tokens_weight_zp": numpy_helper.from_array(_pack_int4(zero_points), "model_embed_tokens_weight_zp"),
    }
    for tensor in replacements.values():
        _replace_initializer(model, tensor)

    output_embed.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_embed.with_name(f"{output_embed.name}_data")
    if output_embed.exists():
        output_embed.unlink()
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        model,
        str(output_embed),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def build_optimized_qwen35_package(
    *,
    source_dir: str | Path,
    template_dir: str | Path,
    output_dir: str | Path,
    block_size: int = 32,
    decoder_dtype: str = "q4",
    include_vision: bool = True,
    vision_dtype: str = "auto",
) -> TransplantReport:
    source = Path(source_dir).expanduser().resolve()
    template = Path(template_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    errors: list[str] = []
    copied: list[str] = []

    output.mkdir(parents=True, exist_ok=True)
    (output / "onnx").mkdir(exist_ok=True)
    resolved_vision_dtype = _resolve_vision_dtype(decoder_dtype=decoder_dtype, vision_dtype=vision_dtype)

    if (source / "config.json").exists():
        if decoder_dtype == "q8":
            _write_q8_browser_config(source / "config.json", output / "config.json", include_vision=include_vision)
        else:
            _write_browser_config(source / "config.json", output / "config.json", include_vision=include_vision)
        copied.append(str(output / "config.json"))
    else:
        errors.append(f"missing required file: {source / 'config.json'}")

    for filename in ("tokenizer.json", "chat_template.jinja"):
        _copy_required_file(source / filename, output / filename, copied, errors)
    _copy_required_file(template / "generation_config.json", output / "generation_config.json", copied, errors)
    if (template / "tokenizer_config.json").exists():
        _write_tokenizer_config(
            template_tokenizer_config=template / "tokenizer_config.json",
            output_tokenizer_config=output / "tokenizer_config.json",
        )
        copied.append(str(output / "tokenizer_config.json"))
    else:
        errors.append(f"missing required file: {template / 'tokenizer_config.json'}")
    _copy_required_file(template / "preprocessor_config.json", output / "preprocessor_config.json", copied, errors)
    if include_vision:
        vision_stem = _vision_artifact_stem(resolved_vision_dtype)
        _copy_required_file(
            template / "onnx" / f"{vision_stem}.onnx",
            output / "onnx" / f"{vision_stem}.onnx",
            copied,
            errors,
        )
        _copy_required_file(
            template / "onnx" / f"{vision_stem}.onnx_data",
            output / "onnx" / f"{vision_stem}.onnx_data",
            copied,
            errors,
        )

    source_safetensors = _source_checkpoint_path(source)
    if source_safetensors is None:
        errors.append(
            "missing source safetensors: expected model.safetensors, "
            "model.safetensors.index.json, or a single *.safetensors file under "
            f"{source}"
        )

    decoder_count = 0
    plain_count = 0
    embed_ok = False
    if not errors:
        if decoder_dtype == "q8":
            raw_embed = output / "onnx" / "embed_tokens.onnx"
            raw_decoder = output / "onnx" / "decoder_model_merged.onnx"
            _write_raw_embed_tokens(
                template_embed=template / "onnx" / "embed_tokens.onnx",
                source_safetensors=source_safetensors,
                output_embed=raw_embed,
            )
            decoder_count, plain_count = _replace_raw_decoder_initializers(
                template_decoder=template / "onnx" / "decoder_model_merged.onnx",
                source_safetensors=source_safetensors,
                output_decoder=raw_decoder,
            )
            _quantize_dynamic_q8(raw_embed, output / "onnx" / "embed_tokens_quantized.onnx")
            _quantize_dynamic_q8(
                raw_decoder,
                output / "onnx" / "decoder_model_merged_quantized.onnx",
                op_types=["MatMul"],
            )
        else:
            _write_embed_tokens(
                template_embed=template / "onnx" / "embed_tokens_q4.onnx",
                source_safetensors=source_safetensors,
                output_embed=output / "onnx" / "embed_tokens_q4.onnx",
                block_size=block_size,
            )
            decoder_count, plain_count = _replace_decoder_initializers(
                template_decoder=template / "onnx" / "decoder_model_merged_q4.onnx",
                source_safetensors=source_safetensors,
                output_decoder=output / "onnx" / f"decoder_model_merged_{decoder_dtype}.onnx",
                block_size=block_size,
                decoder_dtype=decoder_dtype,
            )
        embed_ok = True

    report = TransplantReport(
        ok=not errors,
        output_dir=str(output),
        source_dir=str(source),
        template_dir=str(template),
        decoder_weights_replaced=decoder_count,
        decoder_plain_initializers_replaced=plain_count,
        embed_tokens_replaced=embed_ok,
        copied_files=copied,
        errors=errors,
    )
    (output / "package-report.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Qwen3.5 optimized browser package from a safetensors checkpoint.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--template-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--decoder-dtype", choices=["q4", "fp16", "q8"], default="q4")
    parser.add_argument("--vision-dtype", choices=["auto", "fp16", "q4", "q8"], default="auto")
    parser.add_argument("--text-only", action="store_true", help="Do not copy/package vision encoder ONNX files.")
    args = parser.parse_args(argv)

    report = build_optimized_qwen35_package(
        source_dir=args.source_dir,
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        block_size=args.block_size,
        decoder_dtype=args.decoder_dtype,
        include_vision=not args.text_only,
        vision_dtype=args.vision_dtype,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
