from __future__ import annotations

import json
from pathlib import Path

from .gemma4_export_codegen import ExportContract


def render_gemma4_quantize_runner(
    contract: ExportContract,
    *,
    input_dir: str,
    output_dir: str,
    report_path: str,
    block_size: int,
) -> str:
    contract_literal = repr(contract.to_dict())
    default_input_dir = json.dumps(str(Path(input_dir).expanduser().resolve()))
    default_output_dir = json.dumps(str(Path(output_dir).expanduser().resolve()))
    default_report_path = json.dumps(str(Path(report_path).expanduser().resolve()))
    return f"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTRACT = {contract_literal}


def _tensor_to_dtype(tensor, dtype):
    if tensor.data_type == dtype:
        return tensor
    if dtype == TensorProto.FLOAT:
        np_dtype = np.float32
    elif dtype == TensorProto.FLOAT16:
        np_dtype = np.float16
    else:
        return tensor
    converted = numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np_dtype), name=tensor.name)
    return converted


def _harmonize_float16_elementwise_inputs(model) -> int:
    float_types = {{TensorProto.FLOAT, TensorProto.FLOAT16}}
    value_types = {{}}

    def set_value_type(name, dtype) -> bool:
        if not name or dtype not in float_types | {{TensorProto.BFLOAT16}}:
            return False
        if value_types.get(name) == dtype:
            return False
        value_types[name] = dtype
        return True

    for initializer in model.graph.initializer:
        value_types[initializer.name] = initializer.data_type
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value_info in collection:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type:
                value_types[value_info.name] = tensor_type.elem_type

    constant_outputs = {{}}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                value_types[node.output[0]] = attr.t.data_type
                constant_outputs[node.output[0]] = attr

    initializers = {{initializer.name: initializer for initializer in model.graph.initializer}}
    def convertible_input(name) -> bool:
        return name in initializers or name in constant_outputs

    def input_target_type(node, input_types, output_types):
        for input_name, input_type in zip(node.input, input_types):
            if input_type in float_types and not convertible_input(input_name):
                return input_type
        for output_type in output_types:
            if output_type in float_types:
                return output_type
        for input_type in input_types:
            if input_type in float_types:
                return input_type
        return None

    def propagate_float_types() -> None:
        for _ in range(8):
            changed = False
            for node in model.graph.node:
                if node.op_type == "Cast":
                    cast_type = None
                    for attr in node.attribute:
                        if attr.name == "to":
                            cast_type = attr.i
                            break
                    if cast_type is not None:
                        for output_name in node.output:
                            changed = set_value_type(output_name, cast_type) or changed
                    continue

                input_types = [value_types.get(name) for name in node.input]
                known_float_inputs = [dtype for dtype in input_types if dtype in float_types]
                output_type = None
                if node.op_type in {{"ReduceMean", "Identity", "Neg", "Sqrt", "Reciprocal"}} and known_float_inputs:
                    output_type = known_float_inputs[0]
                elif node.op_type in {{"Add", "Sub", "Mul", "Div", "Pow"}} and known_float_inputs:
                    if len(set(known_float_inputs)) == 1:
                        output_type = known_float_inputs[0]

                if output_type is not None:
                    for output_name in node.output:
                        changed = set_value_type(output_name, output_type) or changed
            if not changed:
                break

    fixed = 0

    for _ in range(8):
        propagate_float_types()
        changed = False
        for node in model.graph.node:
            if node.op_type not in {{"Add", "Sub", "Mul", "Div", "Pow"}}:
                continue
            input_types = [value_types.get(name) for name in node.input]
            output_types = [value_types.get(name) for name in node.output]
            if len({{dtype for dtype in input_types if dtype in float_types}}) < 2:
                continue

            target_type = input_target_type(node, input_types, output_types)
            if target_type not in float_types:
                continue

            for input_name, input_type in zip(node.input, input_types):
                if input_type == target_type:
                    continue
                initializer = initializers.get(input_name)
                if initializer is not None and initializer.data_type in float_types:
                    initializer.CopyFrom(_tensor_to_dtype(initializer, target_type))
                    value_types[input_name] = target_type
                    fixed += 1
                    changed = True
                    continue

                attr = constant_outputs.get(input_name)
                if attr is not None and attr.t.data_type in float_types:
                    attr.t.CopyFrom(_tensor_to_dtype(attr.t, target_type))
                    value_types[input_name] = target_type
                    fixed += 1
                    changed = True
        if not changed:
            break

    return fixed


def _pack_int4(values):
    if values.shape[-1] % 2:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)])
    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).astype(np.uint8)


def _quantize_embedding_initializer(tensor, block_size: int, prefix: str) -> tuple:
    weight = numpy_helper.to_array(tensor)
    if weight.ndim != 2:
        raise ValueError(f"expected 2D embedding weight for {{tensor.name}}, got shape {{weight.shape}}")
    vocab_size, hidden_size = weight.shape
    if hidden_size % block_size:
        raise ValueError(
            f"embedding hidden size {{hidden_size}} for {{tensor.name}} is not divisible by block size {{block_size}}"
        )

    num_blocks = hidden_size // block_size
    packed_quant = np.empty((vocab_size, hidden_size // 2), dtype=np.uint8)
    scales = np.empty((vocab_size, num_blocks), dtype=np.float32)
    zero_points = np.empty((vocab_size, num_blocks), dtype=np.uint8)
    chunk_rows = 1024

    for start in range(0, vocab_size, chunk_rows):
        end = min(start + chunk_rows, vocab_size)
        blocks = weight[start:end].reshape(end - start, num_blocks, block_size).astype(np.float32)
        mins = blocks.min(axis=-1)
        maxs = blocks.max(axis=-1)
        chunk_scales = (maxs - mins) / 15.0
        chunk_scales = np.where(chunk_scales == 0, 1.0, chunk_scales).astype(np.float32)
        chunk_zero_points = np.rint(-mins / chunk_scales).clip(0, 15).astype(np.uint8)
        quantized = np.rint(blocks / chunk_scales[..., None] + chunk_zero_points[..., None])
        quantized = quantized.clip(0, 15).astype(np.uint8).reshape(end - start, hidden_size)
        packed_quant[start:end] = _pack_int4(quantized)
        scales[start:end] = chunk_scales
        zero_points[start:end] = chunk_zero_points

    del weight
    quant_name = f"{{prefix}}_quant"
    scales_name = f"{{prefix}}_scales"
    zero_points_name = f"{{prefix}}_zp"
    return (
        numpy_helper.from_array(packed_quant, quant_name),
        numpy_helper.from_array(scales, scales_name),
        numpy_helper.from_array(_pack_int4(zero_points), zero_points_name),
        {{
            "vocab_size": int(vocab_size),
            "hidden_size": int(hidden_size),
            "block_size": int(block_size),
        }},
    )


def _ensure_ms_opset(model) -> None:
    if any(opset.domain == "com.microsoft" for opset in model.opset_import):
        return
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def _quantize_gemma4_embed_tokens_q4f16(input_path: Path, output_path: Path, block_size: int) -> dict:
    model = onnx.load(str(input_path))
    conversion_mode = "converted_to_fp16"
    try:
        model = onnx_float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            disable_shape_infer=True,
        )
    except ValueError as exc:
        if "already converted to float16" not in str(exc):
            raise
        conversion_mode = "already_fp16_ready"

    initializers = {{initializer.name: initializer for initializer in model.graph.initializer}}
    nodes = list(model.graph.node)
    rewritten_nodes = []
    new_initializers = []
    removed_initializers = set()
    quantized_embeddings = {{}}

    for node in nodes:
        if node.op_type != "Gather" or len(node.input) < 2 or node.input[0] not in initializers:
            rewritten_nodes.append(node)
            continue
        weight_name = node.input[0]
        if "embed_tokens" not in weight_name:
            rewritten_nodes.append(node)
            continue

        safe_prefix = weight_name.replace(".", "_").replace("/", "_")
        quant_tensor, scales_tensor, zero_points_tensor, metadata = _quantize_embedding_initializer(
            initializers[weight_name],
            block_size,
            safe_prefix,
        )
        quantized_embeddings[weight_name] = metadata
        new_initializers.extend([quant_tensor, scales_tensor, zero_points_tensor])
        removed_initializers.add(weight_name)

        float_output = f"{{node.output[0]}}_q4_float"
        rewritten_nodes.append(
            helper.make_node(
                "GatherBlockQuantized",
                [quant_tensor.name, node.input[1], scales_tensor.name, zero_points_tensor.name],
                [float_output],
                name=f"{{node.name}}_Q4" if node.name else "",
                domain="com.microsoft",
                bits=4,
                block_size=block_size,
                gather_axis=0,
                quantize_axis=1,
            )
        )
        rewritten_nodes.append(
            helper.make_node(
                "Cast",
                [float_output],
                [node.output[0]],
                name=f"{{node.name}}_Q4_Cast" if node.name else "",
                to=TensorProto.FLOAT16,
            )
        )

    if not quantized_embeddings:
        raise ValueError("embed_tokens session did not contain Gemma4 embedding Gather initializers to quantize")

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        [initializer for initializer in initializers.values() if initializer.name not in removed_initializers]
    )
    model.graph.initializer.extend(new_initializers)
    _ensure_ms_opset(model)
    fixed_elementwise_inputs = _harmonize_float16_elementwise_inputs(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_name(f"{{output_path.name}}_data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )

    return {{
        "conversion_mode": f"{{conversion_mode}}+gather_block_quantized_q4",
        "fixed_elementwise_inputs": fixed_elementwise_inputs,
        "quantized_embeddings": quantized_embeddings,
        "output_path": str(output_path),
        "external_data_path": str(external_data_path),
    }}


def _quantize_q4f16(input_path: Path, output_path: Path, block_size: int) -> dict:
    model = onnx.load(str(input_path))
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=block_size, is_symmetric=True)
    quantizer.process()
    q4_model = quantizer.model.model
    conversion_mode = "converted_to_fp16"
    try:
        q4f16_model = onnx_float16.convert_float_to_float16(
            q4_model,
            keep_io_types=False,
            disable_shape_infer=True,
        )
    except ValueError as exc:
        if "already converted to float16" not in str(exc):
            raise
        q4f16_model = q4_model
        conversion_mode = "already_fp16_ready"
    fixed_elementwise_inputs = _harmonize_float16_elementwise_inputs(q4f16_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_name(f"{{output_path.name}}_data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        q4f16_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )

    return {{
        "conversion_mode": conversion_mode,
        "fixed_elementwise_inputs": fixed_elementwise_inputs,
        "output_path": str(output_path),
        "external_data_path": str(external_data_path),
    }}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gemma4-quantize-runner")
    parser.add_argument("--input-dir", default={default_input_dir})
    parser.add_argument("--output-dir", default={default_output_dir})
    parser.add_argument("--report-path", default={default_report_path})
    parser.add_argument("--block-size", default={block_size}, type=int)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {{input_dir}}")

    results = {{}}
    missing_inputs: list[str] = []
    for session in CONTRACT["sessions"]:
        raw_path = input_dir / session["raw_filename"]
        if not raw_path.exists():
            missing_inputs.append(str(raw_path))
            continue
        quantized_path = output_dir / session["package_filename"]
        if session["name"] == "embed_tokens":
            result = _quantize_gemma4_embed_tokens_q4f16(raw_path, quantized_path, args.block_size)
        else:
            result = _quantize_q4f16(raw_path, quantized_path, args.block_size)
        results[session["name"]] = {{
            "input_path": str(raw_path),
            **result,
        }}

    report = {{
        "ok": not missing_inputs,
        "block_size": args.block_size,
        "contract": CONTRACT,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "results": results,
        "missing_inputs": missing_inputs,
    }}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing_inputs else 2


if __name__ == "__main__":
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxconverter_common import float16 as onnx_float16
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    raise SystemExit(main())
"""
