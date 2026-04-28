from __future__ import annotations

import json
from pathlib import Path

from .qwen3_5_export_codegen import ExportContract


def render_qwen3_5_quantize_runner(
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
import shutil
from pathlib import Path

CONTRACT = {contract_literal}


def _save_external_data_model(model, output_path: Path) -> str:
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
    return str(external_data_path)


def _convert_to_fp16(model):
    conversion_mode = "converted_to_fp16"
    try:
        converted_model = onnx_float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            disable_shape_infer=True,
        )
    except ValueError as exc:
        if "already converted to float16" not in str(exc):
            raise
        converted_model = model
        conversion_mode = "already_fp16_ready"
    return converted_model, conversion_mode


def _tensor_elem_type(value_info) -> int | None:
    tensor_type = value_info.type.tensor_type
    return tensor_type.elem_type if tensor_type.HasField("elem_type") else None


def _copy_value_info(value_info):
    copied = onnx.ValueInfoProto()
    copied.CopyFrom(value_info)
    return copied


def _copy_model(model):
    copied = onnx.ModelProto()
    copied.CopyFrom(model)
    return copied


def _replace_node_inputs(model, old_name: str, new_name: str) -> None:
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name


def _replace_node_outputs_and_consumers(model, old_name: str, new_name: str) -> None:
    for node in model.graph.node:
        for index, output_name in enumerate(node.output):
            if output_name == old_name:
                node.output[index] = new_name
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name


def _float_tensor_to_fp16(tensor):
    if tensor.data_type != TensorProto.FLOAT:
        return tensor
    return numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16), tensor.name)


def _normalize_float_constants_to_fp16(model) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type == TensorProto.FLOAT:
            model.graph.initializer[index].CopyFrom(_float_tensor_to_fp16(initializer))

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                attr.t.CopyFrom(_float_tensor_to_fp16(attr.t))


def _wrap_fp16_model_with_float32_io(model, original_model) -> None:
    original_inputs = {{value.name: _copy_value_info(value) for value in original_model.graph.input}}
    original_outputs = {{value.name: _copy_value_info(value) for value in original_model.graph.output}}
    prefix_nodes = []
    suffix_nodes = []

    for value in model.graph.input:
        original = original_inputs.get(value.name)
        if original is None or _tensor_elem_type(original) != TensorProto.FLOAT:
            continue
        if _tensor_elem_type(value) != TensorProto.FLOAT16:
            continue

        external_name = value.name
        internal_name = f"{{external_name}}_fp16"
        _replace_node_inputs(model, external_name, internal_name)
        value.CopyFrom(original)
        prefix_nodes.append(
            helper.make_node(
                "Cast",
                [external_name],
                [internal_name],
                name=f"{{external_name}}_to_fp16",
                to=TensorProto.FLOAT16,
            )
        )

    for value in model.graph.output:
        original = original_outputs.get(value.name)
        if original is None or _tensor_elem_type(original) != TensorProto.FLOAT:
            continue
        if _tensor_elem_type(value) != TensorProto.FLOAT16:
            continue

        external_name = value.name
        internal_name = f"{{external_name}}_fp16"
        _replace_node_outputs_and_consumers(model, external_name, internal_name)
        value.CopyFrom(original)
        suffix_nodes.append(
            helper.make_node(
                "Cast",
                [internal_name],
                [external_name],
                name=f"{{external_name}}_to_fp32",
                to=TensorProto.FLOAT,
            )
        )

    if prefix_nodes:
        existing_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(prefix_nodes)
        model.graph.node.extend(existing_nodes)
    if suffix_nodes:
        model.graph.node.extend(suffix_nodes)


def _quantize_q4(input_path: Path, output_path: Path, block_size: int) -> dict:
    model = onnx.load(str(input_path))
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=block_size, is_symmetric=True)
    quantizer.process()
    external_data_path = _save_external_data_model(quantizer.model.model, output_path)

    return {{
        "conversion_mode": "matmul_nbits_q4",
        "output_path": str(output_path),
        "external_data_path": external_data_path,
    }}


def _pack_int4(values):
    if values.shape[-1] % 2:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)])
    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).astype(np.uint8)


def _quantize_gather_block_q4(input_path: Path, output_path: Path, block_size: int) -> dict:
    model = onnx.load(str(input_path))
    gather_nodes = [node for node in model.graph.node if node.op_type == "Gather"]
    if len(gather_nodes) != 1:
        raise ValueError(f"expected exactly one Gather node in embed_tokens session, got {{len(gather_nodes)}}")

    initializers = {{initializer.name: initializer for initializer in model.graph.initializer}}
    gather_node = gather_nodes[0]
    weight_inputs = [input_name for input_name in gather_node.input if input_name in initializers]
    if len(weight_inputs) != 1:
        raise ValueError(
            "expected exactly one Gather initializer input in embed_tokens session, "
            f"got {{weight_inputs}} from inputs {{list(gather_node.input)}}"
        )
    weight_name = weight_inputs[0]
    integer_inputs = [
        value.name
        for value in model.graph.input
        if _tensor_elem_type(value) in (TensorProto.INT32, TensorProto.INT64)
    ]
    if not integer_inputs:
        raise ValueError("embed_tokens session is missing an integer graph input for token ids")
    input_ids_name = integer_inputs[0]
    output_name = model.graph.output[0].name if len(model.graph.output) == 1 else gather_node.output[0]
    if weight_name not in initializers:
        raise ValueError(f"Gather weight initializer was not found: {{weight_name}}")

    weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
    if weight.ndim != 2:
        raise ValueError(f"expected 2D embedding weight, got shape {{weight.shape}}")
    vocab_size, hidden_size = weight.shape
    if hidden_size % block_size:
        raise ValueError(f"embedding hidden size {{hidden_size}} is not divisible by block size {{block_size}}")

    num_blocks = hidden_size // block_size
    blocks = weight.reshape(vocab_size, num_blocks, block_size)
    mins = blocks.min(axis=-1)
    maxs = blocks.max(axis=-1)
    scales = (maxs - mins) / 15.0
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    zero_points = np.rint(-mins / scales).clip(0, 15).astype(np.uint8)
    quantized = np.rint(blocks / scales[..., None] + zero_points[..., None]).clip(0, 15).astype(np.uint8)

    quant_name = weight_name.replace(".", "_") + "_quant"
    scales_name = weight_name.replace(".", "_") + "_scales"
    zero_points_name = weight_name.replace(".", "_") + "_zp"
    quant_tensor = numpy_helper.from_array(_pack_int4(quantized.reshape(vocab_size, hidden_size)), quant_name)
    scales_tensor = numpy_helper.from_array(scales, scales_name)
    zero_points_tensor = numpy_helper.from_array(_pack_int4(zero_points), zero_points_name)

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    for output in outputs:
        if output.name == output_name:
            output.type.tensor_type.elem_type = TensorProto.FLOAT

    graph = helper.make_graph(
        [
            helper.make_node(
                "GatherBlockQuantized",
                [quant_name, input_ids_name, scales_name, zero_points_name],
                [output_name],
                domain="com.microsoft",
                bits=4,
                block_size=block_size,
                gather_axis=0,
                quantize_axis=1,
            )
        ],
        model.graph.name or "embed_tokens_q4",
        inputs,
        outputs,
        [quant_tensor, scales_tensor, zero_points_tensor],
    )
    quantized_model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    quantized_model.ir_version = min(model.ir_version, 10) if model.ir_version else 10
    external_data_path = _save_external_data_model(quantized_model, output_path)

    return {{
        "conversion_mode": "gather_block_quantized_q4",
        "output_path": str(output_path),
        "external_data_path": external_data_path,
        "block_size": block_size,
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
    }}


def _quantize_q4f16(input_path: Path, output_path: Path, block_size: int) -> dict:
    model = onnx.load(str(input_path))
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=block_size, is_symmetric=True)
    quantizer.process()
    q4f16_model, conversion_mode = _convert_to_fp16(quantizer.model.model)
    external_data_path = _save_external_data_model(q4f16_model, output_path)

    return {{
        "conversion_mode": conversion_mode,
        "output_path": str(output_path),
        "external_data_path": external_data_path,
    }}


def _quantize_fp16(input_path: Path, output_path: Path) -> dict:
    model = onnx.load(str(input_path))
    original_model = _copy_model(model)
    conversion_mode = "converted_to_fp16_wrapped_float32_io"
    try:
        fp16_model = onnx_float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            disable_shape_infer=True,
        )
        _normalize_float_constants_to_fp16(fp16_model)
        _wrap_fp16_model_with_float32_io(fp16_model, original_model)
    except ValueError as exc:
        if "already converted to float16" not in str(exc):
            raise
        fp16_model = model
        conversion_mode = "already_fp16_ready"
    external_data_path = _save_external_data_model(fp16_model, output_path)

    return {{
        "conversion_mode": conversion_mode,
        "output_path": str(output_path),
        "external_data_path": external_data_path,
    }}


def _copy_official_onnx_session(output_path: Path, package_filename: str, official_onnx_repo: str) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_path = hf_hub_download(
        official_onnx_repo,
        f"onnx/{{package_filename}}",
    )
    external_path = hf_hub_download(
        official_onnx_repo,
        f"onnx/{{package_filename}}_data",
    )
    shutil.copy2(session_path, output_path)
    external_output_path = output_path.with_name(f"{{output_path.name}}_data")
    shutil.copy2(external_path, external_output_path)

    return {{
        "conversion_mode": "copied_official_qwen35_webgpu_session",
        "official_onnx_repo": official_onnx_repo,
        "output_path": str(output_path),
        "external_data_path": str(external_output_path),
    }}


def _package_dtype(package_filename: str) -> str:
    stem = Path(package_filename).stem
    for suffix in ("q4f16", "q4", "fp16"):
        if stem.endswith(f"_{{suffix}}"):
            return suffix
    raise ValueError(f"unsupported Qwen3.5 package dtype in filename: {{package_filename}}")


def _quantize_session(
    input_path: Path,
    output_path: Path,
    block_size: int,
    package_filename: str,
    official_onnx_repo: str,
) -> dict:
    package_dtype = _package_dtype(package_filename)
    if package_dtype == "q4":
        if Path(package_filename).stem.startswith("embed_tokens_"):
            return _quantize_gather_block_q4(input_path, output_path, block_size)
        return _quantize_q4(input_path, output_path, block_size)
    if package_dtype == "fp16":
        if Path(package_filename).stem.startswith("vision_encoder_"):
            return _copy_official_onnx_session(output_path, package_filename, official_onnx_repo)
        return _quantize_fp16(input_path, output_path)
    if package_dtype == "q4f16":
        return _quantize_q4f16(input_path, output_path, block_size)
    raise ValueError(f"unsupported Qwen3.5 package dtype: {{package_dtype}}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen3_5-quantize-runner")
    parser.add_argument("--input-dir", default={default_input_dir})
    parser.add_argument("--output-dir", default={default_output_dir})
    parser.add_argument("--report-path", default={default_report_path})
    parser.add_argument("--block-size", default={block_size}, type=int)
    parser.add_argument("--official-onnx-repo", default="onnx-community/Qwen3.5-0.8B-ONNX-OPT")
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
        result = _quantize_session(
            raw_path,
            quantized_path,
            args.block_size,
            session["package_filename"],
            args.official_onnx_repo,
        )
        results[session["name"]] = {{
            "input_path": str(raw_path),
            **result,
        }}

    report = {{
        "ok": not missing_inputs,
        "block_size": args.block_size,
        "contract": CONTRACT,
        "input_dir": str(input_dir),
        "official_onnx_repo": args.official_onnx_repo,
        "output_dir": str(output_dir),
        "results": results,
        "missing_inputs": missing_inputs,
    }}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing_inputs else 2


if __name__ == "__main__":
    import numpy as np
    import onnx
    from huggingface_hub import hf_hub_download
    from onnx import TensorProto, helper, numpy_helper
    from onnxconverter_common import float16 as onnx_float16
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    raise SystemExit(main())
"""
