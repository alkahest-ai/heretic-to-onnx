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
    fp16_model, conversion_mode = _convert_to_fp16(model)
    external_data_path = _save_external_data_model(fp16_model, output_path)

    return {{
        "conversion_mode": conversion_mode,
        "output_path": str(output_path),
        "external_data_path": external_data_path,
    }}


def _package_dtype(package_filename: str) -> str:
    stem = Path(package_filename).stem
    for suffix in ("q4f16", "q4", "fp16"):
        if stem.endswith(f"_{{suffix}}"):
            return suffix
    raise ValueError(f"unsupported Qwen3.5 package dtype in filename: {{package_filename}}")


def _quantize_session(input_path: Path, output_path: Path, block_size: int, package_filename: str) -> dict:
    package_dtype = _package_dtype(package_filename)
    if package_dtype == "q4":
        return _quantize_q4(input_path, output_path, block_size)
    if package_dtype == "fp16":
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
        result = _quantize_session(raw_path, quantized_path, args.block_size, session["package_filename"])
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
    from onnxconverter_common import float16 as onnx_float16
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    raise SystemExit(main())
"""
