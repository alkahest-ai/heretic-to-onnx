from __future__ import annotations

import argparse
import json

from .bootstrap import bootstrap_report
from .config import load_manifest
from .convert import run_convert
from .export_gemma4 import export_gemma4
from .export_qwen3_5 import export_qwen3_5
from .gemma4_opt_transplant import build_optimized_gemma4_text_package
from .inspect import inspect_manifest
from .package_repo import package_repo
from .publish_hf import publish_hf, publish_model_card_hf, write_model_card
from .prepare import prepare_repos
from .quantize_gemma4 import quantize_gemma4
from .quantize_qwen3_5 import quantize_qwen3_5
from .render_manifest import render_manifest
from .validate_repo import validate_package


def _dump_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heretic-to-onnx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap", help="Report which external commands the scaffold will use")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect source/base repos against the manifest")
    inspect_parser.add_argument("--config", required=True, help="Path to manifest YAML")

    prepare_parser = subparsers.add_parser("prepare", help="Resolve/download source and base repos into a work dir")
    prepare_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    prepare_parser.add_argument("--work-dir", help="Optional work directory")
    prepare_parser.add_argument(
        "--source-mode",
        choices=["metadata", "full", "skip"],
        default="metadata",
        help="How much of the remote source repo to download",
    )

    render_manifest_parser = subparsers.add_parser(
        "render-manifest",
        help="Render a runtime manifest YAML from a template manifest plus overrides",
    )
    render_manifest_parser.add_argument("--template", required=True, help="Template manifest YAML")
    render_manifest_parser.add_argument("--output", required=True, help="Where to write the rendered manifest YAML")
    render_manifest_parser.add_argument("--source-model-id", help="Optional source model override")
    render_manifest_parser.add_argument("--base-model-id", help="Optional base model override")
    render_manifest_parser.add_argument("--target-repo-id", help="Optional target repo override")

    package_parser = subparsers.add_parser("package", help="Assemble the browser package metadata layout")
    package_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    package_parser.add_argument("--output-dir", help="Where to write the packaged repo")
    package_parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    package_parser.add_argument("--onnx-source-dir", help="Directory containing exported ONNX artifacts")
    package_parser.add_argument(
        "--require-onnx",
        action="store_true",
        help="Fail packaging if ONNX artifacts are not already present",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a packaged repo")
    validate_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    validate_parser.add_argument("--package-dir", required=True, help="Path to a packaged repo directory")
    validate_parser.add_argument("--strict-onnx", action="store_true", help="Treat missing ONNX files as errors")
    validate_parser.add_argument(
        "--skip-runtime-smoke",
        action="store_true",
        help="Skip packaged ONNX session instantiation checks",
    )

    convert_parser = subparsers.add_parser("convert", help="Run inspect -> package -> validate")
    convert_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    convert_parser.add_argument("--output-dir", help="Where to write the packaged repo")
    convert_parser.add_argument("--work-dir", help="Optional work directory")
    convert_parser.add_argument("--onnx-source-dir", help="Directory containing exported ONNX artifacts")
    convert_parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    convert_parser.add_argument("--strict-onnx", action="store_true", help="Require final ONNX artifacts")
    convert_parser.add_argument(
        "--skip-runtime-smoke",
        action="store_true",
        help="Skip packaged ONNX session instantiation checks",
    )
    convert_parser.add_argument(
        "--export-mode",
        choices=["plan", "script", "execute"],
        default="plan",
        help="Whether convert should only plan export, generate the exporter, or execute it",
    )
    convert_parser.add_argument(
        "--quantize-mode",
        choices=["plan", "script", "execute"],
        default="plan",
        help="Whether convert should only plan quantization, generate the quantizer, or execute it",
    )
    convert_parser.add_argument(
        "--python-exec",
        default="python3",
        help="Python executable to use for execute modes",
    )
    convert_parser.add_argument(
        "--export-device",
        default="cpu",
        help="Device string passed to generated export runners in execute/script modes",
    )
    convert_parser.add_argument(
        "--export-torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="torch dtype passed to generated export runners in execute/script modes",
    )
    convert_parser.add_argument(
        "--opset-version",
        default=None,
        type=int,
        help="ONNX opset version for export; defaults to 21 for Gemma4 and 17 for Qwen3.5",
    )
    convert_parser.add_argument(
        "--block-size",
        default=32,
        type=int,
        help="Block size for q4f16 quantization when convert drives quantize-gemma4",
    )

    export_parser = subparsers.add_parser("export-gemma4", help="Plan the Gemma 4 ONNX export stage")
    export_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    export_parser.add_argument("--work-dir", help="Optional work directory")
    export_parser.add_argument(
        "--mode",
        choices=["plan", "script", "execute"],
        default="plan",
        help="Whether to only inspect, generate the export runner, or execute it",
    )
    export_parser.add_argument(
        "--python-exec",
        default="python3",
        help="Python executable to use for execute mode",
    )
    export_parser.add_argument(
        "--device",
        default="cpu",
        help="Device string passed to the generated export runner",
    )
    export_parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="torch dtype passed to the generated export runner",
    )
    export_parser.add_argument(
        "--opset-version",
        default=21,
        type=int,
        help="ONNX opset version for the generated Gemma 4 exporter",
    )

    quantize_parser = subparsers.add_parser(
        "quantize-gemma4",
        help="Plan, script, or execute the Gemma 4 ONNX q4f16 quantization stage",
    )
    quantize_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    quantize_parser.add_argument("--work-dir", help="Optional work directory")
    quantize_parser.add_argument(
        "--mode",
        choices=["plan", "script", "execute"],
        default="plan",
        help="Whether to only inspect, generate the quantize runner, or execute it",
    )
    quantize_parser.add_argument(
        "--python-exec",
        default="python3",
        help="Python executable to use for execute mode",
    )
    quantize_parser.add_argument(
        "--raw-onnx-dir",
        help="Directory containing raw split-session ONNX files from export-gemma4",
    )
    quantize_parser.add_argument(
        "--output-dir",
        help="Directory to write q4f16 ONNX artifacts into",
    )
    quantize_parser.add_argument(
        "--block-size",
        default=32,
        type=int,
        help="Block size for MatMul int4 quantization",
    )

    optimize_gemma4_parser = subparsers.add_parser(
        "optimize-gemma4-text-package",
        help="Replace a Gemma4 text package decoder with the reference optimized WebGPU graph",
    )
    optimize_gemma4_parser.add_argument("--source-dir", required=True, help="Local merged HF checkpoint directory")
    optimize_gemma4_parser.add_argument("--template-dir", required=True, help="Reference Gemma4 ONNX package or onnx dir")
    optimize_gemma4_parser.add_argument("--package-dir", required=True, help="Packaged repo directory to update in place")
    optimize_gemma4_parser.add_argument("--block-size", default=32, type=int, help="MatMulNBits block size")

    qwen_export_parser = subparsers.add_parser("export-qwen3_5", help="Plan the Qwen3.5 ONNX export stage")
    qwen_export_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    qwen_export_parser.add_argument("--work-dir", help="Optional work directory")
    qwen_export_parser.add_argument(
        "--mode",
        choices=["plan", "script", "execute"],
        default="plan",
        help="Whether to only inspect, generate the export runner, or execute it",
    )
    qwen_export_parser.add_argument(
        "--python-exec",
        default="python3",
        help="Python executable to use for execute mode",
    )
    qwen_export_parser.add_argument(
        "--device",
        default="cpu",
        help="Device string passed to the generated export runner",
    )
    qwen_export_parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="torch dtype passed to the generated export runner",
    )
    qwen_export_parser.add_argument(
        "--opset-version",
        default=17,
        type=int,
        help="ONNX opset version for the generated Qwen3.5 exporter",
    )

    qwen_quantize_parser = subparsers.add_parser(
        "quantize-qwen3_5",
        help="Plan, script, or execute the Qwen3.5 ONNX q4f16 quantization stage",
    )
    qwen_quantize_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    qwen_quantize_parser.add_argument("--work-dir", help="Optional work directory")
    qwen_quantize_parser.add_argument(
        "--mode",
        choices=["plan", "script", "execute"],
        default="plan",
        help="Whether to only inspect, generate the quantize runner, or execute it",
    )
    qwen_quantize_parser.add_argument(
        "--python-exec",
        default="python3",
        help="Python executable to use for execute mode",
    )
    qwen_quantize_parser.add_argument(
        "--raw-onnx-dir",
        help="Directory containing raw split-session ONNX files from export-qwen3_5",
    )
    qwen_quantize_parser.add_argument(
        "--output-dir",
        help="Directory to write q4f16 ONNX artifacts into",
    )
    qwen_quantize_parser.add_argument(
        "--block-size",
        default=32,
        type=int,
        help="Block size for MatMul int4 quantization",
    )

    publish_parser = subparsers.add_parser(
        "publish-hf",
        help="Create the target Hugging Face model repo if needed and upload a packaged folder",
    )
    publish_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    publish_parser.add_argument("--package-dir", required=True, help="Path to the packaged repo directory")
    publish_parser.add_argument("--repo-id", help="Override the target repo id from the manifest")
    publish_parser.add_argument("--private", action="store_true", help="Create the target repo as private")
    publish_parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="Worker count for hf upload-large-folder",
    )

    write_model_card_parser = subparsers.add_parser(
        "write-model-card",
        help="Render the autogenerated README/model card to a local file",
    )
    write_model_card_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    write_model_card_parser.add_argument("--output", required=True, help="Path to write README markdown")
    write_model_card_parser.add_argument("--repo-id", help="Override the target repo id from the manifest")

    publish_model_card_parser = subparsers.add_parser(
        "publish-model-card-hf",
        help="Render and upload only the README/model card to Hugging Face",
    )
    publish_model_card_parser.add_argument("--config", required=True, help="Path to manifest YAML")
    publish_model_card_parser.add_argument("--output", required=True, help="Temporary local README markdown path")
    publish_model_card_parser.add_argument("--repo-id", help="Override the target repo id from the manifest")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        _dump_json(bootstrap_report().to_dict())
        return 0

    if args.command == "render-manifest":
        report = render_manifest(
            args.template,
            args.output,
            source_model_id=args.source_model_id,
            base_model_id=args.base_model_id,
            target_repo_id=args.target_repo_id,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "optimize-gemma4-text-package":
        report = build_optimized_gemma4_text_package(
            source_dir=args.source_dir,
            template_dir=args.template_dir,
            package_dir=args.package_dir,
            block_size=args.block_size,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    manifest = load_manifest(args.config)

    if args.command == "inspect":
        report = inspect_manifest(manifest)
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "prepare":
        report = prepare_repos(
            manifest,
            work_dir=args.work_dir,
            source_mode=args.source_mode,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "package":
        report = package_repo(
            manifest,
            output_dir=args.output_dir,
            force=args.force,
            allow_missing_onnx=not args.require_onnx,
            onnx_source_dir=args.onnx_source_dir,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "validate":
        report = validate_package(
            manifest,
            args.package_dir,
            strict_onnx=args.strict_onnx,
            runtime_smoke=not args.skip_runtime_smoke,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "convert":
        report = run_convert(
            manifest,
            output_dir=args.output_dir,
            force=args.force,
            strict_onnx=args.strict_onnx,
            runtime_smoke=False if args.skip_runtime_smoke else None,
            work_dir=args.work_dir,
            onnx_source_dir=args.onnx_source_dir,
            export_mode=args.export_mode,
            quantize_mode=args.quantize_mode,
            python_exec=args.python_exec,
            export_device=args.export_device,
            export_torch_dtype=args.export_torch_dtype,
            opset_version=args.opset_version,
            block_size=args.block_size,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "export-gemma4":
        report = export_gemma4(
            manifest,
            getattr(args, "work_dir", None),
            mode=args.mode,
            python_exec=args.python_exec,
            device=args.device,
            torch_dtype=args.torch_dtype,
            opset_version=args.opset_version,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "quantize-gemma4":
        report = quantize_gemma4(
            manifest,
            getattr(args, "work_dir", None),
            mode=args.mode,
            python_exec=args.python_exec,
            raw_onnx_dir=args.raw_onnx_dir,
            output_dir=args.output_dir,
            block_size=args.block_size,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "export-qwen3_5":
        report = export_qwen3_5(
            manifest,
            getattr(args, "work_dir", None),
            mode=args.mode,
            python_exec=args.python_exec,
            device=args.device,
            torch_dtype=args.torch_dtype,
            opset_version=args.opset_version,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "quantize-qwen3_5":
        report = quantize_qwen3_5(
            manifest,
            getattr(args, "work_dir", None),
            mode=args.mode,
            python_exec=args.python_exec,
            raw_onnx_dir=args.raw_onnx_dir,
            output_dir=args.output_dir,
            block_size=args.block_size,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "publish-hf":
        report = publish_hf(
            manifest,
            package_dir=args.package_dir,
            repo_id=args.repo_id,
            private=args.private,
            num_workers=args.num_workers,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    if args.command == "write-model-card":
        output_path = write_model_card(
            manifest,
            output_path=args.output,
            repo_id=args.repo_id,
        )
        _dump_json(
            {
                "ok": True,
                "output_path": str(output_path),
                "repo_id": args.repo_id or manifest.target_repo_id,
            }
        )
        return 0

    if args.command == "publish-model-card-hf":
        report = publish_model_card_hf(
            manifest,
            output_path=args.output,
            repo_id=args.repo_id,
        )
        _dump_json(report.to_dict())
        return 0 if report.ok else 2

    parser.error(f"unknown command: {args.command}")
    return 2
