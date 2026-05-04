#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.heretic_to_onnx.config import load_manifest
from tools.heretic_to_onnx.convert import _default_opset_version, _raw_export_dir
from tools.heretic_to_onnx.export_gemma4 import export_gemma4
from tools.heretic_to_onnx.export_qwen3_5 import export_qwen3_5
from tools.heretic_to_onnx.package_repo import package_repo
from tools.heretic_to_onnx.prepare import prepare_repos
from tools.heretic_to_onnx.publish_hf import publish_hf
from tools.heretic_to_onnx.quantize_gemma4 import quantize_gemma4
from tools.heretic_to_onnx.quantize_qwen3_5 import quantize_qwen3_5
from tools.heretic_to_onnx.validate_repo import validate_package
from tools.heretic_to_onnx.workdir import resolve_work_dir


def _disk_snapshot(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / 1024**3, 2),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def _delete_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _trim_source_weights(source_path: Path) -> list[str]:
    removed: list[str] = []
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        for candidate in source_path.glob(pattern):
            removed.append(candidate.name)
            candidate.unlink(missing_ok=True)
    return removed


def _run_export(manifest, work_dir: Path, *, python_exec: str, device: str, torch_dtype: str, opset_version: int):
    if manifest.architecture == "gemma4_conditional_generation":
        return export_gemma4(
            manifest,
            work_dir,
            mode="execute",
            python_exec=python_exec,
            device=device,
            torch_dtype=torch_dtype,
            opset_version=opset_version,
        )
    if manifest.architecture == "qwen3_5_conditional_generation":
        return export_qwen3_5(
            manifest,
            work_dir,
            mode="execute",
            python_exec=python_exec,
            device=device,
            torch_dtype=torch_dtype,
            opset_version=opset_version,
        )
    raise ValueError(f"unsupported architecture family: {manifest.architecture}")


def _run_quantize(manifest, work_dir: Path, *, python_exec: str, block_size: int, raw_onnx_dir: Path, output_dir: Path):
    if manifest.architecture == "gemma4_conditional_generation":
        return quantize_gemma4(
            manifest,
            work_dir,
            mode="execute",
            python_exec=python_exec,
            raw_onnx_dir=raw_onnx_dir,
            output_dir=output_dir,
            block_size=block_size,
        )
    if manifest.architecture == "qwen3_5_conditional_generation":
        return quantize_qwen3_5(
            manifest,
            work_dir,
            mode="execute",
            python_exec=python_exec,
            raw_onnx_dir=raw_onnx_dir,
            output_dir=output_dir,
            block_size=block_size,
        )
    raise ValueError(f"unsupported architecture family: {manifest.architecture}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a staged browser export pipeline with optional cleanup between phases."
    )
    parser.add_argument("--config", required=True, help="Path to a rendered manifest YAML")
    parser.add_argument("--work-dir", help="Optional work directory")
    parser.add_argument("--output-dir", help="Optional packaged repo output directory")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable for generated runners")
    parser.add_argument("--export-device", default="cpu", help="Device string for export runners")
    parser.add_argument(
        "--export-torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="torch dtype for export runners",
    )
    parser.add_argument(
        "--opset-version",
        default=None,
        type=int,
        help="ONNX opset version; defaults to 21 for Gemma4 and 17 for Qwen3.5",
    )
    parser.add_argument("--block-size", default=32, type=int)
    parser.add_argument("--trim-source-weights", action="store_true")
    parser.add_argument("--delete-raw-onnx", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-workers", default=8, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = load_manifest(args.config)
    layout = resolve_work_dir(manifest, args.work_dir).ensure()
    raw_onnx_dir = _raw_export_dir(layout, manifest).resolve()
    quantized_dir = layout.export_quantized.resolve()
    package_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else layout.package_dir.resolve()

    report: dict[str, object] = {
        "ok": False,
        "config": str(Path(args.config).expanduser().resolve()),
        "work_dir": str(layout.root),
        "package_dir": str(package_dir),
        "disk": {
            "start": _disk_snapshot(layout.root),
        },
        "cleanup": {},
    }

    export_report = _run_export(
        manifest,
        layout.root,
        python_exec=args.python_exec,
        device=args.export_device,
        torch_dtype=args.export_torch_dtype,
        opset_version=args.opset_version or _default_opset_version(manifest),
    )
    report["export"] = export_report.to_dict()
    report["disk"]["after_export"] = _disk_snapshot(layout.root)
    if not export_report.ok:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    if args.trim_source_weights:
        removed = _trim_source_weights(Path(export_report.source_path))
        report["cleanup"]["trimmed_source_weights"] = removed
        report["disk"]["after_trim_source"] = _disk_snapshot(layout.root)

    quantize_report = _run_quantize(
        manifest,
        layout.root,
        python_exec=args.python_exec,
        block_size=args.block_size,
        raw_onnx_dir=raw_onnx_dir,
        output_dir=quantized_dir,
    )
    report["quantize"] = quantize_report.to_dict()
    report["disk"]["after_quantize"] = _disk_snapshot(layout.root)
    if not quantize_report.ok:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    if args.delete_raw_onnx:
        _delete_if_exists(raw_onnx_dir)
        report["cleanup"]["deleted_raw_onnx_dir"] = str(raw_onnx_dir)
        report["disk"]["after_delete_raw"] = _disk_snapshot(layout.root)

    prepared = prepare_repos(manifest, layout.root, source_mode="metadata")
    report["prepare"] = prepared.to_dict()

    package_report = package_repo(
        manifest,
        output_dir=package_dir,
        force=True,
        allow_missing_onnx=False,
        onnx_source_dir=quantized_dir,
        source_spec=prepared.source_path,
        base_spec=prepared.base_path,
    )
    report["package"] = package_report.to_dict()
    report["disk"]["after_package"] = _disk_snapshot(layout.root)
    if not package_report.ok:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    validation_report = validate_package(
        manifest,
        package_dir,
        strict_onnx=True,
        runtime_smoke=True,
    )
    report["validate"] = validation_report.to_dict()
    report["disk"]["after_validate"] = _disk_snapshot(layout.root)
    if not validation_report.ok:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    if args.publish:
        publish_report = publish_hf(
            manifest,
            package_dir=package_dir,
            private=args.private,
            num_workers=args.num_workers,
        )
        report["publish"] = publish_report.to_dict()
        report["disk"]["after_publish"] = _disk_snapshot(layout.root)
        if not publish_report.ok:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1

    report["ok"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
