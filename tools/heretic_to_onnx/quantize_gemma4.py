from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest
from .gemma4_export_codegen import build_gemma4_export_contract
from .gemma4_quantize_codegen import render_gemma4_quantize_runner
from .prepare import prepare_repos
from .runtime import run_command
from .workdir import resolve_work_dir


@dataclass(slots=True)
class QuantizeReport:
    ok: bool
    mode: str
    work_dir: str
    source_path: str
    base_path: str
    input_dir: str
    output_dir: str
    contract_path: str = ""
    runner_path: str = ""
    runner_report_path: str = ""
    commands: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _all_raw_session_files_exist(raw_dir: Path, contract: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for session in contract["sessions"]:
        raw_filename = session["raw_filename"]
        if not (raw_dir / raw_filename).exists():
            missing.append(str(raw_dir / raw_filename))
    return not missing, missing


def quantize_gemma4(
    manifest: Manifest,
    work_dir: str | Path | None = None,
    *,
    mode: str = "plan",
    python_exec: str = "python3",
    raw_onnx_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    block_size: int = 32,
) -> QuantizeReport:
    layout = resolve_work_dir(manifest, work_dir).ensure()
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else layout.export_quantized.resolve()
    resolved_input_dir = (
        Path(raw_onnx_dir).expanduser().resolve()
        if raw_onnx_dir
        else (layout.export_raw / "gemma4").resolve()
    )
    runner_path = (layout.export_quantized / "quantize_gemma4_runner.py").resolve()
    contract_path = (layout.reports / "gemma4-export-contract.json").resolve()
    runner_report_path = (layout.reports / "gemma4-quantize-execute.json").resolve()

    prepared = prepare_repos(manifest, layout.root, source_mode="metadata")
    contract = build_gemma4_export_contract(manifest, prepared.source_path)
    contract_dict = contract.to_dict()
    contract_path.write_text(json.dumps(contract_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = QuantizeReport(
        ok=contract.ok,
        mode=mode,
        work_dir=str(layout.root),
        source_path=prepared.source_path,
        base_path=prepared.base_path,
        input_dir=str(resolved_input_dir),
        output_dir=str(resolved_output_dir),
        contract_path=str(contract_path),
        runner_path=str(runner_path),
        runner_report_path=str(runner_report_path),
    )
    report.warnings.extend(contract.warnings)
    report.notes.append(f"Gemma 4 export contract written to {contract_path}")

    has_raw_inputs, missing_inputs = _all_raw_session_files_exist(resolved_input_dir, contract_dict)
    if not has_raw_inputs:
        report.warnings.append("raw ONNX export is incomplete; missing session files: " + ", ".join(missing_inputs))

    if mode in {"script", "execute"}:
        runner_path.write_text(
            render_gemma4_quantize_runner(
                contract,
                input_dir=str(resolved_input_dir),
                output_dir=str(resolved_output_dir),
                report_path=str(runner_report_path),
                block_size=block_size,
            ),
            encoding="utf-8",
        )
        runner_path.chmod(0o755)
        report.notes.append(f"generated Gemma 4 quantize runner at {runner_path}")
        report.commands.append(
            [
                python_exec,
                str(runner_path),
                "--input-dir",
                str(resolved_input_dir),
                "--output-dir",
                str(resolved_output_dir),
                "--report-path",
                str(runner_report_path),
                "--block-size",
                str(block_size),
            ]
        )

    if mode == "plan":
        report.notes.extend(
            [
                "Plan mode validates the Gemma 4 quantization contract and expected session file mapping only.",
                "Use `--mode script` to generate a q4f16 quantization runner.",
                "Use `--mode execute` to run q4f16 quantization in a Python environment with onnx, onnxruntime, and onnxconverter_common installed.",
            ]
        )
    elif mode == "script":
        report.notes.append("Script mode does not quantize locally; it only generates the runnable quantizer.")
    elif mode == "execute":
        if not contract.ok:
            report.warnings.append("execute mode was requested, but the Gemma 4 contract is not currently compatible")
        elif not has_raw_inputs:
            report.ok = False
            report.warnings.append("execute mode was requested, but raw ONNX session files are missing")
        else:
            command = report.commands[-1]
            result = run_command(command, cwd=manifest.manifest_dir)
            report.notes.append(f"quantize runner report written to {runner_report_path}")
            if result.stdout.strip():
                report.notes.append("quantize runner completed successfully")
            if result.stderr.strip():
                report.warnings.append("quantize runner produced stderr output")
    else:
        raise ValueError(f"unsupported quantize mode: {mode}")

    report_path = layout.reports / "quantize-plan.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
