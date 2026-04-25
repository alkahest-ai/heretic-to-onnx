from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Manifest
from .gemma4_export_codegen import build_gemma4_export_contract, render_gemma4_export_runner
from .prepare import prepare_repos
from .runtime import run_command
from .workdir import resolve_work_dir


@dataclass(slots=True)
class ExportReport:
    ok: bool
    mode: str
    work_dir: str
    source_path: str
    base_path: str
    output_dir: str
    contract_path: str = ""
    runner_path: str = ""
    runner_report_path: str = ""
    commands: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_gemma4(
    manifest: Manifest,
    work_dir: str | Path | None = None,
    *,
    mode: str = "plan",
    python_exec: str = "python3",
    device: str = "cpu",
    torch_dtype: str = "auto",
    opset_version: int = 17,
) -> ExportReport:
    layout = resolve_work_dir(manifest, work_dir).ensure()
    output_dir = (layout.export_raw / "gemma4").resolve()
    runner_path = (layout.export_raw / "export_gemma4_runner.py").resolve()
    contract_path = (layout.reports / "gemma4-export-contract.json").resolve()
    runner_report_path = (layout.reports / "gemma4-export-execute.json").resolve()

    needs_full_source = mode == "execute"
    prepared = prepare_repos(
        manifest,
        layout.root,
        source_mode="full" if needs_full_source else "metadata",
    )
    contract = build_gemma4_export_contract(manifest, prepared.source_path)

    report = ExportReport(
        ok=contract.ok,
        mode=mode,
        work_dir=str(layout.root),
        source_path=prepared.source_path,
        base_path=prepared.base_path,
        output_dir=str(output_dir),
        contract_path=str(contract_path),
        runner_path=str(runner_path),
        runner_report_path=str(runner_report_path),
    )

    contract_path.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.notes.append(f"Gemma 4 export contract written to {contract_path}")
    report.warnings.extend(contract.warnings)

    if mode in {"script", "execute"}:
        runner_path.write_text(
            render_gemma4_export_runner(
                contract,
                source_path=prepared.source_path,
                base_path=prepared.base_path,
                output_dir=str(output_dir),
                report_path=str(runner_report_path),
                opset_version=opset_version,
            ),
            encoding="utf-8",
        )
        runner_path.chmod(0o755)
        report.notes.append(f"generated Gemma 4 export runner at {runner_path}")
        report.commands.append(
            [
                python_exec,
                str(runner_path),
                "--source-path",
                prepared.source_path,
                "--base-path",
                prepared.base_path,
                "--output-dir",
                str(output_dir),
                "--report-path",
                str(runner_report_path),
                "--device",
                device,
                "--torch-dtype",
                torch_dtype,
                "--opset-version",
                str(opset_version),
            ]
        )

    if mode == "plan":
        report.notes.extend(
            [
                "Plan mode validates the Gemma 4 browser contract and writes the export contract only.",
                "Use `--mode script` to generate a self-contained torch.onnx export runner without downloading weights.",
                "Use `--mode execute` to run the generated exporter in a Python environment that already has torch, transformers, and onnx installed.",
            ]
        )
    elif mode == "script":
        report.notes.append("Script mode does not download model weights; it only generates the runnable exporter.")
    elif mode == "execute":
        if not contract.ok:
            report.warnings.append("execute mode was requested, but the Gemma 4 export contract is not currently compatible")
        else:
            command = report.commands[-1]
            result = run_command(command, cwd=manifest.manifest_dir)
            report.notes.append(f"export runner report written to {runner_report_path}")
            if result.stdout.strip():
                report.notes.append("export runner completed successfully")
            if result.stderr.strip():
                report.warnings.append("export runner produced stderr output")
    else:
        raise ValueError(f"unsupported export mode: {mode}")

    report_path = layout.reports / "export-plan.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
