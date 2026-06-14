#!/usr/bin/env python3
"""Score multiple Rally/Heretic checkpoints against the same RP gate and refusal probe."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated name:model_spec entries, e.g. base:google/gemma-4-E2B-it,heretic:p-e-w/...",
    )
    parser.add_argument("--work-dir", default="/kaggle/working/rally-model-compare")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--refusal-probe-count", type=int, default=100)
    return parser


def _parse_models(raw: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid model entry {item!r}; expected name:model_spec")
        name, model_spec = item.split(":", 1)
        name = name.strip()
        model_spec = model_spec.strip()
        if not name or not model_spec:
            raise ValueError(f"invalid model entry {item!r}")
        specs.append((name, model_spec))
    if not specs:
        raise ValueError("--models did not contain any entries")
    return specs


def main(argv: list[str] | None = None) -> int:
    from scripts.alkahest_rp_scorecard import score_responses
    from scripts.kaggle_rally_e2b_scorecard import (
        _disk,
        _generate,
        _redacted,
        _run_refusal_probe,
        _write_report,
    )

    args = _parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else work_dir / "rally-model-compare-report.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": True,
        "work_dir": str(work_dir),
        "models": {},
        "ranking": [],
        "disk": {"start": _disk(work_dir)},
    }
    _write_report(report, report_path)

    for name, model_spec in _parse_models(args.models):
        entry: dict[str, Any] = {"model_spec": model_spec}
        responses = _generate(
            model_spec,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        score = score_responses(name, model_spec, responses)
        entry["scorecard"] = _redacted(score)
        if args.refusal_probe_count > 0:
            entry["refusal_probe"] = _run_refusal_probe(
                model_spec,
                prompt_count=args.refusal_probe_count,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        entry["disk"] = {"after_model": _disk(work_dir)}
        report["models"][name] = entry
        _write_report(report, report_path)

    report["ranking"] = sorted(
        [
            {
                "name": name,
                "total": item["scorecard"]["total"],
                "passed": item["scorecard"]["passed"],
                "false_refusal_count": (item.get("refusal_probe") or {}).get("false_refusal_count"),
                "false_refusal_rate": (item.get("refusal_probe") or {}).get("false_refusal_rate"),
            }
            for name, item in report["models"].items()
        ],
        key=lambda row: (row["passed"], row["total"], -(row["false_refusal_rate"] or 1.0)),
        reverse=True,
    )
    _write_report(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())