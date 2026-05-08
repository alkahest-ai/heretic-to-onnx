#!/usr/bin/env python3
"""Run the Rally/Gemma E2B RP promotion scorecard on Kaggle."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="", help="Directory containing the completed Rally E2B SFT output.")
    parser.add_argument("--artifact-name", default="rally-e2b-two-stage-sft")
    parser.add_argument("--work-dir", default="/kaggle/working/rally-e2b-scorecard")
    parser.add_argument("--report-path", default="", help="Optional JSON report path.")
    parser.add_argument("--direct-model-id", default="p-e-w/gemma-4-E2B-it-heretic-ara")
    parser.add_argument("--candidate-name", default="a100-b75")
    parser.add_argument("--stage-b-scale", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-total", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--keep-merged", action="store_true")
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Exit nonzero when the RP candidate does not clear the promotion gate.",
    )
    return parser


def _disk(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_gb": round(usage.free / 1024**3, 2),
        "total_gb": round(usage.total / 1024**3, 2),
    }


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _redacted(score: Any) -> dict[str, Any]:
    from scripts.alkahest_rp_scorecard import minor_boundary_diagnostics

    payload = asdict(score)
    responses = payload.get("responses")
    if isinstance(responses, dict) and "minor" in responses:
        payload.setdefault("diagnostics", {})["minor"] = minor_boundary_diagnostics(str(responses["minor"]))
        responses["minor"] = "[redacted; scored but not stored]"
    return payload


def _generate(model_spec: str | Path, *, max_new_tokens: int, temperature: float) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.alkahest_rp_scorecard import SMOKE_PROMPTS

    tokenizer = AutoTokenizer.from_pretrained(model_spec, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_spec, **model_kwargs)
    device = next(model.parameters()).device

    responses: dict[str, str] = {}
    try:
        for name, prompt in SMOKE_PROMPTS.items():
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(device)
            generate_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            with torch.no_grad():
                output = model.generate(**generate_kwargs)
            responses[name] = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            ).strip()
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return responses


def main(argv: list[str] | None = None) -> int:
    from scripts.alkahest_rp_scorecard import promotion_decision, score_responses
    from scripts.kaggle_rally_e2b_two_stage_export import _find_artifacts, _merge_scaled

    args = _parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else work_dir / "rally-e2b-scorecard-report.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "work_dir": str(work_dir),
        "direct_model_id": args.direct_model_id,
        "candidate_name": args.candidate_name,
        "stage_b_scale": args.stage_b_scale,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "scores": {},
        "promotion_decision": {},
        "disk": {"start": _disk(work_dir)},
    }
    _write_report(report, report_path)

    artifacts = _find_artifacts(args.artifact_dir, args.artifact_name)
    report["artifact_dir"] = str(artifacts)
    merged_dir = work_dir / f"{args.candidate_name}-merged"
    _merge_scaled(artifacts / "stage-a-merged", artifacts / "stage-b-adapter", merged_dir, args.stage_b_scale)
    report["merged_dir"] = str(merged_dir)
    report["disk"]["after_merge"] = _disk(work_dir)
    _write_report(report, report_path)

    direct_responses = _generate(
        args.direct_model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    direct_score = score_responses("direct-rally-e2b", args.direct_model_id, direct_responses)
    report["scores"]["direct"] = _redacted(direct_score)
    report["disk"]["after_direct"] = _disk(work_dir)
    _write_report(report, report_path)

    rp_responses = _generate(
        merged_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    rp_score = score_responses(f"rally-e2b-rp-{args.candidate_name}", merged_dir, rp_responses)
    decision = promotion_decision(
        direct_score,
        rp_score,
        min_total=args.min_total,
        min_margin=args.min_margin,
    )
    report["scores"]["rp"] = _redacted(rp_score)
    report["promotion_decision"] = asdict(decision)
    report["ok"] = bool(decision.promoted)
    report["disk"]["after_rp"] = _disk(work_dir)

    if not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)
        report["merged_dir_removed"] = True
        report["disk"]["after_cleanup"] = _disk(work_dir)

    _write_report(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["ok"] or not args.require_promotion else 1


if __name__ == "__main__":
    raise SystemExit(main())
