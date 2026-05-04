#!/usr/bin/env python3
"""Kaggle two-stage SFT runner for Rally/Gemma E2B RP."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="p-e-w/gemma-4-E2B-it-heretic-ara")
    parser.add_argument("--work-dir", default="/kaggle/working/rally-e2b-two-stage-sft")
    parser.add_argument("--stage-a-max-steps", type=int, default=300)
    parser.add_argument("--stage-b-max-steps", type=int, default=450)
    parser.add_argument("--stage-a-repeats", type=int, default=18)
    parser.add_argument("--stage-b-boundary-repeats", type=int, default=80)
    parser.add_argument("--stage-b-adult-repeats", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--stage-b-learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-load-in-4bit", action="store_true")
    parser.add_argument("--dataset-num-proc", type=int, default=0)
    return parser


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=str(cwd))


def _train_command(
    args: argparse.Namespace,
    *,
    model_name: str,
    stage: str,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    merged_output_dir: Path,
    learning_rate: float,
    max_steps: int,
    manifest_path: Path,
    save_merged: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts/train_rally_unsloth.py"),
        "--model-name",
        model_name,
        "--train-file",
        str(train_file),
        "--val-file",
        str(val_file),
        "--dataset-manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--merged-output-dir",
        str(merged_output_dir),
        "--max-seq-length",
        str(args.max_seq_length),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(learning_rate),
        "--warmup-steps",
        str(args.warmup_steps),
        "--max-steps",
        str(max_steps),
        "--lora-rank",
        str(args.lora_rank),
        "--seed",
        str(args.seed),
        "--dataset-num-proc",
        str(args.dataset_num_proc),
    ]
    if save_merged:
        command.append("--save-merged")
    if args.no_load_in_4bit:
        command.append("--no-load-in-4bit")
    print(f"[train-command] {stage} max_steps={max_steps} lr={learning_rate}", flush=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    splits_dir = work_dir / "splits"
    stage_a_adapter = work_dir / "stage-a-adapter"
    stage_a_merged = work_dir / "stage-a-merged"
    stage_b_adapter = work_dir / "stage-b-adapter"
    stage_ab_merged = work_dir / "stage-ab-merged"
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts/prepare_alkahest_two_stage_sft.py"),
            "--output-dir",
            str(splits_dir),
            "--stage-a-repeats",
            str(args.stage_a_repeats),
            "--stage-b-boundary-repeats",
            str(args.stage_b_boundary_repeats),
            "--stage-b-adult-repeats",
            str(args.stage_b_adult_repeats),
            "--val-fraction",
            str(args.val_fraction),
        ],
        cwd=ROOT_DIR,
    )
    manifest_path = splits_dir / "manifest.json"

    _run(
        _train_command(
            args,
            model_name=args.model_name,
            stage="stage-a",
            train_file=splits_dir / "stage_a/train.jsonl",
            val_file=splits_dir / "stage_a/val.jsonl",
            output_dir=stage_a_adapter,
            merged_output_dir=stage_a_merged,
            learning_rate=args.learning_rate,
            max_steps=args.stage_a_max_steps,
            manifest_path=manifest_path,
            save_merged=True,
        ),
        cwd=ROOT_DIR,
    )
    _run(
        _train_command(
            args,
            model_name=str(stage_a_merged),
            stage="stage-b",
            train_file=splits_dir / "stage_b/train.jsonl",
            val_file=splits_dir / "stage_b/val.jsonl",
            output_dir=stage_b_adapter,
            merged_output_dir=stage_ab_merged,
            learning_rate=args.stage_b_learning_rate,
            max_steps=args.stage_b_max_steps,
            manifest_path=manifest_path,
            save_merged=False,
        ),
        cwd=ROOT_DIR,
    )

    report: dict[str, Any] = {
        "ok": True,
        "model_name": args.model_name,
        "work_dir": str(work_dir),
        "splits_dir": str(splits_dir),
        "stage_a_adapter": str(stage_a_adapter),
        "stage_a_merged": str(stage_a_merged),
        "stage_b_adapter": str(stage_b_adapter),
        "stage_ab_merged": str(stage_ab_merged),
        "stage_a_max_steps": args.stage_a_max_steps,
        "stage_b_max_steps": args.stage_b_max_steps,
        "stage_a_repeats": args.stage_a_repeats,
        "stage_b_boundary_repeats": args.stage_b_boundary_repeats,
        "stage_b_adult_repeats": args.stage_b_adult_repeats,
    }
    (work_dir / "rally-e2b-two-stage-sft-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
