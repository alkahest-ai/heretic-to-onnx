#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.heretic_to_onnx.kaggle_heretic import (
    PRESETS,
    build_run_config,
    run_kaggle_heretic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a constrained non-interactive Heretic 2B proof for Kaggle notebooks."
    )
    parser.add_argument("--label", choices=sorted(PRESETS), required=True)
    parser.add_argument("--base-model-id", help="Override the preset base model ID")
    parser.add_argument(
        "--work-root",
        default=None,
        help="Kaggle work directory. Defaults to /kaggle/working/heretic-to-onnx/heretic/<label>",
    )
    parser.add_argument("--merged-output-dir", help="Where Heretic should save the merged checkpoint")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--prompt-rows", type=int, default=160)
    parser.add_argument("--eval-rows", type=int, default=80)
    parser.add_argument("--max-response-length", type=int, default=64)
    parser.add_argument(
        "--accelerator",
        choices=["t4x2", "single-gpu", "auto"],
        default="t4x2",
        help="Memory profile to write into Heretic config.toml.",
    )
    parser.add_argument("--heretic-exec", default="heretic")
    parser.add_argument(
        "--upload-merged-to",
        help="Optional Hugging Face repo ID to upload the merged checkpoint after validation.",
    )
    parser.add_argument(
        "--upload-private",
        action="store_true",
        help="Create/upload the merged checkpoint repo as private when --upload-merged-to is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write config/stdin files and print the report without launching Heretic.",
    )
    parser.add_argument(
        "--native-terminal-mode",
        action="store_true",
        help="Do not force KAGGLE_KERNEL_RUN_TYPE for notebook-style input prompts.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = build_run_config(
        label=args.label,
        base_model_id=args.base_model_id,
        work_root=Path(args.work_root) if args.work_root else None,
        merged_output_dir=Path(args.merged_output_dir) if args.merged_output_dir else None,
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        prompt_rows=args.prompt_rows,
        eval_rows=args.eval_rows,
        max_response_length=args.max_response_length,
        accelerator=args.accelerator,
    )
    report = run_kaggle_heretic(
        config,
        heretic_exec=args.heretic_exec,
        dry_run=args.dry_run,
        force_notebook_mode=not args.native_terminal_mode,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.ok and args.upload_merged_to and not args.dry_run:
        import os

        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("--upload-merged-to requires HF_TOKEN in the environment")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.upload_merged_to,
            repo_type="model",
            private=args.upload_private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.upload_merged_to,
            repo_type="model",
            folder_path=report.merged_output_dir,
            commit_message=f"Upload {args.label} Heretic merged checkpoint from Kaggle",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "uploaded_merged_checkpoint_to": args.upload_merged_to,
                    "source_dir": report.merged_output_dir,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
