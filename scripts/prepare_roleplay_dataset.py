from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from roleplay_dataset_v2 import (
    ROLEPLAY_V2_DIR,
    clean_conversation_for_sft,
    lint_conversations,
    load_conversations,
    to_minimal_chat_rows,
    validate_conversation,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROLEPLAY_V2_DIR / "corpus.jsonl"),
        help="Input JSONL or review-table dataset path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROLEPLAY_V2_DIR / "splits"),
        help="Output directory for train/val splits",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--assistant-line-threshold", type=int, default=2, help="Warning threshold for repeated assistant lines")
    parser.add_argument(
        "--assistant-skeleton-threshold",
        type=int,
        default=3,
        help="Warning threshold for repeated assistant skeletons",
    )
    parser.add_argument(
        "--conversation-shape-threshold",
        type=int,
        default=4,
        help="Warning threshold for repeated conversation shapes",
    )
    parser.add_argument(
        "--clean-assistant-style",
        action="store_true",
        help="Remove generator scaffold phrases from assistant turns before splitting.",
    )
    parser.add_argument(
        "--assistant-direct-dialogue",
        action="store_true",
        help="When cleaning assistant style, train only on quoted assistant dialogue if available.",
    )
    parser.add_argument(
        "--fail-on-style-markers",
        action="store_true",
        help="Fail if assistant meta-style markers remain after optional cleaning.",
    )
    parser.add_argument(
        "--drop-invalid-after-cleaning",
        action="store_true",
        help="Drop conversations that become invalid after assistant-style cleaning.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a compact manifest summary instead of the full lint warning list.",
    )
    parser.add_argument("--fail-on-warnings", action="store_true", help="Treat lint warnings as fatal")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    corpus_manifest_path = input_path.with_name("corpus-manifest.json")

    rows = load_conversations(input_path)
    if not rows:
        raise ValueError("dataset is empty")
    dropped_after_cleaning = 0
    if args.clean_assistant_style:
        cleaned_rows = [
            clean_conversation_for_sft(row, direct_dialogue=args.assistant_direct_dialogue)
            for row in rows
        ]
        if args.drop_invalid_after_cleaning:
            rows = []
            for index, row in enumerate(cleaned_rows, start=1):
                try:
                    validate_conversation(row, index)
                except ValueError:
                    dropped_after_cleaning += 1
                    continue
                rows.append(row)
        else:
            rows = cleaned_rows
        if not rows:
            raise ValueError("all rows were dropped after assistant-style cleaning")

    lint_report = lint_conversations(
        rows,
        assistant_line_threshold=args.assistant_line_threshold,
        assistant_skeleton_threshold=args.assistant_skeleton_threshold,
        conversation_shape_threshold=args.conversation_shape_threshold,
    )
    if lint_report["errors"]:
        raise ValueError("dataset lint failed:\n- " + "\n- ".join(lint_report["errors"]))
    remaining_style_markers = lint_report["stats"].get("assistant_style_markers", {})
    if args.fail_on_style_markers and remaining_style_markers:
        raise ValueError(
            "assistant style markers remain after preparation: "
            + json.dumps(remaining_style_markers, sort_keys=True)
        )
    if args.fail_on_warnings and lint_report["warnings"]:
        raise ValueError("dataset lint warnings treated as fatal:\n- " + "\n- ".join(lint_report["warnings"]))

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.val_fraction))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        raise ValueError("validation split consumed all rows")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "train.minimal.jsonl", to_minimal_chat_rows(train_rows))
    write_jsonl(output_dir / "val.minimal.jsonl", to_minimal_chat_rows(val_rows))
    corpus_manifest = None
    if corpus_manifest_path.exists():
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "rows_total": len(rows),
        "rows_train": len(train_rows),
        "rows_val": len(val_rows),
        "train_minimal_path": str(output_dir / "train.minimal.jsonl"),
        "val_minimal_path": str(output_dir / "val.minimal.jsonl"),
        "source_version": (corpus_manifest or {}).get("source_version", "roleplay_v2"),
        "corpus_manifest_path": str(corpus_manifest_path),
        "corpus_manifest": corpus_manifest,
        "clean_assistant_style": args.clean_assistant_style,
        "assistant_direct_dialogue": args.assistant_direct_dialogue,
        "dropped_after_cleaning": dropped_after_cleaning,
        "lint": lint_report,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.summary_only:
        summary = {
            "input_path": manifest["input_path"],
            "output_dir": manifest["output_dir"],
            "rows_total": manifest["rows_total"],
            "rows_train": manifest["rows_train"],
            "rows_val": manifest["rows_val"],
            "clean_assistant_style": manifest["clean_assistant_style"],
            "assistant_direct_dialogue": manifest["assistant_direct_dialogue"],
            "dropped_after_cleaning": manifest["dropped_after_cleaning"],
            "lint_ok": lint_report["ok"],
            "lint_error_count": len(lint_report["errors"]),
            "lint_warning_count": len(lint_report["warnings"]),
            "assistant_style_markers": lint_report["stats"].get("assistant_style_markers", {}),
        }
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
