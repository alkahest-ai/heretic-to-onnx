from __future__ import annotations

import argparse
import json
from pathlib import Path

from roleplay_dataset_v2 import (
    ROLEPLAY_V2_DIR,
    detect_review_table_mode,
    load_conversations,
    read_review_table,
    review_rows_to_conversations,
    slim_review_rows_to_conversations,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROLEPLAY_V2_DIR / "review_table" / "batch-0001.tsv"),
        help="Input CSV/TSV review table path",
    )
    parser.add_argument(
        "--output",
        default=str(ROLEPLAY_V2_DIR / "approved_jsonl" / "batch-0001.jsonl"),
        help="Output approved JSONL path",
    )
    parser.add_argument(
        "--source-jsonl",
        default="",
        help="Required when compiling a slim review table; points at the source raw JSONL used to create the sheet",
    )
    parser.add_argument(
        "--include-non-approved",
        action="store_true",
        help="Compile rows regardless of status instead of requiring approved rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    review_rows = read_review_table(input_path)
    table_mode = detect_review_table_mode(list(review_rows[0].keys())) if review_rows else "full"
    if table_mode == "slim":
        if not args.source_jsonl:
            raise ValueError("--source-jsonl is required when compiling a slim review table")
        source_path = Path(args.source_jsonl).expanduser().resolve()
        source_conversations = load_conversations(source_path, approved_only=False)
        conversations, skipped = slim_review_rows_to_conversations(
            review_rows,
            source_conversations=source_conversations,
            approved_only=not args.include_non_approved,
        )
    else:
        conversations, skipped = review_rows_to_conversations(review_rows, approved_only=not args.include_non_approved)
    if not conversations:
        raise ValueError("no conversations qualified for export")

    write_jsonl(output_path, conversations)
    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "table_mode": table_mode,
        "source_jsonl": str(Path(args.source_jsonl).expanduser().resolve()) if args.source_jsonl else "",
        "conversations_written": len(conversations),
        **skipped,
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
