from __future__ import annotations

import argparse
import json
from pathlib import Path

from roleplay_dataset_v2 import ROLEPLAY_V2_DIR, read_review_table, review_rows_to_conversations, write_jsonl


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
        "--include-non-approved",
        action="store_true",
        help="Compile rows regardless of status instead of requiring approved rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    review_rows = read_review_table(input_path)
    conversations, skipped = review_rows_to_conversations(review_rows, approved_only=not args.include_non_approved)
    if not conversations:
        raise ValueError("no conversations qualified for export")

    write_jsonl(output_path, conversations)
    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "conversations_written": len(conversations),
        **skipped,
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
