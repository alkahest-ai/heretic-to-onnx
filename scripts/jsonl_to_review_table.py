from __future__ import annotations

import argparse
import json
from pathlib import Path

from roleplay_dataset_v2 import ROLEPLAY_V2_DIR, conversation_to_review_rows, load_conversations, write_review_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROLEPLAY_V2_DIR / "generated_raw" / "batch-0001.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default=str(ROLEPLAY_V2_DIR / "review_table" / "batch-0001.tsv"),
        help="Output CSV/TSV review table path",
    )
    parser.add_argument("--default-status", default="generated", help="Status to set on exported rows")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    conversations = load_conversations(input_path, approved_only=False)
    review_rows = []
    for conversation in conversations:
        review_rows.extend(conversation_to_review_rows(conversation, default_status=args.default_status))

    write_review_table(output_path, review_rows)
    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "conversations": len(conversations),
        "review_rows": len(review_rows),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
