from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from roleplay_dataset_v2 import ROLEPLAY_V2_DIR, load_conversations, read_review_table, review_rows_to_conversations


def _load_sampleable_rows(path: Path, *, approved_only: bool, status: str) -> list[dict]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        review_rows = read_review_table(path)
        filtered_rows = review_rows
        if status:
            filtered_rows = [row for row in review_rows if row.get("status", "").strip().lower() == status.lower()]
        conversations, _ = review_rows_to_conversations(filtered_rows, approved_only=approved_only)
        return conversations
    rows = load_conversations(path, approved_only=approved_only)
    if status:
        rows = [row for row in rows if str(row.get("status", "")).strip().lower() == status.lower()]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROLEPLAY_V2_DIR / "approved_jsonl"),
        help="JSONL file, CSV/TSV review table, or directory of JSONL files",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of conversations to print")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--tag", default="", help="Optional tag filter")
    parser.add_argument("--lane", default="", help="Optional lane filter")
    parser.add_argument("--status", default="", help="Optional status filter")
    parser.add_argument("--approved-only", action="store_true", help="Require approved review-table rows")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    rows: list[dict] = []
    if input_path.is_dir():
        for pattern in ("*.jsonl", "*.tsv", "*.csv"):
            for path in sorted(input_path.glob(pattern)):
                rows.extend(_load_sampleable_rows(path, approved_only=args.approved_only, status=args.status))
    else:
        rows = _load_sampleable_rows(input_path, approved_only=args.approved_only, status=args.status)

    if args.tag:
        rows = [row for row in rows if args.tag in row.get("tags", [])]
    if args.lane:
        rows = [row for row in rows if row.get("lane") == args.lane]
    if not rows:
        raise ValueError("no rows matched the requested input")

    rng = random.Random(args.seed)
    sample = rows if len(rows) <= args.count else rng.sample(rows, args.count)
    for row in sample:
        print(json.dumps(row, indent=2, ensure_ascii=True))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
