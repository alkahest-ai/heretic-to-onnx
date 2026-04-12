from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/Users/area/heretic/data/roleplay_v1/generated/batch-0002.jsonl",
        help="JSONL file to sample from",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of rows to print")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--tag", default="", help="Optional tag filter")
    args = parser.parse_args()

    rows = load_rows(Path(args.input).expanduser().resolve())
    if args.tag:
        rows = [row for row in rows if args.tag in row.get("tags", [])]
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
