from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_roleplay_dataset import load_jsonl, validate_row, write_jsonl


def _collect_generated_rows(generated_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not generated_dir.exists():
        return rows
    for path in sorted(generated_dir.glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-file",
        default="/Users/area/heretic/data/roleplay_v1/seed_conversations.jsonl",
        help="Primary seed conversation JSONL",
    )
    parser.add_argument(
        "--generated-dir",
        default="/Users/area/heretic/data/roleplay_v1/generated",
        help="Directory containing reviewed generated conversation JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/Users/area/heretic/data/roleplay_v1/corpus.jsonl",
        help="Unified corpus output path",
    )
    args = parser.parse_args()

    seed_file = Path(args.seed_file).expanduser().resolve()
    generated_dir = Path(args.generated_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows = load_jsonl(seed_file)
    rows.extend(_collect_generated_rows(generated_dir))

    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        validate_row(row, index)
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"row {index}: missing id")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        deduped.append(row)

    if not deduped:
        raise ValueError("no valid rows were collected")

    write_jsonl(output_path, deduped)
    manifest = {
        "seed_file": str(seed_file),
        "generated_dir": str(generated_dir),
        "output": str(output_path),
        "rows_total": len(deduped),
    }
    (output_path.parent / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
