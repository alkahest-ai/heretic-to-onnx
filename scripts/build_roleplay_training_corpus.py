from __future__ import annotations

import argparse
import json
from pathlib import Path

from roleplay_dataset_v2 import ROLEPLAY_V2_DIR, lint_conversations, load_conversations, write_jsonl


def _collect_jsonl_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(load_conversations(path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold-dir",
        default=str(ROLEPLAY_V2_DIR / "gold"),
        help="Directory containing curated gold conversation JSONL files",
    )
    parser.add_argument(
        "--approved-dir",
        default=str(ROLEPLAY_V2_DIR / "approved_jsonl"),
        help="Directory containing approved conversation JSONL files",
    )
    parser.add_argument(
        "--output",
        default=str(ROLEPLAY_V2_DIR / "corpus.jsonl"),
        help="Unified corpus output path",
    )
    parser.add_argument("--source-version", default="roleplay_v2", help="Dataset version label to record")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir).expanduser().resolve()
    approved_dir = Path(args.approved_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows = _collect_jsonl_rows(gold_dir)
    gold_count = len(rows)
    approved_rows = _collect_jsonl_rows(approved_dir)
    rows.extend(approved_rows)

    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        deduped.append(row)

    if not deduped:
        raise ValueError("no valid rows were collected")

    lint_report = lint_conversations(deduped)
    if lint_report["errors"]:
        raise ValueError("dataset lint failed:\n- " + "\n- ".join(lint_report["errors"]))

    write_jsonl(output_path, deduped)
    manifest = {
        "source_version": args.source_version,
        "gold_dir": str(gold_dir),
        "approved_dir": str(approved_dir),
        "output": str(output_path),
        "rows_gold": gold_count,
        "rows_approved": len(approved_rows),
        "rows_total": len(deduped),
        "lint": lint_report,
    }
    (output_path.parent / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
