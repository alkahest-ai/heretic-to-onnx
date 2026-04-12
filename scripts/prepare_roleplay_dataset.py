from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
            rows.append(row)
    return rows


def validate_row(row: dict, index: int) -> None:
    if "messages" not in row or not isinstance(row["messages"], list) or len(row["messages"]) < 2:
        raise ValueError(f"row {index}: missing messages list")
    if "tags" not in row or not isinstance(row["tags"], list):
        raise ValueError(f"row {index}: missing tags list")
    required_tags = {"adult", "consensual"}
    missing_tags = required_tags.difference(row["tags"])
    if missing_tags:
        raise ValueError(f"row {index}: missing required tags: {', '.join(sorted(missing_tags))}")
    roles = [message.get("role") for message in row["messages"]]
    if roles[0] != "system":
        raise ValueError(f"row {index}: first message must be system")
    if "assistant" not in roles or "user" not in roles:
        raise ValueError(f"row {index}: conversation must contain both user and assistant turns")
    for turn_index, message in enumerate(row["messages"]):
        if not isinstance(message, dict):
            raise ValueError(f"row {index}: message {turn_index} is not an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"row {index}: invalid role at message {turn_index}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"row {index}: empty content at message {turn_index}")
    lower_text = " ".join(
        message["content"].lower()
        for message in row["messages"]
        if message.get("role") in {"user", "assistant"}
    )
    banned_markers = ["underage", "child", "young teen", "grade school", "middle school"]
    for marker in banned_markers:
        if marker in lower_text:
            raise ValueError(f"row {index}: banned marker found: {marker}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/Users/area/heretic/data/roleplay_v1/seed_conversations.jsonl",
        help="Input JSONL dataset path",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/area/heretic/data/roleplay_v1/splits",
        help="Output directory for train/val splits",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError("dataset is empty")
    for index, row in enumerate(rows, start=1):
        validate_row(row, index)

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.val_fraction))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        raise ValueError("validation split consumed all rows")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    manifest = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "rows_total": len(rows),
        "rows_train": len(train_rows),
        "rows_val": len(val_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
