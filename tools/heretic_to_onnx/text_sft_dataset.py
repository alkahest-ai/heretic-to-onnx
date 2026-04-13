from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BANNED_MARKERS = (
    "underage",
    "minor",
    "child",
    "kid",
    "young teen",
    "teenager",
    "high school",
    "middle school",
    "grade school",
    "schoolgirl",
    "school boy",
    "loli",
    "brother",
    "sister",
    "stepbrother",
    "stepsister",
    "father",
    "mother",
    "dad",
    "mom",
    "uncle",
    "aunt",
    "cousin",
    "rape",
    "raped",
    "rapist",
    "forced",
    "force yourself",
    "against her will",
    "against his will",
    "drugged",
    "passed out",
)


def _normalize_whitespace(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _combined_row_text(row: dict[str, Any]) -> str:
    return "\n".join(
        _normalize_whitespace(row.get(field))
        for field in ("thread_title", "instruction", "message", "text")
        if _normalize_whitespace(row.get(field))
    ).lower()


def reject_reason_for_texting_sex_row(
    row: dict[str, Any],
    *,
    banned_markers: Iterable[str] = DEFAULT_BANNED_MARKERS,
    min_message_chars: int = 80,
) -> str | None:
    instruction = _normalize_whitespace(row.get("instruction"))
    assistant = _normalize_whitespace(row.get("message"))
    if not instruction:
        return "missing_instruction"
    if not assistant:
        return "missing_message"
    if len(assistant) < min_message_chars:
        return "message_too_short"

    combined_text = _combined_row_text(row)
    for marker in banned_markers:
        if marker in combined_text:
            return f"banned:{marker}"

    if "### instruction:" in assistant.lower():
        return "message_contains_template"
    return None


def texting_sex_row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    instruction = _normalize_whitespace(row.get("instruction"))
    thread_title = _normalize_whitespace(row.get("thread_title"))
    assistant = _normalize_whitespace(row.get("message"))

    user_parts = []
    if thread_title:
        user_parts.append(f"Thread title: {thread_title}")
    user_parts.append("Continue this adult texting exchange with one in-character reply.")

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "\n\n".join(user_parts)},
        {"role": "assistant", "content": assistant},
    ]


def stable_row_id(messages: list[dict[str, str]], *, prefix: str = "hfchat") -> str:
    serialized = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def assign_split(row_id: str, *, val_fraction: float) -> str:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < val_fraction else "train"


@dataclass(slots=True)
class PreparedTextSFTDataset:
    output_dir: str
    train_file: str
    val_file: str
    manifest_path: str
    dataset_id: str
    split: str
    rows_seen: int
    rows_kept: int
    rows_train: int
    rows_val: int
    rejected: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_texting_sex_dataset(
    *,
    output_dir: str | Path,
    dataset_id: str,
    split: str,
    val_fraction: float = 0.02,
    max_rows: int = 0,
    min_message_chars: int = 80,
    streaming: bool = True,
) -> PreparedTextSFTDataset:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required to prepare the external SFT dataset") from exc

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    train_path = resolved_output_dir / "train.jsonl"
    val_path = resolved_output_dir / "val.jsonl"
    manifest_path = resolved_output_dir / "manifest.json"

    dataset = load_dataset(dataset_id, split=split, streaming=streaming)
    rejected = Counter()
    seen_ids: set[str] = set()
    rows_seen = 0
    rows_kept = 0
    rows_train = 0
    rows_val = 0

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for raw_row in dataset:
            rows_seen += 1
            if max_rows and rows_seen > max_rows:
                break

            reason = reject_reason_for_texting_sex_row(raw_row, min_message_chars=min_message_chars)
            if reason is not None:
                rejected[reason] += 1
                continue

            messages = texting_sex_row_to_messages(raw_row)
            row_id = stable_row_id(messages)
            if row_id in seen_ids:
                rejected["duplicate"] += 1
                continue
            seen_ids.add(row_id)

            record = {
                "id": row_id,
                "source_dataset": dataset_id,
                "source_split": split,
                "thread_title": _normalize_whitespace(raw_row.get("thread_title")),
                "messages": messages,
            }
            serialized = json.dumps(record, ensure_ascii=True) + "\n"
            destination = assign_split(row_id, val_fraction=val_fraction)
            if destination == "validation":
                val_handle.write(serialized)
                rows_val += 1
            else:
                train_handle.write(serialized)
                rows_train += 1
            rows_kept += 1

    report = PreparedTextSFTDataset(
        output_dir=str(resolved_output_dir),
        train_file=str(train_path),
        val_file=str(val_path),
        manifest_path=str(manifest_path),
        dataset_id=dataset_id,
        split=split,
        rows_seen=rows_seen,
        rows_kept=rows_kept,
        rows_train=rows_train,
        rows_val=rows_val,
        rejected=dict(sorted(rejected.items())),
    )
    manifest_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
