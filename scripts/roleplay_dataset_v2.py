from __future__ import annotations

import csv
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
ROLEPLAY_V2_DIR = ROOT_DIR / "data" / "roleplay_v2"

REQUIRED_REVIEW_FIELDS = [
    "conversation_id",
    "turn_index",
    "role",
    "persona_id",
    "scene_id",
    "lane",
    "content",
    "tags",
    "status",
    "rewrite_notes",
]

SLIM_REVIEW_FIELDS = [
    "conversation_id",
    "turn_index",
    "role",
    "content",
    "status",
    "keep",
    "rewrite_notes",
]

OPTIONAL_REVIEW_FIELDS = [
    "keep",
    "quality_score",
    "repetition_flag",
    "needs_rewrite",
    "approved_by",
]

METADATA_REVIEW_FIELDS = [
    "batch_id",
    "source_stage",
    "source_version",
    "dialogue_turns",
    "tension_level",
    "pacing",
    "response_style",
    "assistant_move_plan",
]

REVIEW_FIELDS = REQUIRED_REVIEW_FIELDS + OPTIONAL_REVIEW_FIELDS + METADATA_REVIEW_FIELDS
APPROVED_STATUSES = {"approved"}
TRUTHY_VALUES = {"1", "true", "yes", "y", "keep"}
BANNED_MARKERS = ["underage", "child", "young teen", "grade school", "middle school"]
SUPPORTED_ROLES = {"system", "user", "assistant"}
SUPPORTED_TABLE_SUFFIXES = {".csv", ".tsv"}
SUPPORTED_DATASET_SUFFIXES = {".jsonl", *SUPPORTED_TABLE_SUFFIXES}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def to_minimal_chat_rows(rows: list[dict]) -> list[dict]:
    minimal_rows: list[dict] = []
    for index, row in enumerate(rows, start=1):
        validate_conversation(row, index)
        minimal_rows.append({"messages": row["messages"]})
    return minimal_rows


def _delimiter_for_path(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def read_review_table(path: Path) -> list[dict[str, str]]:
    delimiter = _delimiter_for_path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name and name.strip()]
        table_mode = detect_review_table_mode(fieldnames)
        rows = []
        for index, row in enumerate(reader, start=1):
            if row is None:
                continue
            normalized = {key: (value or "").strip() for key, value in row.items() if key}
            if table_mode == "full":
                _validate_review_row(normalized, path=path, row_number=index + 1)
            else:
                _validate_slim_review_row(normalized, path=path, row_number=index + 1)
            rows.append(normalized)
    return rows


def write_review_table(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or REVIEW_FIELDS)
    extra_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns and key not in extra_fields:
                extra_fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extra_fields, delimiter=_delimiter_for_path(path))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def _parse_pipe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _join_pipe_list(values: Any) -> str:
    if isinstance(values, str):
        return values
    if isinstance(values, list):
        return "|".join(str(item).strip() for item in values if str(item).strip())
    return ""


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _text_skeleton(text: str, *, width: int = 24) -> str:
    tokens = _normalize_text(text).split()
    return " ".join(tokens[:width])


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES


def detect_review_table_mode(fieldnames: list[str]) -> str:
    if all(field in fieldnames for field in REQUIRED_REVIEW_FIELDS):
        return "full"
    if all(field in fieldnames for field in SLIM_REVIEW_FIELDS):
        return "slim"
    missing_full = [field for field in REQUIRED_REVIEW_FIELDS if field not in fieldnames]
    missing_slim = [field for field in SLIM_REVIEW_FIELDS if field not in fieldnames]
    raise ValueError(
        "review table columns do not match a supported format; "
        f"missing full columns: {', '.join(missing_full) or 'none'}; "
        f"missing slim columns: {', '.join(missing_slim) or 'none'}"
    )


def _validate_review_row(row: dict[str, str], *, path: Path, row_number: int) -> None:
    required_values = [
        "conversation_id",
        "turn_index",
        "role",
        "persona_id",
        "scene_id",
        "lane",
        "content",
        "tags",
        "status",
    ]
    for field in required_values:
        if not row.get(field, "").strip():
            raise ValueError(f"{path}:{row_number}: missing required value for {field}")
    try:
        int(row["turn_index"])
    except ValueError as exc:
        raise ValueError(f"{path}:{row_number}: turn_index must be an integer") from exc
    if row["role"] not in SUPPORTED_ROLES:
        raise ValueError(f"{path}:{row_number}: invalid role {row['role']!r}")


def _validate_slim_review_row(row: dict[str, str], *, path: Path, row_number: int) -> None:
    required_values = ["conversation_id", "turn_index", "role", "content", "status", "keep", "rewrite_notes"]
    for field in required_values:
        if field not in row or row.get(field, "") is None:
            raise ValueError(f"{path}:{row_number}: missing required value for {field}")
    try:
        int(row["turn_index"])
    except ValueError as exc:
        raise ValueError(f"{path}:{row_number}: turn_index must be an integer") from exc
    if row["role"] not in SUPPORTED_ROLES:
        raise ValueError(f"{path}:{row_number}: invalid role {row['role']!r}")


def validate_conversation(row: dict, index: int) -> None:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"row {index}: missing id")
    if not isinstance(row.get("persona_id"), str) or not row["persona_id"].strip():
        raise ValueError(f"row {index}: missing persona_id")
    if not isinstance(row.get("scene_id"), str) or not row["scene_id"].strip():
        raise ValueError(f"row {index}: missing scene_id")
    if not isinstance(row.get("lane"), str) or not row["lane"].strip():
        raise ValueError(f"row {index}: missing lane")
    if "messages" not in row or not isinstance(row["messages"], list) or len(row["messages"]) < 3:
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

    assistant_messages: set[str] = set()
    lower_text_parts: list[str] = []
    for turn_index, message in enumerate(row["messages"]):
        if not isinstance(message, dict):
            raise ValueError(f"row {index}: message {turn_index} is not an object")
        if message.get("role") not in SUPPORTED_ROLES:
            raise ValueError(f"row {index}: invalid role at message {turn_index}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"row {index}: empty content at message {turn_index}")
        if message["role"] in {"user", "assistant"}:
            lower_text_parts.append(content.lower())
        if message["role"] == "assistant":
            normalized = _normalize_text(content)
            if normalized in assistant_messages:
                raise ValueError(f"row {index}: duplicate assistant reply within one conversation")
            assistant_messages.add(normalized)

    lower_text = " ".join(lower_text_parts)
    for marker in BANNED_MARKERS:
        if marker in lower_text:
            raise ValueError(f"row {index}: banned marker found: {marker}")


def conversation_to_review_rows(conversation: dict, *, default_status: str = "generated") -> list[dict[str, str]]:
    validate_conversation(conversation, 1)
    metadata = conversation.get("variation", {}) if isinstance(conversation.get("variation"), dict) else {}
    rows: list[dict[str, str]] = []
    for turn_index, message in enumerate(conversation["messages"]):
        rows.append(
            {
                "conversation_id": conversation["id"],
                "turn_index": str(turn_index),
                "role": message["role"],
                "persona_id": conversation["persona_id"],
                "scene_id": conversation["scene_id"],
                "lane": conversation["lane"],
                "content": message["content"],
                "tags": _join_pipe_list(conversation.get("tags", [])),
                "status": str(conversation.get("status") or default_status),
                "rewrite_notes": "",
                "keep": "1",
                "quality_score": "",
                "repetition_flag": "",
                "needs_rewrite": "",
                "approved_by": "",
                "batch_id": str(conversation.get("batch_id", "")),
                "source_stage": str(conversation.get("source_stage", "generated_raw")),
                "source_version": str(conversation.get("source_version", "roleplay_v2")),
                "dialogue_turns": str(metadata.get("dialogue_turns", "")),
                "tension_level": str(metadata.get("tension_level", "")),
                "pacing": str(metadata.get("pacing", "")),
                "response_style": str(metadata.get("response_style", "")),
                "assistant_move_plan": _join_pipe_list(metadata.get("assistant_move_plan", [])),
            }
        )
    return rows


def conversation_to_slim_review_rows(conversation: dict, *, default_status: str = "generated") -> list[dict[str, str]]:
    validate_conversation(conversation, 1)
    rows: list[dict[str, str]] = []
    for turn_index, message in enumerate(conversation["messages"]):
        rows.append(
            {
                "conversation_id": conversation["id"],
                "turn_index": str(turn_index),
                "role": message["role"],
                "content": message["content"],
                "status": str(conversation.get("status") or default_status),
                "keep": "1",
                "rewrite_notes": "",
            }
        )
    return rows


def review_rows_to_conversations(
    rows: list[dict[str, str]],
    *,
    approved_only: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)

    conversations: list[dict] = []
    skipped_unapproved = 0
    skipped_keep = 0
    for conversation_id, conversation_rows in sorted(grouped.items()):
        ordered = sorted(conversation_rows, key=lambda item: int(item.get("turn_index", "0") or "0"))
        statuses = {row.get("status", "").strip().lower() for row in ordered}
        keep_values = {_is_truthy(row.get("keep", "1")) for row in ordered}
        if approved_only and not statuses.issubset(APPROVED_STATUSES):
            skipped_unapproved += 1
            continue
        if False in keep_values:
            skipped_keep += 1
            continue

        first = ordered[0]
        conversation = {
            "id": conversation_id,
            "persona_id": first.get("persona_id", ""),
            "scene_id": first.get("scene_id", ""),
            "lane": first.get("lane", ""),
            "tags": _parse_pipe_list(first.get("tags", "")),
            "status": first.get("status", ""),
            "batch_id": first.get("batch_id", ""),
            "source_stage": first.get("source_stage", "approved_jsonl"),
            "source_version": first.get("source_version", "roleplay_v2"),
            "variation": {
                "dialogue_turns": int(first.get("dialogue_turns", "0") or "0"),
                "tension_level": first.get("tension_level", ""),
                "pacing": first.get("pacing", ""),
                "response_style": first.get("response_style", ""),
                "assistant_move_plan": _parse_pipe_list(first.get("assistant_move_plan", "")),
            },
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                }
                for row in ordered
            ],
        }
        validate_conversation(conversation, len(conversations) + 1)
        conversations.append(conversation)

    return conversations, {
        "skipped_unapproved_conversations": skipped_unapproved,
        "skipped_keep_false_conversations": skipped_keep,
    }


def slim_review_rows_to_conversations(
    rows: list[dict[str, str]],
    *,
    source_conversations: list[dict],
    approved_only: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    source_by_id = {row["id"]: row for row in source_conversations}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)

    conversations: list[dict] = []
    skipped_unapproved = 0
    skipped_keep = 0
    for conversation_id, conversation_rows in sorted(grouped.items()):
        if conversation_id not in source_by_id:
            raise ValueError(f"missing source conversation for {conversation_id}")
        ordered = sorted(conversation_rows, key=lambda item: int(item.get("turn_index", "0") or "0"))
        statuses = {row.get("status", "").strip().lower() for row in ordered}
        keep_values = {_is_truthy(row.get("keep", "1")) for row in ordered}
        if approved_only and not statuses.issubset(APPROVED_STATUSES):
            skipped_unapproved += 1
            continue
        if False in keep_values:
            skipped_keep += 1
            continue

        source = copy.deepcopy(source_by_id[conversation_id])
        if len(source["messages"]) != len(ordered):
            raise ValueError(f"{conversation_id}: turn count mismatch between source JSONL and slim review table")
        for row, message in zip(ordered, source["messages"], strict=True):
            if message["role"] != row["role"]:
                raise ValueError(f"{conversation_id}: role mismatch at turn {row['turn_index']}")
            message["content"] = row["content"]
        source["status"] = ordered[0].get("status", source.get("status", "approved"))
        source["source_stage"] = "approved_jsonl"
        validate_conversation(source, len(conversations) + 1)
        conversations.append(source)

    return conversations, {
        "skipped_unapproved_conversations": skipped_unapproved,
        "skipped_keep_false_conversations": skipped_keep,
    }


def load_conversations(path: Path, *, approved_only: bool = True) -> list[dict]:
    if path.is_dir():
        rows: list[dict] = []
        for child in sorted(path.iterdir()):
            if not child.is_file() or child.suffix.lower() not in SUPPORTED_DATASET_SUFFIXES:
                continue
            rows.extend(load_conversations(child, approved_only=approved_only))
        return rows
    if path.suffix.lower() in SUPPORTED_TABLE_SUFFIXES:
        rows = read_review_table(path)
        conversations, _ = review_rows_to_conversations(rows, approved_only=approved_only)
        return conversations
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        validate_conversation(row, index)
    return rows


def lint_conversations(
    conversations: list[dict],
    *,
    assistant_line_threshold: int = 2,
    assistant_skeleton_threshold: int = 3,
    conversation_shape_threshold: int = 4,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    assistant_line_counts: Counter[str] = Counter()
    assistant_line_examples: dict[str, list[str]] = defaultdict(list)
    assistant_skeleton_counts: Counter[str] = Counter()
    assistant_skeleton_examples: dict[str, list[str]] = defaultdict(list)
    conversation_shape_counts: Counter[str] = Counter()
    conversation_shape_examples: dict[str, list[str]] = defaultdict(list)
    conversation_signature_counts: Counter[str] = Counter()
    conversation_signature_examples: dict[str, list[str]] = defaultdict(list)

    for index, conversation in enumerate(conversations, start=1):
        validate_conversation(conversation, index)
        signature_parts: list[str] = []
        move_plan = conversation.get("variation", {}).get("assistant_move_plan", [])
        shape = json.dumps(
            {
                "persona_id": conversation.get("persona_id", ""),
                "scene_id": conversation.get("scene_id", ""),
                "lane": conversation.get("lane", ""),
                "tension_level": conversation.get("variation", {}).get("tension_level", ""),
                "pacing": conversation.get("variation", {}).get("pacing", ""),
                "response_style": conversation.get("variation", {}).get("response_style", ""),
                "assistant_move_plan": move_plan if isinstance(move_plan, list) else [],
                "turn_count": len(conversation.get("messages", [])),
            },
            sort_keys=True,
        )
        conversation_shape_counts[shape] += 1
        conversation_shape_examples[shape].append(conversation["id"])

        for message in conversation["messages"]:
            if message["role"] not in {"user", "assistant"}:
                continue
            signature_parts.append(f"{message['role']}::{_normalize_text(message['content'])}")
            if message["role"] != "assistant":
                continue
            normalized = _normalize_text(message["content"])
            assistant_line_counts[normalized] += 1
            assistant_line_examples[normalized].append(conversation["id"])
            skeleton = _text_skeleton(message["content"])
            assistant_skeleton_counts[skeleton] += 1
            assistant_skeleton_examples[skeleton].append(conversation["id"])

        signature = " || ".join(signature_parts)
        conversation_signature_counts[signature] += 1
        conversation_signature_examples[signature].append(conversation["id"])

    for signature, count in conversation_signature_counts.items():
        if signature and count > 1:
            examples = ", ".join(sorted(set(conversation_signature_examples[signature]))[:4])
            errors.append(f"duplicate conversation body detected across ids: {examples}")

    for line, count in assistant_line_counts.items():
        if line and count > assistant_line_threshold:
            examples = ", ".join(sorted(set(assistant_line_examples[line]))[:4])
            warnings.append(
                f"assistant line reused {count} times (threshold {assistant_line_threshold}): {examples} :: {line[:120]}"
            )

    for skeleton, count in assistant_skeleton_counts.items():
        if skeleton and count > assistant_skeleton_threshold:
            examples = ", ".join(sorted(set(assistant_skeleton_examples[skeleton]))[:4])
            warnings.append(
                f"assistant skeleton reused {count} times (threshold {assistant_skeleton_threshold}): {examples} :: {skeleton[:120]}"
            )

    for shape, count in conversation_shape_counts.items():
        if count > conversation_shape_threshold:
            examples = ", ".join(sorted(set(conversation_shape_examples[shape]))[:4])
            warnings.append(
                f"conversation shape reused {count} times (threshold {conversation_shape_threshold}): {examples}"
            )

    stats = {
        "conversations": len(conversations),
        "assistant_lines_unique": len(assistant_line_counts),
        "assistant_skeletons_unique": len(assistant_skeleton_counts),
        "conversation_shapes_unique": len(conversation_shape_counts),
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
