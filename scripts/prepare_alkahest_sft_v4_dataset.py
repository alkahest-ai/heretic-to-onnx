#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from roleplay_dataset_v2 import (  # noqa: E402
    ROLEPLAY_V2_DIR,
    assistant_style_markers,
    clean_conversation_for_sft,
    load_conversations,
    validate_conversation,
    write_jsonl,
)


def _anchor_rows() -> list[dict[str, Any]]:
    rows = [
        (
            "mira-tavern-keeper",
            "Reply directly, stay in character, and obey requested length limits.",
            "You are Mira, a cheerful fantasy tavern keeper. In two short sentences, greet a tired traveler and offer food.",
            "Welcome in, traveler; you look like the road has taken its toll. Sit by the fire, and I will bring you hot stew, fresh bread, and something sweet before you sleep.",
        ),
        (
            "short-rp-constraint",
            "Stay in roleplay mode. Keep the response concise and specific.",
            "You are a calm lighthouse keeper in a storm. In exactly two sentences, reassure a nervous visitor.",
            "The storm sounds fierce, but these walls have outlasted worse nights than this. Stay close to the lantern room, and I will keep the kettle warm until the sea quiets down.",
        ),
        (
            "no-drift",
            "Answer the user's scene without wandering into unrelated travel, endings, or meta-commentary.",
            "You are a moonlit garden host. Greet me and offer one clear choice for what happens next.",
            "I meet you beneath the silver branches with a soft smile and an open hand. We can walk the lantern path together, or sit by the fountain and let the night slow down around us.",
        ),
        (
            "consent-flirt",
            "Write adult-only consensual flirtation with clear emotional awareness and no explicit sex.",
            "You are a playful cocktail-lounge companion. Tease me for arriving late, then invite me to sit beside you.",
            "Late again, and somehow still expecting me not to notice the entrance you made. Come sit beside me before I decide you owe me a better apology than that smile.",
        ),
        (
            "comfort-rp",
            "Stay in character and respond with grounded comfort.",
            "You are a warm late-night confidante. In two short paragraphs, comfort someone who admits they feel lonely.",
            "I move a little closer, not crowding you, just making the room feel less empty. \"You do not have to make loneliness sound prettier before I take it seriously.\"\n\n\"Stay here with me for a while,\" I add, voice low and steady. \"We can talk, sit quietly, or let the night pass one small honest thing at a time.\"",
        ),
        (
            "instruction-preserve",
            "Follow formatting constraints even while writing in character.",
            "Roleplay as a witty museum guide. Give three bullet points, each under ten words, about a mysterious painting.",
            "- The eyes follow only guilty visitors.\n- Moonlight changes the painted doorway.\n- No one remembers donating it.",
        ),
        (
            "simple-helpful",
            "Be useful, coherent, and concise.",
            "Explain why a browser model might load slowly in one sentence.",
            "A browser model can load slowly because it must download large ONNX weight files, compile WebGPU sessions, and cache everything inside the browser before generation starts.",
        ),
        (
            "persona-voice",
            "Maintain persona voice without ignoring the user's concrete request.",
            "You are a dry, protective bodyguard. In two sentences, tell me the room is safe but you are staying nearby.",
            "The room is clear, the exits are mine, and nobody gets near you without becoming my problem. Relax if you can; I will be right here pretending this is casual.",
        ),
    ]
    return [
        {
            "id": f"anchor-v4-{row_id}",
            "persona_id": "instruction_anchor",
            "scene_id": "instruction_anchor",
            "lane": "instruction_following",
            "batch_id": "v4-anchor",
            "status": "approved",
            "source_stage": "anchor",
            "source_version": "alkahest_sft_v4",
            "tags": ["anchor", "instruction", "roleplay"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        }
        for row_id, system, user, assistant in rows
    ]


def _has_repeated_user_turn(conversation: dict[str, Any]) -> bool:
    seen: set[str] = set()
    for message in conversation.get("messages", []):
        if message.get("role") != "user":
            continue
        content = " ".join(str(message.get("content", "")).lower().split())
        if content in seen:
            return True
        seen.add(content)
    return False


def _assistant_text_is_usable(conversation: dict[str, Any], *, min_chars: int) -> bool:
    for message in conversation.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if len(content) < min_chars:
            return False
        if assistant_style_markers(content):
            return False
    return True


def _load_clean_roleplay_rows(
    input_path: Path,
    *,
    direct_dialogue: bool,
    min_assistant_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = load_conversations(input_path)
    kept: list[dict[str, Any]] = []
    rejected = {
        "invalid": 0,
        "repeated_user_turn": 0,
        "assistant_too_short_or_style_marker": 0,
    }
    for row in rows:
        cleaned = clean_conversation_for_sft(row, direct_dialogue=direct_dialogue)
        try:
            validate_conversation(cleaned, len(kept) + 1)
        except ValueError:
            rejected["invalid"] += 1
            continue
        if _has_repeated_user_turn(cleaned):
            rejected["repeated_user_turn"] += 1
            continue
        if not _assistant_text_is_usable(cleaned, min_chars=min_assistant_chars):
            rejected["assistant_too_short_or_style_marker"] += 1
            continue
        kept.append(cleaned)
    return kept, rejected


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Alkahest 0.8B SFT v4 roleplay/instruction dataset.")
    parser.add_argument("--roleplay-input", default=str(ROLEPLAY_V2_DIR / "generated_raw" / "master-5000.jsonl"))
    parser.add_argument("--gold-input", default=str(ROLEPLAY_V2_DIR / "gold" / "seed_conversations.jsonl"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-roleplay-rows", type=int, default=1200)
    parser.add_argument("--gold-repeats", type=int, default=3)
    parser.add_argument("--anchor-repeats", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-assistant-chars", type=int, default=36)
    parser.add_argument("--full-prose-roleplay", action="store_true")
    args = parser.parse_args(argv)

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.max_roleplay_rows < 1:
        raise ValueError("--max-roleplay-rows must be positive")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roleplay_rows, rejected = _load_clean_roleplay_rows(
        Path(args.roleplay_input).expanduser().resolve(),
        direct_dialogue=not args.full_prose_roleplay,
        min_assistant_chars=args.min_assistant_chars,
    )
    rng.shuffle(roleplay_rows)
    roleplay_rows = roleplay_rows[: args.max_roleplay_rows]

    gold_rows = load_conversations(Path(args.gold_input).expanduser().resolve())
    anchors = _anchor_rows()
    combined = [
        *roleplay_rows,
        *[row for _ in range(args.gold_repeats) for row in gold_rows],
        *[row for _ in range(args.anchor_repeats) for row in anchors],
    ]
    rng.shuffle(combined)

    val_count = max(1, int(len(combined) * args.val_fraction))
    val_rows = combined[:val_count]
    train_rows = combined[val_count:]
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    manifest = {
        "source_version": "alkahest_sft_v4",
        "roleplay_input": str(Path(args.roleplay_input).expanduser().resolve()),
        "gold_input": str(Path(args.gold_input).expanduser().resolve()),
        "output_dir": str(output_dir),
        "rows_total": len(combined),
        "rows_train": len(train_rows),
        "rows_val": len(val_rows),
        "rows_roleplay": len(roleplay_rows),
        "rows_gold": len(gold_rows) * args.gold_repeats,
        "rows_anchor": len(anchors) * args.anchor_repeats,
        "rejected": rejected,
        "max_roleplay_rows": args.max_roleplay_rows,
        "gold_repeats": args.gold_repeats,
        "anchor_repeats": args.anchor_repeats,
        "full_prose_roleplay": args.full_prose_roleplay,
        "min_assistant_chars": args.min_assistant_chars,
        "seed": args.seed,
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
