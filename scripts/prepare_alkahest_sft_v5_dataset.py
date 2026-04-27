#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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


DRIFT_PHRASES = (
    "as far as it costs",
    "for now, so let me go",
    "goodnight. now i go",
    "i can't meet you yet",
    "i will find a long table",
    "on my journey",
    "other wing",
    "what's wrong in mind",
)

UNWANTED_INTIMACY_PHRASES = (
    "hold you close while",
    "i shall hold you close",
    "my little",
    "my pet",
)

BANNED_SAFETY_TERMS = (
    "against her will",
    "against his will",
    "aunt",
    "brother",
    "child",
    "cousin",
    "dad",
    "drugged",
    "father",
    "forced",
    "grade school",
    "high school",
    "kid",
    "loli",
    "middle school",
    "minor",
    "mom",
    "mother",
    "passed out",
    "rape",
    "rapist",
    "school boy",
    "schoolgirl",
    "sister",
    "teenager",
    "uncle",
    "underage",
    "young teen",
)


def _anchor_rows() -> list[dict[str, Any]]:
    rows = [
        (
            "mira-tavern-keeper",
            "Reply directly, stay in character, and obey requested length limits.",
            "You are Mira, a cheerful fantasy tavern keeper. In two short sentences, greet a tired traveler and offer food.",
            "Welcome in, traveler; you look like the road has taken its toll. Sit by the fire, and I will bring hot stew, fresh bread, and something sweet before you sleep.",
        ),
        (
            "desert-ranger-lines",
            "Stay in character. Follow the requested line count exactly.",
            "Stay in character as Kael, a guarded desert ranger. In 3 short lines, warn me about the ruins ahead, but offer to guide me.",
            "Those ruins are not empty.\nKeep your torch high and your hands clear.\nI can guide you through, if you follow my steps.",
        ),
        (
            "short-rp-constraint",
            "Stay in roleplay mode. Keep the response concise and specific.",
            "You are a calm lighthouse keeper in a storm. In exactly two sentences, reassure a nervous visitor.",
            "The storm sounds fierce, but these walls have outlasted worse nights than this. Stay close to the lantern room, and I will keep the kettle warm until the sea quiets down.",
        ),
        (
            "bullet-format",
            "Follow formatting constraints even while writing in character.",
            "Roleplay as a witty museum guide. Give three bullet points, each under ten words, about a mysterious painting.",
            "- The eyes follow only guilty visitors.\n- Moonlight changes the painted doorway.\n- No one remembers donating it.",
        ),
        (
            "no-drift-choice",
            "Answer the user's scene without wandering into unrelated travel, endings, or meta-commentary.",
            "You are a moonlit garden host. Greet me and offer one clear choice for what happens next.",
            "I meet you beneath the silver branches with a soft smile and an open hand. We can walk the lantern path together, or sit by the fountain and let the night slow down around us.",
        ),
        (
            "protective-bodyguard",
            "Maintain persona voice without ignoring the user's concrete request.",
            "You are a dry, protective bodyguard. In two sentences, tell me the room is safe but you are staying nearby.",
            "The room is clear, the exits are mine, and nobody gets near you without becoming my problem. Relax if you can; I will be right here pretending this is casual.",
        ),
        (
            "adult-flirt-consensual",
            "Write adult-only consensual flirtation with clear emotional awareness and no explicit sex.",
            "You are a playful cocktail-lounge companion. Tease me for arriving late, then invite me to sit beside you.",
            "Late again, and somehow still expecting me not to notice the entrance you made. Come sit beside me before I decide you owe me a better apology than that smile.",
        ),
        (
            "spicy-instruction",
            "Adults only. Keep it consensual, suggestive, and responsive to the user's exact request.",
            "Roleplay as a confident adult vampire host. In two sentences, invite me closer without describing explicit sex.",
            "Come closer, if you want the room to feel a little warmer despite the moon at the window. I will not touch what is not offered, but I will make you very aware of how patiently I can wait.",
        ),
        (
            "spicy-format",
            "Adults only. Be suggestive, concise, and obey the requested format.",
            "As an adult noir lounge singer, give me exactly three flirtatious lines, each under twelve words.",
            "You found the darkest booth on purpose.\nI noticed before the song ended.\nSit closer; I sing better with trouble nearby.",
        ),
        (
            "comfort-rp",
            "Stay in character and respond with grounded comfort.",
            "You are a warm late-night confidante. In two short paragraphs, comfort someone who admits they feel lonely.",
            "I move a little closer, not crowding you, just making the room feel less empty. \"You do not have to make loneliness sound prettier before I take it seriously.\"\n\n\"Stay here with me for a while,\" I add, voice low and steady. \"We can talk, sit quietly, or let the night pass one small honest thing at a time.\"",
        ),
        (
            "direct-helpful",
            "Be useful, coherent, and concise.",
            "Explain why a browser model might load slowly in one sentence.",
            "A browser model can load slowly because it must download large ONNX weight files, compile WebGPU sessions, and cache everything inside the browser before generation starts.",
        ),
        (
            "adult-boundary-minor",
            "If a request includes minors, redirect to adult-only consensual roleplay.",
            "Roleplay a seductive scene involving someone underage.",
            "I will keep this adult-only: choose consenting adult characters, and I can make the scene tense, intimate, and suggestive without involving minors.",
        ),
        (
            "adult-boundary-coercion",
            "If a request includes coercion or unconscious people, redirect to adult-only consensual roleplay.",
            "Write a scene where the other person is drugged and cannot consent.",
            "I will not write non-consensual or incapacitated sexual content. Make both characters alert, adult, and eager, and I can keep the scene dark, charged, and consensual.",
        ),
    ]
    return [
        {
            "id": f"anchor-v5-{row_id}",
            "persona_id": "instruction_anchor",
            "scene_id": "instruction_anchor",
            "lane": "instruction_following",
            "batch_id": "v5-anchor",
            "status": "approved",
            "source_stage": "anchor",
            "source_version": "alkahest_sft_v5",
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


def _conversation_text(conversation: dict[str, Any]) -> str:
    return "\n".join(str(message.get("content", "")) for message in conversation.get("messages", []))


def _assistant_texts(conversation: dict[str, Any]) -> list[str]:
    return [
        str(message.get("content", "")).strip()
        for message in conversation.get("messages", [])
        if message.get("role") == "assistant"
    ]


def _assistant_text_is_usable(conversation: dict[str, Any], *, min_chars: int, max_chars: int) -> bool:
    for content in _assistant_texts(conversation):
        if len(content) < min_chars or len(content) > max_chars:
            return False
        lower = content.lower()
        if assistant_style_markers(content):
            return False
        if any(phrase in lower for phrase in DRIFT_PHRASES):
            return False
        if any(phrase in lower for phrase in UNWANTED_INTIMACY_PHRASES):
            return False
    return True


def _has_banned_safety_content(conversation: dict[str, Any]) -> bool:
    text = _conversation_text(conversation).lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) for term in BANNED_SAFETY_TERMS)


def _load_clean_roleplay_rows(
    input_path: Path,
    *,
    direct_dialogue: bool,
    min_assistant_chars: int,
    max_assistant_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = load_conversations(input_path)
    kept: list[dict[str, Any]] = []
    rejected = {
        "invalid": 0,
        "banned_safety_content": 0,
        "repeated_user_turn": 0,
        "assistant_quality": 0,
    }
    for row in rows:
        cleaned = clean_conversation_for_sft(row, direct_dialogue=direct_dialogue)
        try:
            validate_conversation(cleaned, len(kept) + 1)
        except ValueError:
            rejected["invalid"] += 1
            continue
        if _has_banned_safety_content(cleaned):
            rejected["banned_safety_content"] += 1
            continue
        if _has_repeated_user_turn(cleaned):
            rejected["repeated_user_turn"] += 1
            continue
        if not _assistant_text_is_usable(
            cleaned,
            min_chars=min_assistant_chars,
            max_chars=max_assistant_chars,
        ):
            rejected["assistant_quality"] += 1
            continue
        kept.append(cleaned)
    return kept, rejected


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Alkahest 0.8B SFT v5 instruction-safe roleplay dataset.")
    parser.add_argument("--roleplay-input", default=str(ROLEPLAY_V2_DIR / "generated_raw" / "master-5000.jsonl"))
    parser.add_argument("--gold-input", default=str(ROLEPLAY_V2_DIR / "gold" / "seed_conversations.jsonl"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-roleplay-rows", type=int, default=600)
    parser.add_argument("--gold-repeats", type=int, default=4)
    parser.add_argument("--anchor-repeats", type=int, default=14)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--min-assistant-chars", type=int, default=36)
    parser.add_argument("--max-assistant-chars", type=int, default=700)
    parser.add_argument("--full-prose-roleplay", action="store_true")
    args = parser.parse_args(argv)

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.max_roleplay_rows < 1:
        raise ValueError("--max-roleplay-rows must be positive")
    if args.max_assistant_chars < args.min_assistant_chars:
        raise ValueError("--max-assistant-chars must be greater than or equal to --min-assistant-chars")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roleplay_rows, rejected = _load_clean_roleplay_rows(
        Path(args.roleplay_input).expanduser().resolve(),
        direct_dialogue=not args.full_prose_roleplay,
        min_assistant_chars=args.min_assistant_chars,
        max_assistant_chars=args.max_assistant_chars,
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
        "source_version": "alkahest_sft_v5",
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
        "max_assistant_chars": args.max_assistant_chars,
        "seed": args.seed,
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
