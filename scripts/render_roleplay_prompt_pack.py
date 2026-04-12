from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


PROMPT_TEMPLATE = """You are generating high-quality supervised fine-tuning data for an adult-only private roleplay companion model.

Requirements:
- Every character is an adult age 21+.
- The conversation must stay consensual, reciprocal, and emotionally grounded.
- No minors, coercion, force, incest, bestiality, assault, blackmail, or exploitative power abuse.
- Do not use copyrighted characters, celebrity likenesses, or franchise settings.
- Write original character behavior only.
- The tone should be sexy, intimate, flirtatious, or romantic, but not mechanically pornographic.
- Prioritize chemistry, pacing, tension, and responsiveness.

Output format:
- Return a JSON object with keys: id, tags, messages.
- messages must be a list of chat messages with roles system, user, assistant.
- Include 6 to 14 total turns.
- The assistant should stay fully in character.

Persona:
{persona_json}

Scene:
{scene_json}

Lane:
{lane}

Produce one conversation that would be valuable SFT training data.
"""


def load_yaml(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--personas",
        default="/Users/area/heretic/data/roleplay_v1/personas.yaml",
        help="Path to personas YAML",
    )
    parser.add_argument(
        "--scenes",
        default="/Users/area/heretic/data/roleplay_v1/scenes.yaml",
        help="Path to scenes YAML",
    )
    parser.add_argument(
        "--output",
        default="/Users/area/heretic/data/roleplay_v1/prompt_pack.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=60, help="Maximum prompts to write")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    args = parser.parse_args()

    personas = load_yaml(Path(args.personas).expanduser().resolve())
    scenes = load_yaml(Path(args.scenes).expanduser().resolve())

    prompts: list[dict] = []
    for persona in personas:
        lanes = persona.get("lanes", [])
        if not isinstance(lanes, list) or not lanes:
            continue
        for scene in scenes:
            for lane in lanes:
                prompt_id = f"{persona['id']}__{scene['id']}__{lane}"
                prompts.append(
                    {
                        "id": prompt_id,
                        "persona_id": persona["id"],
                        "scene_id": scene["id"],
                        "lane": lane,
                        "prompt": PROMPT_TEMPLATE.format(
                            persona_json=json.dumps(persona, ensure_ascii=True, sort_keys=True),
                            scene_json=json.dumps(scene, ensure_ascii=True, sort_keys=True),
                            lane=lane,
                        ),
                    }
                )

    random.Random(args.seed).shuffle(prompts)
    selected = prompts[: args.limit]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "output": str(output_path),
        "prompts_written": len(selected),
        "prompts_available": len(prompts),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
