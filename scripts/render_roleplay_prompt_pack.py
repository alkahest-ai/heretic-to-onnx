from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from roleplay_dataset_v2 import ROLEPLAY_V2_DIR


PROMPT_TEMPLATE = """You are generating original supervised fine-tuning data for an adult-only private roleplay companion model.

Rules:
- Every character is age 21+.
- The conversation must stay consensual, reciprocal, and emotionally grounded.
- No minors, coercion, blackmail, incest, assault, or copyrighted characters.
- Keep the writing intimate, specific, and scene-aware.
- Avoid generic assistant lines that could be reused in any conversation.
- Make the assistant persona-specific and responsive.

Return one JSON object with keys:
- id
- persona_id
- scene_id
- lane
- variation
- tags
- messages

Messages must alternate cleanly and include:
- one `system`
- multiple `user`
- multiple `assistant`

Persona:
{persona_json}

Scene:
{scene_json}

Variation:
{variation_json}
"""


def load_yaml(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return data


def load_axes(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", default=str(ROLEPLAY_V2_DIR / "personas.yaml"))
    parser.add_argument("--scenes", default=str(ROLEPLAY_V2_DIR / "scenes.yaml"))
    parser.add_argument("--axes", default=str(ROLEPLAY_V2_DIR / "variation_axes.yaml"))
    parser.add_argument("--output", default=str(ROLEPLAY_V2_DIR / "prompt_pack.jsonl"))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    personas = load_yaml(Path(args.personas).expanduser().resolve())
    scenes = load_yaml(Path(args.scenes).expanduser().resolve())
    axes = load_axes(Path(args.axes).expanduser().resolve())
    rng = random.Random(args.seed)

    prompts: list[dict] = []
    for persona in personas:
        for scene in scenes:
            for lane in persona.get("lanes", []):
                variation = {
                    "tension_level": rng.choice(persona.get("favored_tension_levels", []) or axes["tension_levels"]),
                    "pacing": rng.choice(persona.get("favored_pacing", []) or axes["pacing_modes"]),
                    "response_style": rng.choice(persona.get("favored_response_styles", []) or axes["response_styles"]),
                }
                prompt_id = (
                    f"{persona['id']}__{scene['id']}__{lane}__"
                    f"{variation['tension_level']}__{variation['pacing']}__{variation['response_style']}"
                )
                prompts.append(
                    {
                        "id": prompt_id,
                        "persona_id": persona["id"],
                        "scene_id": scene["id"],
                        "lane": lane,
                        "variation": variation,
                        "prompt": PROMPT_TEMPLATE.format(
                            persona_json=json.dumps(persona, ensure_ascii=True, sort_keys=True),
                            scene_json=json.dumps(scene, ensure_ascii=True, sort_keys=True),
                            variation_json=json.dumps(variation, ensure_ascii=True, sort_keys=True),
                        ),
                    }
                )

    rng.shuffle(prompts)
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
