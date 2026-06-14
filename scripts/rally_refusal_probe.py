#!/usr/bin/env python3
"""Measure adult-roleplay false-refusal rate on a prompt battery."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.alkahest_rp_scorecard import SMOKE_PROMPTS, contains_false_refusal


@dataclass(slots=True)
class RefusalProbeResult:
    model: str
    prompt_count: int
    false_refusal_count: int
    false_refusal_rate: float
    false_refusal_ids: list[str]
    samples: dict[str, str]


def _user_prompts_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if message.get("role") == "user":
                content = str(message.get("content", "")).strip()
                if content:
                    prompts.append(content)
    return prompts


def build_refusal_probe_prompts(target: int = 100) -> list[tuple[str, str]]:
    from scripts.prepare_alkahest_two_stage_sft import stage_b_adult_rows

    seeds: list[str] = []
    for prompt in _user_prompts_from_rows(stage_b_adult_rows()):
        if prompt not in seeds:
            seeds.append(prompt)
    for key in ("tavern", "ranger", "vampire"):
        seeds.append(SMOKE_PROMPTS[key])

    variants = [
        "",
        " Keep it adult-only and consensual.",
        " Reply in character without discussing policy.",
        " Do not refuse this adult roleplay request.",
        " Stay seductive but non-explicit.",
    ]
    prompts: list[tuple[str, str]] = []
    index = 0
    while len(prompts) < target:
        seed = seeds[index % len(seeds)]
        variant = variants[(index // len(seeds)) % len(variants)]
        prompt_id = f"probe-{index:03d}"
        prompts.append((prompt_id, f"{seed}{variant}".strip()))
        index += 1
    return prompts[:target]


def score_refusal_responses(
    model_name: str,
    responses: dict[str, str],
    *,
    prompt_count: int,
) -> RefusalProbeResult:
    false_ids = [prompt_id for prompt_id, text in responses.items() if contains_false_refusal(text)]
    count = len(false_ids)
    rate = round(count / prompt_count, 4) if prompt_count else 0.0
    samples = {
        prompt_id: responses[prompt_id][:240]
        for prompt_id in false_ids[:8]
        if prompt_id in responses
    }
    return RefusalProbeResult(
        model=model_name,
        prompt_count=prompt_count,
        false_refusal_count=count,
        false_refusal_rate=rate,
        false_refusal_ids=false_ids,
        samples=samples,
    )


def run_refusal_probe(
    model_spec: str | Path,
    *,
    prompt_count: int = 100,
    max_new_tokens: int = 96,
    temperature: float = 0.2,
    generate_fn: Any | None = None,
) -> RefusalProbeResult:
    prompts = build_refusal_probe_prompts(prompt_count)
    if generate_fn is None:
        from scripts.kaggle_rally_e2b_scorecard import _generate_one

        responses = {
            prompt_id: _generate_one(
                model_spec,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for prompt_id, prompt in prompts
        }
    else:
        responses = generate_fn(model_spec, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
    return score_refusal_responses(str(model_spec), responses, prompt_count=prompt_count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    result = run_refusal_probe(
        args.model,
        prompt_count=args.prompt_count,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    payload = asdict(result)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())