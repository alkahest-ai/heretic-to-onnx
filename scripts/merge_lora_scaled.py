#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a safetensors checkpoint with scaled deltas."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--base-weight", default="model.safetensors")
    parser.add_argument("--adapter-weight", default="adapter_model.safetensors")
    return parser.parse_args()


def adapter_target_key(lora_a_key: str) -> str:
    prefix = "base_model.model."
    suffix = ".lora_A.weight"
    if not lora_a_key.startswith(prefix) or not lora_a_key.endswith(suffix):
        raise ValueError(f"unexpected LoRA A key: {lora_a_key}")
    return lora_a_key[len(prefix) : -len(suffix)] + ".weight"


def main() -> None:
    args = parse_args()
    if args.scale < 0:
        raise ValueError("--scale must be non-negative")

    base_path = args.base_dir / args.base_weight
    adapter_path = args.adapter_dir / args.adapter_weight
    config_path = args.adapter_dir / "adapter_config.json"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    adapter_config = json.loads(config_path.read_text())
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    lora_multiplier = (alpha / rank) * args.scale

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        src = args.base_dir / name
        if src.exists():
            shutil.copy2(src, args.output_dir / name)

    with safe_open(base_path, framework="pt", device="cpu") as base_file:
        tensors = {key: base_file.get_tensor(key) for key in base_file.keys()}
        metadata = base_file.metadata()

    with safe_open(adapter_path, framework="pt", device="cpu") as adapter_file:
        adapter_keys = list(adapter_file.keys())
        lora_a_keys = sorted(key for key in adapter_keys if key.endswith(".lora_A.weight"))
        applied = []
        missing = []

        for lora_a_key in lora_a_keys:
            lora_b_key = lora_a_key.replace(".lora_A.weight", ".lora_B.weight")
            target_key = adapter_target_key(lora_a_key)
            if lora_b_key not in adapter_keys:
                raise ValueError(f"missing LoRA B key for {lora_a_key}")
            if target_key not in tensors:
                missing.append(target_key)
                continue

            base_weight = tensors[target_key]
            lora_a = adapter_file.get_tensor(lora_a_key).to(torch.float32)
            lora_b = adapter_file.get_tensor(lora_b_key).to(torch.float32)
            delta = torch.matmul(lora_b, lora_a) * lora_multiplier
            if tuple(delta.shape) != tuple(base_weight.shape):
                raise ValueError(
                    f"shape mismatch for {target_key}: delta={tuple(delta.shape)} base={tuple(base_weight.shape)}"
                )
            tensors[target_key] = (base_weight.to(torch.float32) + delta).to(base_weight.dtype)
            applied.append(target_key)

    if missing:
        raise ValueError(f"{len(missing)} adapter targets missing from base, first missing: {missing[:5]}")

    save_file(tensors, args.output_dir / args.base_weight, metadata=metadata)
    report = {
        "adapter_dir": str(args.adapter_dir),
        "applied": len(applied),
        "base_dir": str(args.base_dir),
        "lora_alpha": alpha,
        "lora_rank": rank,
        "ok": True,
        "output_dir": str(args.output_dir),
        "scale": args.scale,
    }
    (args.output_dir / "scaled_lora_merge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
