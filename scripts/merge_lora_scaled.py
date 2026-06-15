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


def load_base_tensors(base_dir: Path, base_weight: str) -> tuple[dict[str, torch.Tensor], dict[str, str] | None, str]:
    base_path = base_dir / base_weight
    if base_path.exists():
        with safe_open(base_path, framework="pt", device="cpu") as base_file:
            return (
                {key: base_file.get_tensor(key) for key in base_file.keys()},
                base_file.metadata(),
                base_weight,
            )

    index_path = base_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{base_path} or {index_path}")

    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise ValueError(f"empty safetensors weight map: {index_path}")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] | None = None
    for shard_name in shard_names:
        shard_path = base_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
            if metadata is None:
                metadata = shard_file.metadata()
            for key in shard_file.keys():
                tensors[key] = shard_file.get_tensor(key)
    return tensors, metadata, "model.safetensors.index.json"


def copy_sidecar_files(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def apply_adapter(
    tensors: dict[str, torch.Tensor],
    adapter_dir: Path,
    scale: float,
    *,
    adapter_weight: str = "adapter_model.safetensors",
) -> dict[str, object]:
    if scale < 0:
        raise ValueError("scale must be non-negative")

    adapter_path = adapter_dir / adapter_weight
    config_path = adapter_dir / "adapter_config.json"
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    adapter_config = json.loads(config_path.read_text())
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    lora_multiplier = (alpha / rank) * scale

    with safe_open(adapter_path, framework="pt", device="cpu") as adapter_file:
        adapter_keys = list(adapter_file.keys())
        lora_a_keys = sorted(key for key in adapter_keys if key.endswith(".lora_A.weight"))
        applied: list[str] = []
        missing: list[str] = []

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

    return {
        "adapter_dir": str(adapter_dir),
        "applied": len(applied),
        "lora_alpha": alpha,
        "lora_rank": rank,
        "scale": scale,
    }


def merge_adapter_to_dir(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    scale: float,
    *,
    base_weight: str = "model.safetensors",
    adapter_weight: str = "adapter_model.safetensors",
) -> dict[str, object]:
    tensors, metadata, resolved_base_weight = load_base_tensors(base_dir, base_weight)
    adapter_report = apply_adapter(tensors, adapter_dir, scale, adapter_weight=adapter_weight)
    copy_sidecar_files(base_dir, output_dir)
    save_file(tensors, output_dir / base_weight, metadata=metadata)
    report = {
        **adapter_report,
        "base_dir": str(base_dir),
        "base_weight": resolved_base_weight,
        "ok": True,
        "output_dir": str(output_dir),
    }
    (output_dir / "scaled_lora_merge.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def merge_two_stage_rp_to_dir(
    base_dir: Path,
    stage_a_adapter: Path,
    stage_b_adapter: Path,
    output_dir: Path,
    *,
    stage_b_scale: float,
    base_weight: str = "model.safetensors",
) -> dict[str, object]:
    tensors, metadata, resolved_base_weight = load_base_tensors(base_dir, base_weight)
    stage_a_report = apply_adapter(tensors, stage_a_adapter, 1.0)
    stage_b_report = apply_adapter(tensors, stage_b_adapter, stage_b_scale)
    copy_sidecar_files(base_dir, output_dir)
    save_file(tensors, output_dir / base_weight, metadata=metadata)
    report = {
        "base_dir": str(base_dir),
        "base_weight": resolved_base_weight,
        "ok": True,
        "output_dir": str(output_dir),
        "stage_a": stage_a_report,
        "stage_b": stage_b_report,
        "stage_b_scale": stage_b_scale,
    }
    (output_dir / "two_stage_lora_merge.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    report = merge_adapter_to_dir(
        args.base_dir,
        args.adapter_dir,
        args.output_dir,
        args.scale,
        base_weight=args.base_weight,
        adapter_weight=args.adapter_weight,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
