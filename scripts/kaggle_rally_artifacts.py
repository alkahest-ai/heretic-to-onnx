#!/usr/bin/env python3
"""Resolve Rally two-stage SFT artifacts on Kaggle without redundant merged checkpoints."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def has_merged_checkpoint(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()


def has_training_artifacts(path: Path) -> bool:
    return (
        (path / "stage-a-adapter" / "adapter_model.safetensors").exists()
        and (path / "stage-b-adapter" / "adapter_model.safetensors").exists()
    )


def resolve_hf_snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id))


def ensure_stage_a_merged(
    artifacts: Path,
    *,
    base_model_id: str,
    scratch_dir: Path,
    scale: float = 1.0,
) -> Path:
    merged = artifacts / "stage-a-merged"
    if has_merged_checkpoint(merged):
        return merged

    adapter = artifacts / "stage-a-adapter"
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"missing stage-a adapter under {adapter}")

    output_dir = scratch_dir / "stage-a-merged"
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    base_dir = resolve_hf_snapshot(base_model_id)
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts/merge_lora_scaled.py"),
        "--base-dir",
        str(base_dir),
        "--adapter-dir",
        str(adapter),
        "--output-dir",
        str(output_dir),
        "--scale",
        str(scale),
    ]
    subprocess.check_call(command, cwd=str(ROOT_DIR))
    return output_dir


def find_artifacts(explicit: str, artifact_name: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path("/kaggle/input") / artifact_name,
            Path("/kaggle/input") / artifact_name / artifact_name,
            Path("/kaggle/working") / artifact_name,
        ]
    )
    candidates.extend(Path("/kaggle/input").glob(f"**/{artifact_name}"))

    for candidate in candidates:
        if has_training_artifacts(candidate):
            return candidate.resolve()
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Rally two-stage artifacts for {artifact_name}. Checked:\n{checked}")


def artifact_manifest(artifacts: Path, *, base_model_id: str) -> dict[str, object]:
    return {
        "artifacts": str(artifacts),
        "stage_a_adapter": str(artifacts / "stage-a-adapter"),
        "stage_b_adapter": str(artifacts / "stage-b-adapter"),
        "stage_a_merged_present": has_merged_checkpoint(artifacts / "stage-a-merged"),
        "base_model_id": base_model_id,
    }


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")