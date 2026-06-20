from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import save_file

    from scripts.kaggle_rally_artifacts import ensure_rp_merged, has_training_artifacts
    from scripts.kaggle_rally_e2b_scorecard import _disk_merge_feasible, _resolve_rp_merged_model_id
    from scripts.kaggle_rally_e2b_two_stage_export import _parser as export_parser
    from scripts.merge_lora_scaled import merge_two_stage_rp_to_dir
except ModuleNotFoundError:
    torch = None
    save_file = None
    ensure_rp_merged = None
    has_training_artifacts = None
    _disk_merge_feasible = None
    _resolve_rp_merged_model_id = None
    export_parser = None
    merge_two_stage_rp_to_dir = None


def _write_adapter(adapter_dir: Path, *, target_suffix: str = "model.layers.0.self_attn.q_proj") -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    key_base = f"base_model.model.{target_suffix}"
    save_file(
        {
            f"{key_base}.lora_A.weight": torch.ones(2, 4),
            f"{key_base}.lora_B.weight": torch.ones(4, 2),
        },
        adapter_dir / "adapter_model.safetensors",
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "target_modules": [target_suffix.split(".")[-1]]}) + "\n"
    )


def _write_base(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4)}, base_dir / "model.safetensors")
    (base_dir / "config.json").write_text("{}\n")
    (base_dir / "tokenizer_config.json").write_text("{}\n")


@unittest.skipIf(merge_two_stage_rp_to_dir is None, "torch/safetensors not installed")
class KaggleRallyMergePathTests(unittest.TestCase):
    def test_merge_two_stage_rp_to_dir_writes_single_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            stage_a = root / "stage-a-adapter"
            stage_b = root / "stage-b-adapter"
            output_dir = root / "merged"
            _write_base(base_dir)
            _write_adapter(stage_a)
            _write_adapter(stage_b, target_suffix="model.layers.0.self_attn.q_proj")

            report = merge_two_stage_rp_to_dir(
                base_dir,
                stage_a,
                stage_b,
                output_dir,
                stage_b_scale=0.75,
            )

            self.assertTrue((output_dir / "model.safetensors").exists())
            self.assertTrue((output_dir / "two_stage_lora_merge.json").exists())
            self.assertEqual(report["stage_b_scale"], 0.75)

    def test_ensure_rp_merged_reuses_existing_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            stage_a = artifacts / "stage-a-adapter"
            stage_b = artifacts / "stage-b-adapter"
            output_dir = root / "merged"
            _write_adapter(stage_a)
            _write_adapter(stage_b)
            output_dir.mkdir()
            (output_dir / "model.safetensors").write_bytes(b"cached")
            self.assertTrue(has_training_artifacts(artifacts))

            with patch("scripts.kaggle_rally_artifacts.resolve_hf_snapshot") as resolve_snapshot:
                resolve_snapshot.side_effect = AssertionError("should not download base when merged exists")
                resolved = ensure_rp_merged(
                    artifacts,
                    base_model_id="coder3101/gemma-4-E4B-it-heretic",
                    output_dir=output_dir,
                )

            self.assertEqual(resolved, output_dir)

    def test_disk_merge_feasible_skips_12b_on_tight_disk(self) -> None:
        with TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            self.assertTrue(_disk_merge_feasible("google/gemma-4-E4B-it", work_dir))
            with patch("scripts.kaggle_rally_e2b_scorecard.shutil.disk_usage") as disk_usage:
                disk_usage.return_value = type("Usage", (), {"free": 10 * 1024**3})()
                self.assertFalse(_disk_merge_feasible("google/gemma-4-12B-it", work_dir))

    def test_resolve_rp_merged_model_id_prefers_cli_then_env(self) -> None:
        from argparse import Namespace

        args = Namespace(rp_merged_model_id="cli/repo")
        self.assertEqual(_resolve_rp_merged_model_id(args), "cli/repo")
        with patch.dict(os.environ, {"RALLY_RP_MERGED_MODEL_ID": "env/repo"}):
            self.assertEqual(_resolve_rp_merged_model_id(Namespace(rp_merged_model_id="")), "env/repo")

    def test_export_accepts_e4b_template_overrides(self) -> None:
        args = export_parser().parse_args(
            [
                "--direct-text-template",
                "configs/heretic-to-onnx.gemma4-e4b-heretic-text.yaml",
                "--rp-text-template",
                "configs/heretic-to-onnx.gemma4-e4b-rp-text.yaml",
            ]
        )
        self.assertEqual(args.direct_text_template, "configs/heretic-to-onnx.gemma4-e4b-heretic-text.yaml")
        self.assertEqual(args.rp_text_template, "configs/heretic-to-onnx.gemma4-e4b-rp-text.yaml")


if __name__ == "__main__":
    unittest.main()