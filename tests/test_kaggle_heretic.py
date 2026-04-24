from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.heretic_to_onnx.kaggle_heretic import (
    build_run_config,
    build_stdin_answers,
    render_config_toml,
    run_kaggle_heretic,
    validate_merged_checkpoint,
)
from tools.heretic_to_onnx.render_manifest import render_manifest


class KaggleHereticTests(unittest.TestCase):
    def test_config_generation_uses_constrained_kaggle_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_run_config(label="alkahest-2b", work_root=tmpdir)
            toml = render_config_toml(config)

        self.assertIn('model = "Qwen/Qwen3.5-2B"', toml)
        self.assertIn('quantization = "bnb_4bit"', toml)
        self.assertIn('device_map = "auto"', toml)
        self.assertIn('"0" = "14GiB"', toml)
        self.assertIn('"1" = "14GiB"', toml)
        self.assertIn("n_trials = 20", toml)
        self.assertIn("n_startup_trials = 8", toml)
        self.assertIn("seed = 111", toml)
        self.assertIn("max_response_length = 64", toml)
        self.assertIn("offload_outputs_to_cpu = true", toml)
        self.assertIn('split = "train[:160]"', toml)
        self.assertIn('split = "test[:80]"', toml)

    def test_rally_preset_uses_official_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_run_config(label="rally-2b", work_root=tmpdir, accelerator="auto")

        self.assertEqual(config.base_model_id, "google/gemma-4-E2B-it")
        self.assertEqual(config.merged_output_dir.name, "rally-2b-heretic-merged")
        self.assertEqual(config.max_memory, {})

    def test_stdin_answers_select_first_trial_save_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_run_config(label="rally-2b", work_root=tmpdir)
            answers = build_stdin_answers(config).splitlines()

        self.assertEqual(answers[0], "1")
        self.assertEqual(answers[1], "1")
        self.assertEqual(answers[2], str(config.merged_output_dir))
        self.assertEqual(answers[3], "1")

    def test_validate_merged_checkpoint_reports_missing_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = validate_merged_checkpoint(tmpdir)

        self.assertFalse(report.ok)
        self.assertIn("config.json", report.missing)
        self.assertIn("generation_config.json", report.missing)
        self.assertIn("*.safetensors|pytorch_model*.bin", report.missing)

    def test_validate_merged_checkpoint_accepts_hf_checkpoint_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "generation_config.json").write_text("{}\n", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"")

            report = validate_merged_checkpoint(root)

        self.assertTrue(report.ok, report.missing)
        self.assertEqual(report.weight_files, ["model.safetensors"])
        self.assertEqual(report.tokenizer_files, ["tokenizer.json"])

    def test_dry_run_writes_config_and_report_without_launching_heretic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_run_config(label="alkahest-2b", work_root=tmpdir)
            report = run_kaggle_heretic(config, dry_run=True)
            report_json = json.dumps(report.to_dict())

            self.assertTrue(Path(report.config_path).is_file())
            self.assertTrue(Path(report.stdin_answers_path).is_file())

        self.assertTrue(report.ok)
        self.assertTrue(report.dry_run)
        self.assertIn("Qwen/Qwen3.5-2B", report_json)

    def test_manifest_render_keeps_base_model_when_source_is_local_heretic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            output = root / "manifest.yaml"
            template.write_text(
                "\n".join(
                    [
                        "source_model_id: old/heretic",
                        "base_model_id: old/base",
                        "target_repo_id: old/target",
                        "architecture: qwen3_5_conditional_generation",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = render_manifest(
                template,
                output,
                source_model_id="/models/alkahest-2b-heretic-merged",
                base_model_id="Qwen/Qwen3.5-2B",
                target_repo_id="alkahest-ai/alkahest-2b-v2",
            )
            data = yaml.safe_load(output.read_text(encoding="utf-8"))

        self.assertTrue(report.ok)
        self.assertEqual(data["source_model_id"], "/models/alkahest-2b-heretic-merged")
        self.assertEqual(data["base_model_id"], "Qwen/Qwen3.5-2B")
        self.assertEqual(data["target_repo_id"], "alkahest-ai/alkahest-2b-v2")


if __name__ == "__main__":
    unittest.main()
