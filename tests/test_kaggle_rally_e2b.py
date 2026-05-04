from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.kaggle_rally_e2b_two_stage_export import _find_artifacts, _has_merged_checkpoint, _parser as export_parser
from scripts.kaggle_rally_e2b_two_stage_sft import _parser as sft_parser, _train_command
from scripts.train_rally_unsloth import _patch_unsloth_text_only_processor


class KaggleRallyE2BTests(unittest.TestCase):
    def test_find_artifacts_accepts_sharded_rally_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "rally-e2b-two-stage-sft"
            for subdir in ["stage-a-adapter", "stage-b-adapter", "stage-a-merged"]:
                (root / subdir).mkdir(parents=True)
            (root / "stage-a-adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "stage-b-adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "stage-a-merged" / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')

            self.assertTrue(_has_merged_checkpoint(root / "stage-a-merged"))
            self.assertEqual(_find_artifacts(str(root), "rally-e2b-two-stage-sft"), root.resolve())

    def test_sft_defaults_use_v8_stage_mix(self) -> None:
        args = sft_parser().parse_args([])

        self.assertEqual(args.stage_a_repeats, 18)
        self.assertEqual(args.stage_b_boundary_repeats, 80)
        self.assertEqual(args.stage_b_adult_repeats, 40)
        self.assertEqual(args.stage_b_max_steps, 450)

    def test_export_can_skip_full_packages(self) -> None:
        args = export_parser().parse_args(["--skip-full-packages"])

        self.assertTrue(args.skip_full_packages)

    def test_stage_b_command_can_skip_full_merge(self) -> None:
        args = sft_parser().parse_args([])
        common = {
            "args": args,
            "model_name": "model",
            "stage": "stage-b",
            "train_file": Path("/tmp/train.jsonl"),
            "val_file": Path("/tmp/val.jsonl"),
            "output_dir": Path("/tmp/adapter"),
            "merged_output_dir": Path("/tmp/merged"),
            "learning_rate": 2e-4,
            "max_steps": 1,
            "manifest_path": Path("/tmp/manifest.json"),
        }

        self.assertIn("--save-merged", _train_command(**common, save_merged=True))
        self.assertNotIn("--save-merged", _train_command(**common, save_merged=False))

    def test_unsloth_text_processor_patch_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            site_root = Path(tmp)
            vision_py = site_root / "unsloth/models/vision.py"
            vision_py.parent.mkdir(parents=True)
            vision_py.write_text(
                "\n".join(
                    [
                        "def loader():",
                        "    auto_processor = AutoProcessor if (is_vlm or is_whisper) else AutoTokenizer",
                        "    if True:",
                        "        patch_saving_functions(tokenizer, vision = True)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("scripts.train_rally_unsloth._candidate_site_roots", return_value=[site_root]):
                _patch_unsloth_text_only_processor()
                _patch_unsloth_text_only_processor()

            patched = vision_py.read_text(encoding="utf-8")
            self.assertEqual(patched.count("UNSLOTH_TEXT_ONLY_PROCESSOR"), 2)
            self.assertIn("        if os.environ.get(\"UNSLOTH_TEXT_ONLY_PROCESSOR\", \"0\") != \"1\":", patched)
            self.assertIn("            patch_saving_functions(tokenizer, vision = True)", patched)


if __name__ == "__main__":
    unittest.main()
