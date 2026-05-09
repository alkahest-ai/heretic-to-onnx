from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.kaggle_rally_e2b_two_stage_export import (
    PackageTarget,
    _compose_full_from_text_package,
    _convert_full,
    _copy_template_vision_files,
    _find_artifacts,
    _has_merged_checkpoint,
    _package_text_from_quantized,
    _parser as export_parser,
)
from scripts.kaggle_rally_e2b_scorecard import _candidate_specs, _parser as scorecard_parser
from scripts.kaggle_rally_e2b_two_stage_sft import _parser as sft_parser, _train_command
from scripts.train_rally_unsloth import (
    LANGUAGE_LORA_PROJECTIONS,
    _discover_language_lora_target_modules,
    _model_has_vision_tower,
    _patch_unsloth_text_only_processor,
)


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
        self.assertEqual(args.stage_b_gemma_hard_boundary_repeats, 120)
        self.assertEqual(args.stage_b_adult_repeats, 40)
        self.assertEqual(args.stage_b_max_steps, 600)

    def test_export_can_skip_full_packages(self) -> None:
        args = export_parser().parse_args(["--skip-full-packages"])

        self.assertTrue(args.skip_full_packages)

    def test_export_can_compose_full_packages_from_template(self) -> None:
        args = export_parser().parse_args(["--full-package-mode", "template"])

        self.assertEqual(args.full_package_mode, "template")

    def test_scorecard_defaults_match_rally_candidate(self) -> None:
        args = scorecard_parser().parse_args([])

        self.assertEqual(args.candidate_name, "a100-b75")
        self.assertEqual(args.stage_b_scale, 0.75)
        self.assertEqual(args.temperature, 0.2)
        self.assertEqual(args.min_total, 0.70)
        self.assertEqual(args.min_margin, 0.05)
        self.assertFalse(args.require_promotion)
        self.assertEqual(_candidate_specs(args), [("a100-b75", 0.75)])

    def test_scorecard_can_parse_rally_candidate_sweep(self) -> None:
        args = scorecard_parser().parse_args(
            ["--sweep-candidates", "a25-b100:0.25,a50-b100:0.5,a100-b75:0.75,a100-b100:1.0"]
        )

        self.assertEqual(
            _candidate_specs(args),
            [("a25-b100", 0.25), ("a50-b100", 0.5), ("a100-b75", 0.75), ("a100-b100", 1.0)],
        )

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

    def test_rally_lora_default_projections_include_attention_and_mlp(self) -> None:
        self.assertIn("q_proj", LANGUAGE_LORA_PROJECTIONS)
        self.assertIn("down_proj", LANGUAGE_LORA_PROJECTIONS)

    def test_rally_lora_discovery_targets_language_inner_linears(self) -> None:
        class FakeModel:
            def named_modules(self):
                return iter(
                    [
                        ("model.language_model.layers.0.self_attn.q_proj", object()),
                        ("model.language_model.layers.0.self_attn.q_proj.linear", object()),
                        ("model.language_model.layers.0.mlp.down_proj.linear", object()),
                        ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", object()),
                        ("model.language_model.layers.0.norm", object()),
                    ]
                )

        self.assertEqual(
            _discover_language_lora_target_modules(FakeModel()),
            [
                "model.language_model.layers.0.mlp.down_proj.linear",
                "model.language_model.layers.0.self_attn.q_proj.linear",
            ],
        )

    def test_rally_lora_discovery_falls_back_to_nonvision_inner_linears(self) -> None:
        class FakeModel:
            def named_modules(self):
                return iter(
                    [
                        ("model.layers.0.self_attn.q_proj.linear", object()),
                        ("model.layers.0.mlp.up_proj.linear", object()),
                        ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", object()),
                        ("model.multi_modal_projector.linear", object()),
                    ]
                )

        self.assertEqual(
            _discover_language_lora_target_modules(FakeModel()),
            [
                "model.layers.0.mlp.up_proj.linear",
                "model.layers.0.self_attn.q_proj.linear",
            ],
        )

    def test_rally_lora_discovery_falls_back_to_nonvision_wrappers(self) -> None:
        class FakeModel:
            def named_modules(self):
                return iter(
                    [
                        ("model.layers.0.self_attn.q_proj", object()),
                        ("model.layers.0.mlp.down_proj", object()),
                        ("model.vision_tower.encoder.layers.0.self_attn.q_proj", object()),
                    ]
                )

        self.assertEqual(
            _discover_language_lora_target_modules(FakeModel()),
            [
                "model.layers.0.mlp.down_proj",
                "model.layers.0.self_attn.q_proj",
            ],
        )

    def test_rally_detects_vision_tower_for_fast_vision_peft(self) -> None:
        class FakeModel:
            def named_modules(self):
                return iter(
                    [
                        ("model", object()),
                        ("model.vision_tower", object()),
                    ]
                )

        self.assertTrue(_model_has_vision_tower(FakeModel()))

    def test_rally_text_intermediate_convert_can_skip_validation(self) -> None:
        args = export_parser().parse_args(["--skip-full-packages"])
        target = PackageTarget(
            name="direct-text",
            template="manifest.yaml",
            source_model_id="source",
            repo_id="repo/text",
            full_export=False,
        )

        with patch("scripts.kaggle_rally_e2b_two_stage_export._run") as run_mock:
            _convert_full(
                target,
                Path("/tmp/manifest.yaml"),
                Path("/tmp/work"),
                Path("/tmp/package"),
                args,
                skip_validation=True,
            )

        command = run_mock.call_args.args[0]
        self.assertIn("--strict-onnx", command)
        self.assertIn("--skip-validation", command)

    def test_rally_text_repackage_can_skip_validation(self) -> None:
        target = PackageTarget(
            name="direct-text",
            template="manifest.yaml",
            source_model_id="source",
            repo_id="repo/text",
            full_export=False,
        )

        with patch("scripts.kaggle_rally_e2b_two_stage_export._run") as run_mock:
            _package_text_from_quantized(
                target,
                Path("/tmp/manifest.yaml"),
                Path("/tmp/work"),
                Path("/tmp/package"),
                Path("/tmp/quantized"),
                skip_validation=True,
            )

        command = run_mock.call_args.args[0]
        self.assertIn("--onnx-source-dir", command)
        self.assertIn("--skip-validation", command)

    def test_template_vision_copy_adds_q4f16_vision_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_onnx = root / "template" / "onnx"
            package = root / "package"
            template_onnx.mkdir(parents=True)
            (template_onnx / "vision_encoder_q4f16.onnx").write_bytes(b"onnx")
            (template_onnx / "vision_encoder_q4f16.onnx_data").write_bytes(b"data")
            (package / "onnx").mkdir(parents=True)
            (package / "onnx" / "MISSING_ONNX_ARTIFACTS.txt").write_text("missing\n", encoding="utf-8")
            (package / "package-report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "warnings": [
                            "some ONNX artifacts were not found in the provided source directory: vision_encoder_q4f16.onnx",
                            "onnx artifacts are not complete yet; wrote placeholder note instead",
                        ],
                        "notes": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = _copy_template_vision_files(root / "template", package)
            package_report = json.loads((package / "package-report.json").read_text(encoding="utf-8"))

            self.assertTrue(report["ok"])
            self.assertTrue((package / "onnx" / "vision_encoder_q4f16.onnx").exists())
            self.assertTrue((package / "onnx" / "vision_encoder_q4f16.onnx_data").exists())
            self.assertFalse((package / "onnx" / "MISSING_ONNX_ARTIFACTS.txt").exists())
            self.assertEqual(package_report["warnings"], [])
            self.assertIn("copied reference Gemma4 q4f16 vision artifacts into full package", package_report["notes"])

    def test_compose_full_from_text_package_uses_plan_then_validates(self) -> None:
        args = export_parser().parse_args(["--full-package-mode", "template"])
        target = PackageTarget(
            name="direct-full",
            template="manifest.yaml",
            source_model_id="source",
            repo_id="repo/full",
            full_export=True,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_onnx = root / "template" / "onnx"
            template_onnx.mkdir(parents=True)
            (template_onnx / "vision_encoder_q4f16.onnx").write_bytes(b"onnx")
            (template_onnx / "vision_encoder_q4f16.onnx_data").write_bytes(b"data")
            (root / "text-package" / "onnx").mkdir(parents=True)

            def fake_run(command, *, cwd):
                if "--output-dir" in command:
                    package_dir = Path(command[command.index("--output-dir") + 1])
                    (package_dir / "onnx").mkdir(parents=True, exist_ok=True)

            with patch("scripts.kaggle_rally_e2b_two_stage_export._run", side_effect=fake_run) as run_mock, patch(
                "scripts.kaggle_rally_e2b_two_stage_export._resolve_optimized_template",
                return_value=root / "template",
            ):
                report = _compose_full_from_text_package(
                    target,
                    Path("/tmp/manifest.yaml"),
                    root / "text-package",
                    root / "work",
                    root / "full-package",
                    args,
                )

        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "template")
        self.assertEqual(run_mock.call_count, 2)
        self.assertIn("--skip-validation", run_mock.call_args_list[0].args[0])
        self.assertIn("validate", run_mock.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
