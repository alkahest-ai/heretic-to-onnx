from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.package_repo import package_repo
from tools.heretic_to_onnx.validate_repo import validate_package


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class PackageRepoTests(unittest.TestCase):
    def test_package_repo_normalizes_bfloat16_config_fields_for_q4f16_browser_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            base_dir = root / "base"
            output_dir = root / "out"

            _write_json(
                source_dir / "config.json",
                {
                    "architectures": ["Gemma4ForConditionalGeneration"],
                    "dtype": "bfloat16",
                    "audio_config": {"dtype": "bfloat16"},
                    "text_config": {"dtype": "bfloat16", "torch_dtype": "bfloat16"},
                    "vision_config": {"dtype": "bfloat16"},
                },
            )
            _write_json(source_dir / "generation_config.json", {})
            _write_json(source_dir / "tokenizer.json", {})
            _write_json(source_dir / "tokenizer_config.json", {})
            (source_dir / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
            _write_json(base_dir / "processor_config.json", {"image_processor": {"size": 224}})

            manifest = Manifest(
                source_model_id=str(source_dir),
                base_model_id=str(base_dir),
                architecture="gemma4_conditional_generation",
                target_repo_id="alkahest-ai/rally-2b",
                target_dtype="q4f16",
                target_device="webgpu",
                modalities=["text", "image"],
                inherit_assets=InheritAssets(
                    from_source=[
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                    ],
                    from_base_if_missing=["preprocessor_config.json"],
                ),
                expected_architecture="Gemma4ForConditionalGeneration",
                expected_onnx_files=["onnx/decoder_model_merged_q4f16.onnx"],
                validation=ValidationConfig(),
                manifest_path=root / "manifest.yaml",
            )

            report = package_repo(
                manifest,
                output_dir=output_dir,
                allow_missing_onnx=True,
            )

            self.assertTrue(report.ok)
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["dtype"], "float16")
            self.assertEqual(config["audio_config"]["dtype"], "float16")
            self.assertEqual(config["text_config"]["dtype"], "float16")
            self.assertEqual(config["text_config"]["torch_dtype"], "float16")
            self.assertEqual(config["vision_config"]["dtype"], "float16")
            validation = validate_package(manifest, output_dir)
            self.assertTrue(validation.ok, validation.errors)

    def test_package_repo_normalizes_qwen_q4_webgpu_config_and_external_data_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            base_dir = root / "base"
            output_dir = root / "out"

            _write_json(
                source_dir / "config.json",
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "dtype": "bfloat16",
                    "text_config": {"dtype": "bfloat16", "torch_dtype": "bfloat16"},
                    "vision_config": {"dtype": "bfloat16"},
                },
            )
            _write_json(source_dir / "generation_config.json", {})
            _write_json(source_dir / "tokenizer.json", {})
            _write_json(source_dir / "tokenizer_config.json", {})
            (source_dir / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
            _write_json(base_dir / "preprocessor_config.json", {"size": {"shortest_edge": 224}})

            manifest = Manifest(
                source_model_id=str(source_dir),
                base_model_id=str(base_dir),
                architecture="qwen3_5_conditional_generation",
                target_repo_id="thomasjvu/alkahest-0.8b-q4-webgpu",
                target_dtype="q4",
                target_device="webgpu",
                modalities=["text", "image"],
                inherit_assets=InheritAssets(
                    from_source=[
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                    ],
                    from_base_if_missing=["preprocessor_config.json"],
                ),
                expected_architecture="Qwen3_5ForConditionalGeneration",
                expected_onnx_files=[
                    "onnx/vision_encoder_fp16.onnx",
                    "onnx/embed_tokens_q4.onnx",
                    "onnx/decoder_model_merged_q4.onnx",
                ],
                validation=ValidationConfig(),
                manifest_path=root / "manifest.yaml",
            )

            report = package_repo(
                manifest,
                output_dir=output_dir,
                allow_missing_onnx=True,
            )

            self.assertTrue(report.ok)
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["dtype"], "float16")
            self.assertEqual(config["text_config"]["dtype"], "float16")
            self.assertEqual(config["text_config"]["torch_dtype"], "float16")
            self.assertEqual(config["vision_config"]["dtype"], "float16")
            transformers_js_config = config["transformers.js_config"]
            self.assertEqual(
                transformers_js_config["use_external_data_format"],
                {"vision_encoder": 1, "embed_tokens": 1, "decoder_model_merged": 1},
            )
            self.assertEqual(transformers_js_config["kv_cache_dtype"]["q4"], "float16")

    def test_package_repo_copies_video_preprocessor_from_base_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            base_dir = root / "base"
            output_dir = root / "out"

            _write_json(source_dir / "config.json", {"architectures": ["Qwen3_5ForConditionalGeneration"]})
            _write_json(source_dir / "generation_config.json", {})
            _write_json(source_dir / "tokenizer.json", {})
            _write_json(source_dir / "tokenizer_config.json", {})
            (source_dir / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
            _write_json(base_dir / "preprocessor_config.json", {"size": {"shortest_edge": 224}})
            _write_json(base_dir / "video_preprocessor_config.json", {"num_frames": 8})

            manifest = Manifest(
                source_model_id=str(source_dir),
                base_model_id=str(base_dir),
                architecture="qwen3_5_conditional_generation",
                target_repo_id="alkahest-ai/alkahest-0.8b-v2",
                target_dtype="q4f16",
                target_device="webgpu",
                modalities=["text", "image", "video"],
                inherit_assets=InheritAssets(
                    from_source=[
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                    ],
                    from_base_if_missing=["preprocessor_config.json", "video_preprocessor_config.json"],
                ),
                expected_architecture="Qwen3_5ForConditionalGeneration",
                expected_onnx_files=["onnx/vision_encoder_q4f16.onnx"],
                validation=ValidationConfig(),
                manifest_path=root / "manifest.yaml",
            )

            report = package_repo(
                manifest,
                output_dir=output_dir,
                allow_missing_onnx=True,
            )

            self.assertTrue(report.ok)
            self.assertEqual(
                json.loads((output_dir / "video_preprocessor_config.json").read_text(encoding="utf-8")),
                {"num_frames": 8},
            )

    def test_package_repo_synthesizes_video_preprocessor_from_processor_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            base_dir = root / "base"
            output_dir = root / "out"

            _write_json(source_dir / "config.json", {"architectures": ["Gemma4ForConditionalGeneration"]})
            _write_json(source_dir / "generation_config.json", {})
            _write_json(source_dir / "tokenizer.json", {})
            _write_json(source_dir / "tokenizer_config.json", {})
            (source_dir / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
            _write_json(
                base_dir / "processor_config.json",
                {"video_processor": {"do_resize": True, "size": {"height": 384, "width": 384}}},
            )

            manifest = Manifest(
                source_model_id=str(source_dir),
                base_model_id=str(base_dir),
                architecture="gemma4_conditional_generation",
                target_repo_id="alkahest-ai/rally-2b-v2",
                target_dtype="q4f16",
                target_device="webgpu",
                modalities=["text", "image", "audio", "video"],
                inherit_assets=InheritAssets(
                    from_source=[
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                    ],
                    from_base_if_missing=["video_preprocessor_config.json"],
                ),
                expected_architecture="Gemma4ForConditionalGeneration",
                expected_onnx_files=["onnx/vision_encoder_q4f16.onnx"],
                validation=ValidationConfig(),
                manifest_path=root / "manifest.yaml",
            )

            report = package_repo(
                manifest,
                output_dir=output_dir,
                allow_missing_onnx=True,
            )

            self.assertTrue(report.ok)
            self.assertEqual(
                json.loads((output_dir / "video_preprocessor_config.json").read_text(encoding="utf-8")),
                {"do_resize": True, "size": {"height": 384, "width": 384}},
            )
            self.assertIn(
                "synthesized video_preprocessor_config.json from processor_config.json",
                report.notes,
            )


if __name__ == "__main__":
    unittest.main()
