from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.publish_hf import _ensure_model_card, write_model_card


def _manifest(*, target_repo_id: str, modalities: list[str], architecture: str) -> Manifest:
    return Manifest(
        source_model_id="example/source",
        base_model_id="example/base",
        architecture=architecture,
        target_repo_id=target_repo_id,
        target_dtype="q4f16",
        target_device="webgpu",
        modalities=modalities,
        inherit_assets=InheritAssets(
            from_source=["config.json", "tokenizer.json"],
            from_base_if_missing=["preprocessor_config.json"],
        ),
        expected_architecture="Qwen3_5ForConditionalGeneration"
        if architecture == "qwen3_5_conditional_generation"
        else "Gemma4ForConditionalGeneration",
        expected_onnx_files=[
            "onnx/vision_encoder_q4f16.onnx",
            "onnx/embed_tokens_q4f16.onnx",
            "onnx/decoder_model_merged_q4f16.onnx",
        ],
        validation=ValidationConfig(),
        manifest_path=Path("/tmp/manifest.yaml"),
    )


class PublishHFTests(unittest.TestCase):
    def test_qwen_v1_model_card_links_to_v2_and_lists_missing_video(self) -> None:
        manifest = _manifest(
            target_repo_id="thomasjvu/alkahest-0.8b",
            modalities=["text", "image"],
            architecture="qwen3_5_conditional_generation",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir)
            model_card_path = _ensure_model_card(manifest, package_dir, manifest.target_repo_id)
            content = model_card_path.read_text(encoding="utf-8")

        self.assertIn("Supported inputs: `text`, `image`", content)
        self.assertIn("Not included in this package: `video`", content)
        self.assertIn("[thomasjvu/alkahest-0.8b-v2](https://huggingface.co/thomasjvu/alkahest-0.8b-v2)", content)

    def test_qwen_v2_model_card_links_back_to_v1_and_lists_video(self) -> None:
        manifest = _manifest(
            target_repo_id="thomasjvu/alkahest-0.8b-v2",
            modalities=["text", "image", "video"],
            architecture="qwen3_5_conditional_generation",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir)
            model_card_path = _ensure_model_card(manifest, package_dir, manifest.target_repo_id)
            content = model_card_path.read_text(encoding="utf-8")

        self.assertIn("Supported inputs: `text`, `image`, `video`", content)
        self.assertIn("adds support for `video`", content)
        self.assertIn("[thomasjvu/alkahest-0.8b](https://huggingface.co/thomasjvu/alkahest-0.8b)", content)

    def test_ensure_model_card_rewrites_existing_generated_readme(self) -> None:
        manifest = _manifest(
            target_repo_id="thomasjvu/rally-2b-v2",
            modalities=["text", "image", "audio", "video"],
            architecture="gemma4_conditional_generation",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir)
            readme_path = package_dir / "README.md"
            readme_path.write_text("stale", encoding="utf-8")

            _ensure_model_card(manifest, package_dir, manifest.target_repo_id)
            content = readme_path.read_text(encoding="utf-8")

        self.assertNotIn("stale", content)
        self.assertIn("Supported inputs: `text`, `image`, `audio`, `video`", content)
        self.assertIn("The lighter v1 package remains available", content)

    def test_unsuffixed_enhanced_repo_is_described_as_v2_contract(self) -> None:
        manifest = _manifest(
            target_repo_id="thomasjvu/rally-2b-rp",
            modalities=["text", "image", "audio", "video"],
            architecture="gemma4_conditional_generation",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir)
            model_card_path = _ensure_model_card(manifest, package_dir, manifest.target_repo_id)
            content = model_card_path.read_text(encoding="utf-8")

        self.assertIn("enhanced browser `v2` multimodal contract", content)
        self.assertIn("repo name does not carry a `-v2` suffix", content)
        self.assertIn("this package itself is the multimodal variant", content)
        self.assertNotIn("stable v1 browser package", content)
        self.assertNotIn("thomasjvu/rally-2b-rp-v2", content)

    def test_write_model_card_renders_to_explicit_output_path(self) -> None:
        manifest = _manifest(
            target_repo_id="thomasjvu/alkahest-2b-v2",
            modalities=["text", "image", "video"],
            architecture="qwen3_5_conditional_generation",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "README.md"
            written_path = write_model_card(manifest, output_path=output_path)
            content = written_path.read_text(encoding="utf-8")

        self.assertEqual(written_path, output_path.resolve())
        self.assertIn("# alkahest-2b-v2", content)
        self.assertIn("Supported inputs: `text`, `image`, `video`", content)

    def test_local_source_model_id_is_not_emitted_as_base_model_metadata(self) -> None:
        manifest = _manifest(
            target_repo_id="thomasjvu/alkahest-4b-v2",
            modalities=["text", "image", "video"],
            architecture="qwen3_5_conditional_generation",
        )
        manifest.source_model_id = "/home/jovyan/work/heretic-to-onnx/build/phala_gpu_tee/alkahest-4b-direct/inputs/source"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "README.md"
            written_path = write_model_card(manifest, output_path=output_path)
            content = written_path.read_text(encoding="utf-8")

        self.assertNotIn("base_model: /home/jovyan/work/heretic-to-onnx/build/phala_gpu_tee/alkahest-4b-direct/inputs/source", content)
        self.assertIn("tags:", content)


if __name__ == "__main__":
    unittest.main()
