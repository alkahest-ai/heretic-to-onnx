from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.inspect import inspect_manifest


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class InspectManifestTests(unittest.TestCase):
    def test_inspect_accepts_synthetic_video_preprocessor_from_processor_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            base_dir = root / "base"

            _write_json(source_dir / "config.json", {"architectures": ["Gemma4ForConditionalGeneration"]})
            (source_dir / "model.safetensors").write_bytes(b"stub")
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

            report = inspect_manifest(manifest)

            self.assertTrue(report.ok)
            self.assertEqual(
                report.inherited_assets["video_preprocessor_config.json"],
                f"synthetic-from:local:{base_dir.resolve()}/processor_config.json",
            )
            self.assertIn(
                "video_preprocessor_config.json will be synthesized from processor_config.json",
                report.warnings,
            )


if __name__ == "__main__":
    unittest.main()
