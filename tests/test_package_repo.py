from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.config import InheritAssets, Manifest, ValidationConfig
from tools.heretic_to_onnx.package_repo import package_repo


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class PackageRepoTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
