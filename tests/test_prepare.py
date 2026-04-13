from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.prepare import _can_reuse_base_snapshot, _has_all_files


class PrepareTests(unittest.TestCase):
    def test_has_all_files_requires_existing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "missing"
            self.assertFalse(_has_all_files(root, []))

    def test_can_reuse_base_snapshot_accepts_processor_fallback_for_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "processor_config.json").write_text('{"image_processor": {"size": 224}}', encoding="utf-8")
            self.assertTrue(_can_reuse_base_snapshot(root, ["preprocessor_config.json"]))

    def test_can_reuse_base_snapshot_rejects_empty_existing_dir_for_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertFalse(_can_reuse_base_snapshot(root, ["preprocessor_config.json"]))

    def test_can_reuse_base_snapshot_accepts_processor_fallback_for_video_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "processor_config.json").write_text(
                '{"video_processor": {"do_rescale": true, "image_mean": [0.5, 0.5, 0.5]}}',
                encoding="utf-8",
            )
            self.assertTrue(_can_reuse_base_snapshot(root, ["video_preprocessor_config.json"]))


if __name__ == "__main__":
    unittest.main()
