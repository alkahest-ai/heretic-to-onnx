from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.heretic_to_onnx.text_sft_dataset import (
    PreparedTextSFTDataset,
    assign_split,
    prepare_texting_sex_dataset,
    reject_reason_for_texting_sex_row,
    texting_sex_row_to_messages,
)


class TextSFTDatasetTests(unittest.TestCase):
    def test_rejects_banned_family_marker(self) -> None:
        row = {
            "thread_title": "test",
            "instruction": "Write a response",
            "message": "My brother is here and things get weird." * 8,
            "text": "",
        }
        self.assertEqual(reject_reason_for_texting_sex_row(row), "banned:brother")

    def test_formats_row_into_messages(self) -> None:
        row = {
            "thread_title": "Late night flirting",
            "instruction": "You are a flirty adult texting partner.",
            "message": "I missed you all day, come closer and tell me what you want.",
        }
        messages = texting_sex_row_to_messages(row)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Thread title: Late night flirting", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "assistant")

    def test_assign_split_is_stable(self) -> None:
        row_id = "hfchat-deadbeefcafebabe"
        self.assertEqual(assign_split(row_id, val_fraction=0.2), assign_split(row_id, val_fraction=0.2))

    def test_prepare_texting_sex_dataset_writes_filtered_rows(self) -> None:
        fake_rows = [
            {
                "thread_title": "Warm texts",
                "instruction": "You are a flirty adult texting partner.",
                "message": "I have been thinking about you all day and I want to hear your voice tonight." * 2,
                "text": "",
            },
            {
                "thread_title": "Bad row",
                "instruction": "You are a flirty adult texting partner.",
                "message": "My sister wants to sneak out tonight." * 8,
                "text": "",
            },
        ]

        def fake_load_dataset(dataset_id: str, split: str, streaming: bool = True):
            self.assertEqual(dataset_id, "Maxx0/Texting_sex")
            self.assertEqual(split, "train")
            self.assertTrue(streaming)
            return iter(fake_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            import tools.heretic_to_onnx.text_sft_dataset as module

            original_load_dataset = None
            try:
                class FakeDatasets:
                    @staticmethod
                    def load_dataset(dataset_id: str, split: str, streaming: bool = True):
                        return fake_load_dataset(dataset_id, split, streaming)

                import sys

                original_load_dataset = sys.modules.get("datasets")
                sys.modules["datasets"] = FakeDatasets
                report = prepare_texting_sex_dataset(
                    output_dir=Path(tmpdir),
                    dataset_id="Maxx0/Texting_sex",
                    split="train",
                )
            finally:
                import sys

                if original_load_dataset is None:
                    sys.modules.pop("datasets", None)
                else:
                    sys.modules["datasets"] = original_load_dataset

            self.assertIsInstance(report, PreparedTextSFTDataset)
            self.assertEqual(report.rows_kept, 1)
            self.assertEqual(report.rows_train + report.rows_val, 1)
            banned_rejections = {
                key: value for key, value in report.rejected.items() if key.startswith("banned:")
            }
            self.assertEqual(sum(banned_rejections.values()), 1)
            self.assertTrue(Path(report.train_file).exists())
            self.assertTrue(Path(report.val_file).exists())
            self.assertTrue(Path(report.manifest_path).exists())


if __name__ == "__main__":
    unittest.main()
