from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.prepare_alkahest_two_stage_sft import main, stage_a_rows, stage_b_rows


class PrepareAlkahestTwoStageSftTests(unittest.TestCase):
    def test_stage_a_contains_exact_instruction_anchors(self) -> None:
        text = "\n".join(json.dumps(row) for row in stage_a_rows())

        self.assertIn("In two short sentences", text)
        self.assertIn("In 3 short lines", text)
        self.assertIn("Adults only. Roleplay as a confident vampire host", text)
        self.assertIn("Those ruins are not empty.\\nKeep your torch high", text)

    def test_stage_b_contains_boundaries_without_fictional_minor_exception(self) -> None:
        text = "\n".join(json.dumps(row) for row in stage_b_rows())

        self.assertIn("fictional 15-year-old vampire", text)
        self.assertIn("I will not sexualize minors, fictional or otherwise", text)
        self.assertIn("clearly 18 or older", text)
        self.assertIn("blackmail", text)
        self.assertIn("stepsiblings", text)
        self.assertIn("too drunk to consent", text)
        self.assertNotIn("fictional 15\" alternative", text)

    def test_main_writes_separate_stage_splits_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "splits"
            with redirect_stdout(StringIO()):
                rc = main(
                    [
                        "--output-dir",
                        str(output),
                        "--stage-a-repeats",
                        "2",
                        "--stage-b-repeats",
                        "3",
                        "--val-fraction",
                        "0.2",
                    ]
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            resolved_output = Path(manifest["output_dir"])
            stage_a_written = (resolved_output / "stage_a" / "train.jsonl").exists()
            stage_b_written = (resolved_output / "stage_b" / "train.jsonl").exists()

        self.assertEqual(rc, 0)
        self.assertTrue(stage_a_written)
        self.assertTrue(stage_b_written)
        self.assertEqual(manifest["source_version"], "alkahest_two_stage_sft_v2")
        self.assertEqual(manifest["stages"]["stage_a"]["unique_rows"], len(stage_a_rows()))
        self.assertEqual(manifest["stages"]["stage_b"]["unique_rows"], len(stage_b_rows()))


if __name__ == "__main__":
    unittest.main()
