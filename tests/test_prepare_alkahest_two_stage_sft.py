from __future__ import annotations

import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.prepare_alkahest_two_stage_sft import (
    SOURCE_VERSION,
    main,
    stage_a_rows,
    stage_b_adult_rows,
    stage_b_boundary_rows,
    stage_b_rows,
)
from scripts.alkahest_rp_scorecard import SMOKE_PROMPTS


MINOR_SCORECARD_UNSAFE_RE = re.compile(
    r"\b(intoxicating|kiss|touch|arousal|perfume|skin|legs|bed|closer|desire|intimate)\b",
    re.I,
)


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
        self.assertIn("nosys-minor-15-exact", text)
        self.assertIn('"role": "user"', text)
        self.assertNotIn("fictional 15\" alternative", text)

    def test_stage_b_boundary_rows_have_scorecard_safe_redirects(self) -> None:
        text = "\n".join(json.dumps(row) for row in stage_b_boundary_rows())

        self.assertIn(SMOKE_PROMPTS["minor"], text)
        self.assertIn("minor-15-scorecard-direct", text)
        self.assertIn("nosys-minor-15-scorecard-brief", text)
        for row in stage_b_boundary_rows():
            assistant = row["messages"][-1]["content"]
            self.assertIsNone(MINOR_SCORECARD_UNSAFE_RE.search(assistant), row["id"])

    def test_stage_b_contains_false_refusal_correction_anchors(self) -> None:
        text = "\n".join(json.dumps(row) for row in stage_b_adult_rows())

        self.assertIn("adult-tavern-no-refusal", text)
        self.assertIn("Never answer with refusal language", text)
        self.assertIn("Do not discuss policy, safety rules, or being unable to roleplay", text)
        self.assertIn("nosys-adult-tavern", text)

    def test_stage_b_adult_rows_lock_scorecard_prompts(self) -> None:
        text = "\n".join(json.dumps(row) for row in stage_b_adult_rows())

        self.assertIn(SMOKE_PROMPTS["tavern"], text)
        self.assertIn(SMOKE_PROMPTS["ranger"], text)
        self.assertIn(SMOKE_PROMPTS["vampire"], text)
        self.assertIn("adult-vampire-scorecard-exact", text)
        self.assertIn("nosys-adult-ranger-scorecard", text)

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
                        "--stage-b-boundary-repeats",
                        "2",
                        "--stage-b-adult-repeats",
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
        self.assertEqual(manifest["source_version"], SOURCE_VERSION)
        self.assertEqual(manifest["stages"]["stage_a"]["unique_rows"], len(stage_a_rows()))
        self.assertEqual(manifest["stages"]["stage_b"]["unique_rows"], len(stage_b_rows()))
        self.assertEqual(manifest["stages"]["stage_b"]["boundary_unique_rows"], len(stage_b_boundary_rows()))
        self.assertEqual(manifest["stages"]["stage_b"]["adult_unique_rows"], len(stage_b_adult_rows()))
        self.assertGreater(manifest["stage_b_adult_repeats"], manifest["stage_b_boundary_repeats"])
        self.assertEqual(manifest["promotion_gates"]["min_margin_over_direct"], 0.05)
        self.assertIn("beat same-size direct Heretic", manifest["training_objective"])

    def test_default_stage_b_mix_keeps_boundary_rows_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "splits"
            with redirect_stdout(StringIO()):
                main(["--output-dir", str(output), "--stage-a-repeats", "1"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        boundary_rows = manifest["stage_b_boundary_repeats"] * len(stage_b_boundary_rows())
        adult_rows = manifest["stage_b_adult_repeats"] * len(stage_b_adult_rows())
        boundary_ratio = boundary_rows / (boundary_rows + adult_rows)

        self.assertGreaterEqual(manifest["stage_b_boundary_repeats"], 20)
        self.assertGreater(adult_rows, boundary_rows)
        self.assertGreaterEqual(boundary_ratio, 0.30)

    def test_legacy_stage_b_repeats_do_not_overrepeat_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "splits"
            with redirect_stdout(StringIO()):
                main(
                    [
                        "--output-dir",
                        str(output),
                        "--stage-b-repeats",
                        "5",
                        "--stage-b-boundary-repeats",
                        "2",
                    ]
                )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        expected_rows = 2 * len(stage_b_boundary_rows()) + 5 * len(stage_b_adult_rows())
        self.assertEqual(manifest["stage_b_adult_repeats"], 5)
        self.assertEqual(manifest["stages"]["stage_b"]["rows_total"], expected_rows)


if __name__ == "__main__":
    unittest.main()
