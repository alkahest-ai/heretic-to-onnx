from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from roleplay_dataset_v2 import (  # noqa: E402
    conversation_to_review_rows,
    lint_conversations,
    load_conversations,
    read_review_table,
    review_rows_to_conversations,
    write_jsonl,
    write_review_table,
)


def _sample_conversation(conversation_id: str, *, user_tail: str = "closer", assistant_tail: str = "steady") -> dict:
    return {
        "id": conversation_id,
        "persona_id": "velvet_host",
        "scene_id": "after_hours_lounge",
        "lane": "praise",
        "tags": ["adult", "consensual", "original"],
        "status": "approved",
        "batch_id": "batch-0001",
        "source_stage": "approved_jsonl",
        "source_version": "roleplay_v2",
        "variation": {
            "tension_level": "warm",
            "pacing": "measured",
            "response_style": "praise",
            "assistant_move_plan": ["welcome", "praise", "invite"],
        },
        "messages": [
            {"role": "system", "content": "Adult-only consensual roleplay between fictional adults."},
            {"role": "user", "content": f"I stay close and ask for a slower answer {user_tail}."},
            {"role": "assistant", "content": f'I keep my voice soft and say, "Stay with me; you look incredible tonight" {assistant_tail}.'},
        ],
    }


class RoleplayDatasetV2Tests(unittest.TestCase):
    def test_round_trip_jsonl_review_table_jsonl_preserves_structure(self) -> None:
        original = _sample_conversation("conv-roundtrip")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_jsonl = tmp_path / "source.jsonl"
            review_tsv = tmp_path / "review.tsv"

            write_jsonl(source_jsonl, [original])
            loaded = load_conversations(source_jsonl, approved_only=False)
            rows = []
            for conversation in loaded:
                rows.extend(conversation_to_review_rows(conversation, default_status="approved"))
            for row in rows:
                row["status"] = "approved"
                row["keep"] = "1"
            write_review_table(review_tsv, rows)

            read_rows = read_review_table(review_tsv)
            rebuilt, skipped = review_rows_to_conversations(read_rows, approved_only=True)

        self.assertEqual(skipped["skipped_unapproved_conversations"], 0)
        self.assertEqual(skipped["skipped_keep_false_conversations"], 0)
        self.assertEqual(rebuilt, [original])

    def test_missing_required_review_columns_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bad_review = tmp_path / "bad.tsv"
            bad_review.write_text(
                "conversation_id\tturn_index\trole\tcontent\n"
                "conv-1\t0\tsystem\tAdult-only consensual roleplay.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_review_table(bad_review)

    def test_lint_flags_reused_assistant_lines(self) -> None:
        conversations = [
            _sample_conversation("conv-a", user_tail="near the sofa", assistant_tail="with a smile."),
            _sample_conversation("conv-b", user_tail="by the bar", assistant_tail="with a smile."),
        ]

        report = lint_conversations(conversations, assistant_line_threshold=1)

        self.assertTrue(report["ok"])
        self.assertFalse(report["errors"])
        self.assertTrue(any("assistant line reused" in warning for warning in report["warnings"]))

    def test_load_conversations_from_review_directory(self) -> None:
        first = _sample_conversation("conv-dir-1", user_tail="first", assistant_tail="first.")
        second = _sample_conversation("conv-dir-2", user_tail="second", assistant_tail="second.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            review_dir = tmp_path / "review_table"
            review_dir.mkdir(parents=True, exist_ok=True)

            for index, conversation in enumerate((first, second), start=1):
                rows = conversation_to_review_rows(conversation, default_status="approved")
                for row in rows:
                    row["status"] = "approved"
                    row["keep"] = "1"
                write_review_table(review_dir / f"batch-{index:04d}.tsv", rows)

            loaded = load_conversations(review_dir, approved_only=True)

        self.assertEqual([row["id"] for row in loaded], ["conv-dir-1", "conv-dir-2"])


if __name__ == "__main__":
    unittest.main()
