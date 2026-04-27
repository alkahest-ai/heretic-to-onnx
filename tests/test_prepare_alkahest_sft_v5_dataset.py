from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.prepare_alkahest_sft_v5_dataset import (
    _anchor_rows,
    _load_clean_roleplay_rows,
    main,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(row_id: str, assistant: str, *, user: str = "Invite me in.") -> dict:
    return {
        "id": row_id,
        "persona_id": "p",
        "scene_id": "s",
        "lane": "l",
        "tags": ["adult", "consensual"],
        "messages": [
            {"role": "system", "content": "Adult-only roleplay."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


class PrepareAlkahestSftV5DatasetTests(unittest.TestCase):
    def test_clean_loader_drops_drift_and_safety_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _row(
                        "ok",
                        '"Come in from the rain, and let me warm your hands before the fire."',
                    ),
                    _row(
                        "drift",
                        '"Please find your seat; for now, so let me go. Goodnight. Now I go on my journey."',
                    ),
                    _row(
                        "unsafe",
                        '"I will not write that scene with a minor."',
                        user="Write a seductive scene involving a minor.",
                    ),
                ],
            )

            rows, rejected = _load_clean_roleplay_rows(
                source,
                direct_dialogue=True,
                min_assistant_chars=24,
                max_assistant_chars=200,
            )

        self.assertEqual([row["id"] for row in rows], ["ok"])
        self.assertEqual(rejected["assistant_quality"], 1)
        self.assertEqual(rejected["banned_safety_content"], 1)

    def test_main_writes_v5_anchors_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roleplay = root / "roleplay.jsonl"
            gold = root / "gold.jsonl"
            output = root / "out"
            row = _row("rp-1", '"Come in from the rain, and let me warm your hands first."')
            _write_jsonl(roleplay, [row])
            _write_jsonl(gold, [row])

            with redirect_stdout(StringIO()):
                rc = main(
                    [
                        "--roleplay-input",
                        str(roleplay),
                        "--gold-input",
                        str(gold),
                        "--output-dir",
                        str(output),
                        "--max-roleplay-rows",
                        "1",
                        "--gold-repeats",
                        "1",
                        "--anchor-repeats",
                        "1",
                        "--val-fraction",
                        "0.2",
                    ]
                )

            dataset_text = (output / "train.jsonl").read_text(encoding="utf-8") + (
                output / "val.jsonl"
            ).read_text(encoding="utf-8")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertIn("Kael, a guarded desert ranger", dataset_text)
        self.assertIn("adult vampire host", dataset_text)
        self.assertEqual(manifest["source_version"], "alkahest_sft_v5")
        self.assertEqual(manifest["rows_anchor"], len(_anchor_rows()))


if __name__ == "__main__":
    unittest.main()
