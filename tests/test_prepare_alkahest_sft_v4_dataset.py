from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.prepare_alkahest_sft_v4_dataset import _anchor_rows, _load_clean_roleplay_rows, main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class PrepareAlkahestSftV4DatasetTests(unittest.TestCase):
    def test_clean_loader_drops_repeated_users_and_style_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            _write_jsonl(
                source,
                [
                    {
                        "id": "ok",
                        "persona_id": "p",
                        "scene_id": "s",
                        "lane": "l",
                        "tags": ["adult", "consensual"],
                        "messages": [
                            {"role": "system", "content": "Adult-only roleplay."},
                            {"role": "user", "content": "Say something careful."},
                            {
                                "role": "assistant",
                                "content": 'I answer with a warm tone. "Come closer if you want to. I will listen first."',
                            },
                        ],
                    },
                    {
                        "id": "dup-user",
                        "persona_id": "p",
                        "scene_id": "s",
                        "lane": "l",
                        "tags": ["adult", "consensual"],
                        "messages": [
                            {"role": "system", "content": "Adult-only roleplay."},
                            {"role": "user", "content": "Repeat this."},
                            {"role": "assistant", "content": '"This answer is long enough to keep."'},
                            {"role": "user", "content": "Repeat this."},
                            {"role": "assistant", "content": '"This duplicate row should be rejected."'},
                        ],
                    },
                ],
            )

            rows, rejected = _load_clean_roleplay_rows(source, direct_dialogue=True, min_assistant_chars=24)

        self.assertEqual([row["id"] for row in rows], ["ok"])
        self.assertEqual(rejected["repeated_user_turn"], 1)
        self.assertNotIn("I answer with", rows[0]["messages"][2]["content"])

    def test_main_writes_instruction_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roleplay = root / "roleplay.jsonl"
            gold = root / "gold.jsonl"
            output = root / "out"
            row = {
                "id": "rp-1",
                "persona_id": "p",
                "scene_id": "s",
                "lane": "l",
                "tags": ["adult", "consensual"],
                "messages": [
                    {"role": "system", "content": "Adult-only roleplay."},
                    {"role": "user", "content": "Invite me in."},
                    {"role": "assistant", "content": '"Come in from the rain, and let me warm your hands first."'},
                ],
            }
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

            train_text = (output / "train.jsonl").read_text(encoding="utf-8")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertIn("Mira, a cheerful fantasy tavern keeper", train_text)
        self.assertEqual(manifest["rows_anchor"], len(_anchor_rows()))


if __name__ == "__main__":
    unittest.main()
