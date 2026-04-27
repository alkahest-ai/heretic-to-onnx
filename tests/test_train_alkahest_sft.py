from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from train_alkahest_sft import _messages_to_features  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        if tokenize or add_generation_prompt:
            raise AssertionError("test fake only renders complete chat text")
        return "".join(f"<|{message['role']}|>\n{message['content']}\n" for message in messages)

    def __call__(self, text: str, *, truncation: bool, max_length: int, return_offsets_mapping: bool = False):
        input_ids = [ord(ch) for ch in text[:max_length]]
        encoded = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(input_ids))]
        return encoded


class TrainAlkahestSftTests(unittest.TestCase):
    def test_assistant_only_loss_masks_user_and_system_tokens(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "welcome"},
            ]
        }

        features = _messages_to_features(
            row,
            FakeTokenizer(),
            max_length=200,
            assistant_only_loss=True,
        )
        rendered = "".join(chr(token_id) for token_id in features["input_ids"])
        labeled = "".join(
            chr(label) for label in features["labels"] if label != -100
        )

        self.assertIn("system prompt", rendered)
        self.assertIn("hello", rendered)
        self.assertEqual(labeled, "welcome")

    def test_full_chat_loss_leaves_labels_to_collator(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "welcome"},
            ]
        }

        features = _messages_to_features(
            row,
            FakeTokenizer(),
            max_length=200,
            assistant_only_loss=False,
        )

        self.assertNotIn("labels", features)
        self.assertNotIn("offset_mapping", features)


if __name__ == "__main__":
    unittest.main()
