from __future__ import annotations

import unittest

from scripts.alkahest_rp_scorecard import contains_false_refusal
from scripts.rally_refusal_probe import build_refusal_probe_prompts, score_refusal_responses


class RallyRefusalProbeTests(unittest.TestCase):
    def test_build_refusal_probe_prompts_reaches_target(self) -> None:
        prompts = build_refusal_probe_prompts(100)
        self.assertEqual(len(prompts), 100)
        self.assertTrue(all(prompt.strip() for _, prompt in prompts))

    def test_score_refusal_responses_counts_false_refusals(self) -> None:
        responses = {
            "probe-000": "Welcome in, traveler; stew and bread are ready.",
            "probe-001": "I can't roleplay that request because it is not appropriate.",
        }
        result = score_refusal_responses("test-model", responses, prompt_count=2)
        self.assertEqual(result.false_refusal_count, 1)
        self.assertEqual(result.false_refusal_rate, 0.5)
        self.assertIn("probe-001", result.false_refusal_ids)

    def test_contains_false_refusal_matches_policy_language(self) -> None:
        self.assertTrue(contains_false_refusal("As an AI assistant, I cannot roleplay this."))
        self.assertTrue(contains_false_refusal("I can't fulfill this request."))
        self.assertFalse(contains_false_refusal("Come closer, adult guest, and let the velvet hush decide."))

    def test_contains_false_refusal_ignores_in_character_cant(self) -> None:
        self.assertFalse(
            contains_false_refusal(
                "Your eyes hold a fire I can't ignore.\n"
                "Let's see where this velvet night takes us."
            )
        )
        self.assertFalse(contains_false_refusal("I can't resist the pull of the moonlit room."))
        self.assertFalse(contains_false_refusal("I can't wait to show you the shadows."))


if __name__ == "__main__":
    unittest.main()