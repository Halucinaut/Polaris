"""Tests for M2.5 Math reward function.

Covers: correct, incorrect, equivalent answers (fractions), unparseable,
empty output, format bonus, and structured reward breakdown.
"""

import unittest

from polaris.rewards.math_answer import RewardResult, compute_math_reward


DEFAULT_REWARD_CONFIG = {
    "correct": 1.0,
    "incorrect": 0.0,
    "unparseable": -0.5,
    "empty": -1.0,
    "format_bonus": 0.05,
}


class TestComputeMathReward(unittest.TestCase):

    # --- Correct answers ---

    def test_correct_answer_with_think_and_boxed(self):
        completion = "<think>2+3=5</think>\n\\boxed{5}"
        result = compute_math_reward(completion, "5", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.extracted_answer, "5")
        self.assertEqual(result.extraction_method, "boxed")
        self.assertTrue(result.format_adherent)
        self.assertAlmostEqual(result.reward, 1.05)  # correct + format_bonus
        self.assertIsNone(result.invalid_reason)

    def test_correct_answer_numeric_equivalence(self):
        # 1/2 == 0.5 via fraction comparison
        completion = "<think>...</think>\n\\boxed{\\frac{1}{2}}"
        result = compute_math_reward(completion, "0.5", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)

    def test_correct_answer_thousands_separator(self):
        completion = "<think>...</think>\n\\boxed{1,000}"
        result = compute_math_reward(completion, "1000", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)

    # --- Incorrect answers ---

    def test_incorrect_answer(self):
        completion = "<think>2+3=6</think>\n\\boxed{6}"
        result = compute_math_reward(completion, "5", DEFAULT_REWARD_CONFIG)
        self.assertFalse(result.answer_correct)
        self.assertEqual(result.extraction_method, "boxed")
        self.assertAlmostEqual(result.reward, 0.05)  # incorrect + format_bonus

    def test_incorrect_answer_no_format(self):
        completion = "\\boxed{6}"
        result = compute_math_reward(completion, "5", DEFAULT_REWARD_CONFIG)
        self.assertFalse(result.answer_correct)
        self.assertFalse(result.format_adherent)
        self.assertAlmostEqual(result.reward, 0.0)  # incorrect, no format

    # --- Unparseable ---

    def test_unparseable_output(self):
        completion = "I think the answer is probably around five"
        result = compute_math_reward(completion, "5", DEFAULT_REWARD_CONFIG)
        self.assertIsNone(result.answer_correct)
        self.assertIsNone(result.extracted_answer)
        self.assertEqual(result.invalid_reason, "unparseable")
        self.assertAlmostEqual(result.reward, -0.5)

    def test_unparseable_with_format_bonus(self):
        # Has think tags but no extractable answer after them
        completion = "<think>Let me think... I'm not sure</think>\nThe answer is unclear"
        result = compute_math_reward(completion, "5", DEFAULT_REWARD_CONFIG)
        self.assertIsNone(result.answer_correct)
        self.assertEqual(result.invalid_reason, "unparseable")
        # format_bonus not awarded because no boxed answer after think
        self.assertFalse(result.format_adherent)

    # --- Empty output ---

    def test_empty_output(self):
        result = compute_math_reward("", "5", DEFAULT_REWARD_CONFIG)
        self.assertIsNone(result.answer_correct)
        self.assertEqual(result.invalid_reason, "empty_output")
        self.assertAlmostEqual(result.reward, -1.0)

    def test_whitespace_only_output(self):
        result = compute_math_reward("   \n  ", "5", DEFAULT_REWARD_CONFIG)
        self.assertEqual(result.invalid_reason, "empty_output")
        self.assertAlmostEqual(result.reward, -1.0)

    # --- Format bonus ---

    def test_format_bonus_only_when_adherent(self):
        # Good format
        r1 = compute_math_reward(
            "<think>Work</think>\n\\boxed{42}", "42", DEFAULT_REWARD_CONFIG,
        )
        self.assertTrue(r1.format_adherent)
        self.assertAlmostEqual(r1.reward_breakdown["format_bonus"], 0.05)

        # No think block
        r2 = compute_math_reward("\\boxed{42}", "42", DEFAULT_REWARD_CONFIG)
        self.assertFalse(r2.format_adherent)
        self.assertAlmostEqual(r2.reward_breakdown["format_bonus"], 0.0)

    # --- Reward breakdown structure ---

    def test_reward_breakdown_keys(self):
        result = compute_math_reward(
            "<think>Work</think>\n\\boxed{10}", "10", DEFAULT_REWARD_CONFIG,
        )
        self.assertIn("base", result.reward_breakdown)
        self.assertIn("format_bonus", result.reward_breakdown)

    # --- Custom reward config ---

    def test_custom_reward_values(self):
        custom = {
            "correct": 2.0,
            "incorrect": -1.0,
            "unparseable": -2.0,
            "empty": -3.0,
            "format_bonus": 0.1,
        }
        result = compute_math_reward(
            "<think>OK</think>\n\\boxed{7}", "7", custom,
        )
        self.assertAlmostEqual(result.reward, 2.1)

    # --- Equivalent answers ---

    def test_equivalent_fraction_answers(self):
        completion = "<think>...</think>\n\\boxed{2/4}"
        result = compute_math_reward(completion, "1/2", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)

    def test_latex_fraction_vs_plain(self):
        completion = "<think>...</think>\n\\boxed{\\frac{3}{6}}"
        result = compute_math_reward(completion, "1/2", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)

    # --- Answer tag extraction ---

    def test_answer_tag_extraction(self):
        completion = "<think>Work</think>\n<answer>42</answer>"
        result = compute_math_reward(completion, "42", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.extraction_method, "answer_tag")

    # --- Hash answer extraction ---

    def test_hash_answer_extraction(self):
        completion = "<think>Work</think>\n#### 42"
        result = compute_math_reward(completion, "42", DEFAULT_REWARD_CONFIG)
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.extraction_method, "hash_answer")


if __name__ == "__main__":
    unittest.main()
