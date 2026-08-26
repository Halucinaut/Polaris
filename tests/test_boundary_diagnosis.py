"""Tests for boundary diagnosis and binary prefix DPO data."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_style_dpo import check_style_adherence


class TestStyleCheckConsistency(unittest.TestCase):
    """Verify that the style check used in diagnosis matches eval_style_dpo."""

    def test_valid_template_passes(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. Step one.\n"
            "2. Step two.\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        adherent, reasons = check_style_adherence(text)
        self.assertTrue(adherent)
        self.assertEqual(reasons, [])

    def test_missing_solution_prefix_fails(self):
        text = (
            "<think>\n"
            "Step one.\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        adherent, reasons = check_style_adherence(text)
        self.assertFalse(adherent)
        self.assertIn("missing_solution_prefix", reasons)

    def test_missing_final_fails(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. Step one.\n"
            "</think>\n\n"
            "\\boxed{42}"
        )
        adherent, reasons = check_style_adherence(text)
        self.assertFalse(adherent)
        self.assertIn("invalid_final_template", reasons)

    def test_steps_not_contiguous_fails(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. Step one.\n"
            "3. Step three.\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        adherent, reasons = check_style_adherence(text)
        self.assertFalse(adherent)
        self.assertIn("steps_not_contiguous", reasons)

    def test_nested_boxed_passes(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. Compute.\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{\\frac{3}{4}}."
        )
        adherent, reasons = check_style_adherence(text)
        self.assertTrue(adherent)

    def test_forced_prefix_maintains_structure(self):
        """A complete forced-prefix generation should pass if well-formed."""
        text = (
            "<think>\nSolution:\n"
            "1. First, compute 2 + 3 = 5.\n"
            "2. The answer is 5.\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{5}."
        )
        adherent, reasons = check_style_adherence(text)
        self.assertTrue(adherent)


class TestBinaryPrefixDataExists(unittest.TestCase):
    """Check that binary prefix data can be produced (requires M1 model)."""

    DATA_PATH = "data/math/pilots/binary_prefix_dpo_control_480.jsonl"

    def test_data_exists(self):
        if not os.path.exists(self.DATA_PATH):
            self.skipTest("Binary prefix data not generated yet")
        import json
        with open(self.DATA_PATH, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(records), 480)

    def test_records_have_required_fields(self):
        if not os.path.exists(self.DATA_PATH):
            self.skipTest("Binary prefix data not generated yet")
        import json
        with open(self.DATA_PATH, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        for r in records[:5]:
            self.assertIn("problem_id", r)
            self.assertIn("chosen", r)
            self.assertIn("rejected", r)
            self.assertIn("metadata", r)
            meta = r["metadata"]
            self.assertIn("chosen_token_id", meta)
            self.assertIn("rejected_token_id", meta)
            self.assertIn("chosen_token_logprob_m1", meta)
            self.assertIn("rejected_token_logprob_m1", meta)
            self.assertIn("re_encoding_consistent", meta)


if __name__ == "__main__":
    unittest.main()
