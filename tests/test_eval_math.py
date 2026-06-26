import unittest

from scripts.eval_math import (
    answers_match,
    extract_predicted_answer,
    has_m1_format_adherence,
)


class EvalMathTests(unittest.TestCase):
    def test_answers_match_ignores_thousands_separators(self):
        self.assertTrue(answers_match("1,000", "1000"))
        self.assertTrue(answers_match(" 12,345 ", "12345"))

    def test_answers_match_accepts_equivalent_fractions(self):
        self.assertTrue(answers_match("2/4", "1/2"))
        self.assertFalse(answers_match("2/3", "1/2"))

    def test_answers_match_accepts_latex_fractions(self):
        self.assertTrue(answers_match("\\frac{2}{4}", "1/2"))
        self.assertTrue(answers_match("\\dfrac{3}{6}", "1/2"))

    def test_extract_predicted_answer_handles_nested_boxed_braces(self):
        prediction = "<think>Work</think>\n\\boxed{\\frac{1}{2}}"

        self.assertEqual(
            extract_predicted_answer(prediction),
            ("\\frac{1}{2}", "boxed"),
        )

    def test_extract_predicted_answer_uses_post_think_answer(self):
        prediction = (
            "<think>The intermediate result is \\boxed{7}.</think>\n"
            "\\boxed{42}"
        )

        self.assertEqual(extract_predicted_answer(prediction), ("42", "boxed"))

    def test_format_adherence_requires_boxed_answer_after_think(self):
        self.assertTrue(
            has_m1_format_adherence("<think>Work</think>\n\\boxed{42}")
        )
        self.assertFalse(
            has_m1_format_adherence("<think>Work with \\boxed{42}</think>")
        )


if __name__ == "__main__":
    unittest.main()
