import unittest

from scripts.prepare_math_data import (
    build_report,
    build_target,
    clean_solution,
    convert,
    validate_target,
)


class PrepareMathDataTests(unittest.TestCase):
    def test_clean_solution_removes_calculation_markers_and_hash_answer(self):
        solution = (
            "There are 3 <<2+1=3>> items.\n"
            "So the result is 3.\n"
            "#### 3"
        )

        self.assertEqual(
            clean_solution(solution),
            "There are 3  items.\nSo the result is 3.",
        )

    def test_build_target_and_validation_agree(self):
        target = build_target("Add the two values.", "42")

        self.assertEqual(
            target,
            "<think>\nAdd the two values.\n</think>\n\n\\boxed{42}",
        )
        self.assertEqual(
            validate_target(target, " 42 "),
            {
                "format_valid": True,
                "answer_extracted": True,
                "answer_match": True,
            },
        )

    def test_convert_preserves_metadata_and_builds_valid_report(self):
        samples = [
            {
                "problem_id": "gsm8k-1",
                "source": "gsm8k",
                "domain": "math",
                "split": "train",
                "problem": "What is 20 + 22?",
                "solution": "Add the values. <<20+22=42>>\n#### 42",
                "answer": "42",
            }
        ]

        records = convert(samples)
        report = build_report(records)

        self.assertEqual(records[0]["metadata"]["problem_id"], "gsm8k-1")
        self.assertNotIn("<<", records[0]["target"])
        self.assertEqual(report["num_samples"], 1)
        self.assertEqual(report["format_valid_rate"], 1.0)
        self.assertEqual(report["answer_extraction_success_rate"], 1.0)
        self.assertEqual(report["invalid_samples_count"], 0)
        self.assertEqual(report["split_distribution"], {"train": 1})

    def test_build_report_handles_empty_records(self):
        report = build_report([])

        self.assertEqual(report["num_samples"], 0)
        self.assertEqual(report["format_valid_rate"], 0)
        self.assertEqual(report["answer_extraction_success_rate"], 0)
        self.assertEqual(report["max_target_chars"], 0)


if __name__ == "__main__":
    unittest.main()
