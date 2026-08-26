"""Tests for split_dpo_v2_style.py validation and classification logic."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from split_dpo_v2_style import (
    answers_match,
    classify_record,
    extract_boxed_answer,
    load_jsonl,
    main,
    normalize_answer,
    parse_numeric,
    validate_record,
)


class TestNormalizeAnswer(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(normalize_answer("  42  "), "42")

    def test_lowercases(self):
        self.assertEqual(normalize_answer("ABC"), "abc")

    def test_removes_commas(self):
        self.assertEqual(normalize_answer("1,234"), "1234")

    def test_removes_spaces(self):
        self.assertEqual(normalize_answer("1 2 3"), "123")


class TestParseNumeric(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(parse_numeric("42"), 42)

    def test_fraction(self):
        self.assertEqual(parse_numeric("3/4"), 3 / 4)

    def test_latex_frac(self):
        self.assertEqual(parse_numeric(r"\frac{3}{4}"), 3 / 4)

    def test_latex_dfrac(self):
        self.assertEqual(parse_numeric(r"\dfrac{3}{4}"), 3 / 4)

    def test_invalid(self):
        self.assertIsNone(parse_numeric("not_a_number"))


class TestAnswersMatch(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(answers_match("42", "42"))

    def test_normalized_match(self):
        self.assertTrue(answers_match(" 42 ", "42"))

    def test_comma_match(self):
        self.assertTrue(answers_match("1,234", "1234"))

    def test_fraction_equivalence(self):
        self.assertTrue(answers_match("6/8", "3/4"))

    def test_mismatch(self):
        self.assertFalse(answers_match("42", "43"))


class TestExtractBoxedAnswer(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(extract_boxed_answer(r"\boxed{42}"), "42")

    def test_nested(self):
        self.assertEqual(extract_boxed_answer(r"\boxed{\frac{3}{4}}"), r"\frac{3}{4}")

    def test_multiple_takes_last(self):
        text = r"\boxed{wrong}\boxed{correct}"
        self.assertEqual(extract_boxed_answer(text), "correct")

    def test_missing(self):
        self.assertIsNone(extract_boxed_answer("no boxed answer here"))


class TestValidateRecord(unittest.TestCase):
    """Test the full validation protocol."""

    def _make_record(
        self,
        chosen: str,
        rejected: str,
        answer: str = "42",
        problem_id: str = "test_001",
    ) -> dict:
        return {
            "chosen": chosen,
            "rejected": rejected,
            "answer": answer,
            "metadata": {"problem_id": problem_id},
        }

    def _valid_chosen(self, answer: str = "42") -> str:
        return (
            "<think>\n"
            "Solution:\n"
            "1. Compute 40 + 2 = 42\n"
            "2. Verify: 42 is correct\n"
            "</think>\n\n"
            f"Final: The answer is \\boxed{{{answer}}}."
        )

    def _valid_rejected(self, answer: str = "42") -> str:
        return (
            "<think>\n"
            "The answer is computed as follows:\n"
            "40 + 2 = 42\n"
            "</think>\n\n"
            f"\\boxed{{{answer}}}"
        )

    def test_valid_record_passes(self):
        record = self._make_record(
            chosen=self._valid_chosen(),
            rejected=self._valid_rejected(),
        )
        result = validate_record(record)
        self.assertTrue(result["ok"], f"Expected ok but got errors: {result['errors']}")

    def test_chosen_boxed_mismatch(self):
        record = self._make_record(
            chosen=self._valid_chosen("99"),
            rejected=self._valid_rejected(),
        )
        result = validate_record(record)
        self.assertIn("chosen_boxed_answer_mismatch", result["errors"])

    def test_rejected_boxed_mismatch(self):
        record = self._make_record(
            chosen=self._valid_chosen(),
            rejected=self._valid_rejected("99"),
        )
        result = validate_record(record)
        self.assertIn("rejected_boxed_answer_mismatch", result["errors"])

    def test_missing_solution_prefix(self):
        chosen = "<think>\n40 + 2 = 42\n</think>\n\nFinal: The answer is \\boxed{42}."
        record = self._make_record(chosen=chosen, rejected=self._valid_rejected())
        result = validate_record(record)
        self.assertIn("chosen_missing_solution_prefix", result["errors"])

    def test_invalid_think_tags(self):
        chosen = "<think>\nSolution:\n1. Step one\n</think> extra <think>\n</think>\n\nFinal: The answer is \\boxed{42}."
        record = self._make_record(chosen=chosen, rejected=self._valid_rejected())
        result = validate_record(record)
        self.assertIn("chosen_invalid_think_tags", result["errors"])

    def test_steps_not_contiguous(self):
        chosen = (
            "<think>\n"
            "Solution:\n"
            "1. First step\n"
            "3. Skipped step 2\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        record = self._make_record(chosen=chosen, rejected=self._valid_rejected())
        result = validate_record(record)
        self.assertIn("chosen_steps_not_contiguous", result["errors"])

    def test_invalid_final_template(self):
        chosen = "<think>\nSolution:\n1. Step one\n</think>\nThe answer is 42."
        record = self._make_record(chosen=chosen, rejected=self._valid_rejected())
        result = validate_record(record)
        self.assertIn("chosen_invalid_final_template", result["errors"])

    def test_length_ratio_out_of_range_too_short(self):
        chosen = "<think>\nSolution:\n1. X\n</think>\n\nFinal: The answer is \\boxed{42}."
        rejected = self._valid_rejected() * 3
        record = self._make_record(chosen=chosen, rejected=rejected)
        result = validate_record(record)
        self.assertIn("chosen_length_ratio_out_of_range", result["errors"])

    def test_length_ratio_out_of_range_too_long(self):
        chosen = self._valid_chosen() * 3
        rejected = "<think>\nX\n</think>\n\n\\boxed{42}"
        record = self._make_record(chosen=chosen, rejected=rejected)
        result = validate_record(record)
        self.assertIn("chosen_length_ratio_out_of_range", result["errors"])

    def test_similarity_too_high(self):
        text = self._valid_chosen()
        record = self._make_record(chosen=text, rejected=text)
        result = validate_record(record)
        self.assertIn("chosen_too_similar_to_rejected", result["errors"])

    def test_diagnostics_populated(self):
        record = self._make_record(
            chosen=self._valid_chosen(),
            rejected=self._valid_rejected(),
        )
        result = validate_record(record)
        self.assertEqual(result["problem_id"], "test_001")
        self.assertEqual(result["chosen_answer"], "42")
        self.assertEqual(result["rejected_answer"], "42")
        self.assertGreater(result["chosen_estimated_token_length"], 0)
        self.assertGreater(result["rejected_estimated_token_length"], 0)
        self.assertGreater(result["length_ratio"], 0)
        self.assertGreater(result["character_similarity"], 0)
        self.assertEqual(result["step_count"], 2)


class TestClassifyRecord(unittest.TestCase):
    """Test the split classification logic with exact set matching."""

    def _make_record(
        self,
        chosen: str,
        rejected: str,
        answer: str = "42",
        problem_id: str = "test_001",
    ) -> dict:
        return {
            "chosen": chosen,
            "rejected": rejected,
            "answer": answer,
            "metadata": {"problem_id": problem_id},
        }

    def _valid_chosen(self, answer: str = "42") -> str:
        return (
            "<think>\n"
            "Solution:\n"
            "1. Compute 40 + 2 = 42\n"
            "2. Verify: 42 is correct\n"
            "</think>\n\n"
            f"Final: The answer is \\boxed{{{answer}}}."
        )

    def _valid_rejected(self, answer: str = "42") -> str:
        return (
            "<think>\n"
            "The answer is computed as follows:\n"
            "40 + 2 = 42\n"
            "</think>\n\n"
            f"\\boxed{{{answer}}}"
        )

    def test_valid_goes_to_train(self):
        """Empty errors → train."""
        record = self._make_record(
            chosen=self._valid_chosen(),
            rejected=self._valid_rejected(),
        )
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "train")
        self.assertEqual(result["errors"], [])

    def test_length_ratio_only_goes_to_stress(self):
        """errors == {chosen_length_ratio_out_of_range} → stress."""
        chosen = self._valid_chosen()
        rejected = "<think>\nX\n</think>\n\n\\boxed{42}"
        chosen_long = chosen.replace(
            "1. Compute 40 + 2 = 42",
            "1. Compute 40 + 2 = 42\n" + "Additional reasoning step. " * 20,
        )
        record = self._make_record(chosen=chosen_long, rejected=rejected)
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "stress")
        self.assertEqual(set(result["errors"]), {"chosen_length_ratio_out_of_range"})

    def test_similarity_only_goes_to_quarantine(self):
        """errors == {chosen_too_similar_to_rejected} → quarantine."""
        # Nearly identical chosen/rejected, but valid answers and template
        text = self._valid_chosen()
        record = self._make_record(chosen=text, rejected=text)
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "quarantine")
        self.assertIn("chosen_too_similar_to_rejected", result["errors"])

    def test_similarity_and_length_combined_goes_to_quarantine(self):
        """When both similarity and length errors would fire, quarantine is still correct.

        Note: mathematically, ratio > 1.60 implies similarity < 0.97 (SequenceMatcher
        upper bound is 2/(1+ratio)), so both errors cannot co-occur in practice.
        This test verifies that a length-only error still routes to quarantine
        (since the error set is {chosen_length_ratio_out_of_range} which is stress,
        but we also test the quarantine path for combined template+length errors).
        """
        # Template error + length error → quarantine (not stress)
        extra_steps = "\n".join(f"{i}. Extra reasoning step {i}." for i in range(2, 22))
        chosen = f"<think>\n1. Compute 40 + 2 = 42\n{extra_steps}\n</think>\n\nFinal: The answer is \\boxed{{42}}."
        rejected = "<think>\nX\n</think>\n\n\\boxed{42}"
        record = self._make_record(chosen=chosen, rejected=rejected)
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "quarantine")
        errors = set(result["errors"])
        self.assertIn("chosen_missing_solution_prefix", errors)
        self.assertIn("chosen_length_ratio_out_of_range", errors)

    def test_answer_mismatch_goes_to_quarantine(self):
        """Critical error → quarantine."""
        record = self._make_record(
            chosen=self._valid_chosen("99"),
            rejected=self._valid_rejected(),
        )
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "quarantine")
        self.assertIn("chosen_boxed_answer_mismatch", result["errors"])

    def test_template_issue_goes_to_quarantine(self):
        """Template error → quarantine."""
        chosen = "<think>\n40 + 2 = 42\n</think>\n\nFinal: The answer is \\boxed{42}."
        record = self._make_record(chosen=chosen, rejected=self._valid_rejected())
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "quarantine")
        self.assertIn("chosen_missing_solution_prefix", result["errors"])

    def test_rejected_mismatch_goes_to_quarantine(self):
        """rejected_boxed_answer_mismatch alone → quarantine."""
        record = self._make_record(
            chosen=self._valid_chosen(),
            rejected=self._valid_rejected("99"),
        )
        split_name, result = classify_record(record)
        self.assertEqual(split_name, "quarantine")
        self.assertIn("rejected_boxed_answer_mismatch", result["errors"])


class TestIdempotency(unittest.TestCase):
    """Verify that validation is deterministic."""

    def _make_record(self) -> dict:
        return {
            "chosen": (
                "<think>\n"
                "Solution:\n"
                "1. Compute 40 + 2 = 42\n"
                "2. Verify: 42 is correct\n"
                "</think>\n\n"
                "Final: The answer is \\boxed{42}."
            ),
            "rejected": (
                "<think>\n"
                "The answer is computed as follows:\n"
                "40 + 2 = 42\n"
                "</think>\n\n"
                "\\boxed{42}"
            ),
            "answer": "42",
            "metadata": {"problem_id": "test_001"},
        }

    def test_deterministic_result(self):
        record = self._make_record()
        result1 = validate_record(record)
        result2 = validate_record(record)
        self.assertEqual(result1, result2)

    def test_deterministic_classification(self):
        record = self._make_record()
        split1, _ = classify_record(record)
        split2, _ = classify_record(record)
        self.assertEqual(split1, split2)


class TestMainIntegration(unittest.TestCase):
    """Integration tests for the main split function, using temp directories."""

    VALID_CHOSEN = (
        "<think>\n"
        "Solution:\n"
        "1. Compute 40 + 2 = 42\n"
        "2. Verify: 42 is correct\n"
        "</think>\n\n"
        "Final: The answer is \\boxed{42}."
    )
    VALID_REJECTED = (
        "<think>\n"
        "The answer is computed as follows:\n"
        "40 + 2 = 42\n"
        "</think>\n\n"
        "\\boxed{42}"
    )

    def _make_valid_record(self, problem_id: str, answer: str = "42") -> dict:
        return {
            "id": f"dpo_style_{problem_id}",
            "messages": [
                {"role": "system", "content": "You are a math assistant."},
                {"role": "user", "content": "What is 40 + 2?"},
            ],
            "chosen": self.VALID_CHOSEN,
            "rejected": self.VALID_REJECTED,
            "answer": answer,
            "pair_type": "correct_style_preference",
            "quality_tag": "auto_validated_pending_human_review",
            "metadata": {
                "problem_id": problem_id,
                "source_dataset": "sft_d5_500",
            },
        }

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _run_main(self, input_path, train_path, stress_path, quarantine_path, report_path):
        """Run main() with patched argv, capture stdout, return (exit_code, stdout_text)."""
        import unittest.mock
        buf = io.StringIO()
        with unittest.mock.patch(
            "sys.argv",
            [
                "split_dpo_v2_style.py",
                "--input", str(input_path),
                "--train-path", str(train_path),
                "--stress-path", str(stress_path),
                "--quarantine-path", str(quarantine_path),
                "--report-path", str(report_path),
            ],
        ), redirect_stdout(buf):
            result = main()
        return result, buf.getvalue()

    def test_duplicate_problem_id_aborts_no_files_written(self):
        """Exactly 500 inputs, 499 unique + 1 duplicate → abort with 'duplicate problem_id', no files written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.jsonl"
            train_path = tmpdir / "train.jsonl"
            stress_path = tmpdir / "stress.jsonl"
            quarantine_path = tmpdir / "quarantine.jsonl"
            report_path = tmpdir / "report.json"

            # 499 unique records + 1 duplicate of the first → 500 total, 499 unique pids
            records = [self._make_valid_record(f"gsm8k_train_d5_{i:04d}") for i in range(499)]
            records.append(self._make_valid_record("gsm8k_train_d5_0000"))  # duplicate
            self.assertEqual(len(records), 500)
            self._write_jsonl(input_path, records)

            result, stdout = self._run_main(input_path, train_path, stress_path, quarantine_path, report_path)

            self.assertEqual(result, 1)
            self.assertIn("duplicate problem_id", stdout)
            self.assertFalse(train_path.exists())
            self.assertFalse(stress_path.exists())
            self.assertFalse(quarantine_path.exists())
            self.assertFalse(report_path.exists())

    def test_wrong_input_count_aborts_no_files_written(self):
        """Input count != 500 should cause abort without writing any output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.jsonl"
            train_path = tmpdir / "train.jsonl"
            stress_path = tmpdir / "stress.jsonl"
            quarantine_path = tmpdir / "quarantine.jsonl"
            report_path = tmpdir / "report.json"

            records = [
                self._make_valid_record("gsm8k_train_d5_0001"),
                self._make_valid_record("gsm8k_train_d5_0002"),
                self._make_valid_record("gsm8k_train_d5_0003"),
            ]
            self._write_jsonl(input_path, records)

            result, stdout = self._run_main(input_path, train_path, stress_path, quarantine_path, report_path)

            self.assertEqual(result, 1)
            self.assertIn("expected 500 input records", stdout)
            self.assertFalse(train_path.exists())
            self.assertFalse(stress_path.exists())
            self.assertFalse(quarantine_path.exists())
            self.assertFalse(report_path.exists())

    def test_count_mismatch_aborts_no_files_written(self):
        """Split counts != expected (449/50/1) should abort without writing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.jsonl"
            train_path = tmpdir / "train.jsonl"
            stress_path = tmpdir / "stress.jsonl"
            quarantine_path = tmpdir / "quarantine.jsonl"
            report_path = tmpdir / "report.json"

            # 500 valid records → all go to train (500/0/0 ≠ 449/50/1)
            records = [self._make_valid_record(f"gsm8k_train_d5_{i:04d}") for i in range(500)]
            self._write_jsonl(input_path, records)

            result, stdout = self._run_main(input_path, train_path, stress_path, quarantine_path, report_path)

            self.assertEqual(result, 1)
            self.assertIn("train expected 449, got 500", stdout)
            self.assertFalse(train_path.exists())
            self.assertFalse(stress_path.exists())
            self.assertFalse(quarantine_path.exists())
            self.assertFalse(report_path.exists())

    def test_custom_path_already_exists_refuses_overwrite(self):
        """Any output path already existing → return 1, no files modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.jsonl"
            train_path = tmpdir / "train.jsonl"
            stress_path = tmpdir / "stress.jsonl"
            quarantine_path = tmpdir / "quarantine.jsonl"
            report_path = tmpdir / "report.json"

            # Create the input with 500 valid records
            records = [self._make_valid_record(f"gsm8k_train_d5_{i:04d}") for i in range(500)]
            self._write_jsonl(input_path, records)

            # Pre-create one output file with sentinel content
            sentinel = '{"sentinel": true}\n'
            stress_path.write_text(sentinel)

            result, stdout = self._run_main(input_path, train_path, stress_path, quarantine_path, report_path)

            self.assertEqual(result, 1)
            self.assertIn("already exists", stdout)
            # Sentinel file unchanged
            self.assertEqual(stress_path.read_text(), sentinel)
            # No other output files created
            self.assertFalse(train_path.exists())
            self.assertFalse(quarantine_path.exists())
            self.assertFalse(report_path.exists())

    def test_reproducibility_with_real_data(self):
        """Run split on real source data to temp dir; output id order must match official splits."""
        import unittest.mock

        real_input = Path("transfer/style_dpo_v2_returned/dpo_v2_style_controlled.jsonl")
        if not real_input.exists():
            self.skipTest("Real input file not available")

        # Load official split ids
        official_ids: dict[str, list[str]] = {}
        for name, count in [("train", 449), ("stress", 50)]:
            path = Path(f"data/math/splits/dpo_v2_style_{name}_{count}.jsonl")
            if not path.exists():
                self.skipTest(f"Official split {path} not available")
            with path.open() as f:
                official_ids[name] = [json.loads(line)["id"] for line in f]
        quarantine_path = Path("data/math/quarantine/dpo_v2_style_invalid_1.jsonl")
        if not quarantine_path.exists():
            self.skipTest("Official quarantine split not available")
        with quarantine_path.open() as f:
            official_ids["quarantine"] = [json.loads(line)["id"] for line in f]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            out_train = tmpdir / "train.jsonl"
            out_stress = tmpdir / "stress.jsonl"
            out_quarantine = tmpdir / "quarantine.jsonl"
            out_report = tmpdir / "report.json"

            buf = io.StringIO()
            with unittest.mock.patch(
                "sys.argv",
                [
                    "split_dpo_v2_style.py",
                    "--input", str(real_input),
                    "--train-path", str(out_train),
                    "--stress-path", str(out_stress),
                    "--quarantine-path", str(out_quarantine),
                    "--report-path", str(out_report),
                ],
            ), redirect_stdout(buf):
                result = main()

            self.assertEqual(result, 0, f"main() returned {result}:\n{buf.getvalue()}")

            for name, out_path in [("train", out_train), ("stress", out_stress), ("quarantine", out_quarantine)]:
                with out_path.open() as f:
                    actual_ids = [json.loads(line)["id"] for line in f]
                self.assertEqual(actual_ids, official_ids[name], f"{name} id order mismatch")


if __name__ == "__main__":
    unittest.main()
