"""Tests for eval_style_dpo.py — style adherence and answer correctness evaluation."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_style_dpo import (
    answers_match,
    check_style_adherence,
    evaluate_sample,
    extract_boxed_answer,
    main,
    resolve_answer,
    resolve_problem_id,
)


class TestExtractBoxedAnswer(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(extract_boxed_answer(r"\boxed{42}"), "42")

    def test_nested(self):
        self.assertEqual(extract_boxed_answer(r"\boxed{\frac{3}{4}}"), r"\frac{3}{4}")

    def test_missing(self):
        self.assertIsNone(extract_boxed_answer("no boxed here"))

    def test_last_wins(self):
        self.assertEqual(extract_boxed_answer(r"\boxed{a}\boxed{b}"), "b")

    def test_empty_content(self):
        """Empty braces → empty string."""
        self.assertEqual(extract_boxed_answer(r"\boxed{}"), "")


class TestAnswersMatch(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(answers_match("42", "42"))

    def test_comma(self):
        self.assertTrue(answers_match("1,234", "1234"))

    def test_fraction(self):
        self.assertTrue(answers_match("6/8", "3/4"))

    def test_latex(self):
        self.assertTrue(answers_match(r"\frac{3}{4}", "3/4"))

    def test_mismatch(self):
        self.assertFalse(answers_match("42", "43"))


class TestResolveProblemId(unittest.TestCase):
    def test_top_level(self):
        self.assertEqual(resolve_problem_id({"problem_id": "p1"}), "p1")

    def test_metadata_fallback(self):
        self.assertEqual(resolve_problem_id({"metadata": {"problem_id": "p1"}}), "p1")

    def test_top_level_none_falls_back(self):
        self.assertEqual(
            resolve_problem_id({"problem_id": None, "metadata": {"problem_id": "p1"}}),
            "p1",
        )

    def test_top_level_empty_falls_back(self):
        self.assertEqual(
            resolve_problem_id({"problem_id": "", "metadata": {"problem_id": "p1"}}),
            "p1",
        )

    def test_top_level_whitespace_falls_back(self):
        self.assertEqual(
            resolve_problem_id({"problem_id": "   ", "metadata": {"problem_id": "p1"}}),
            "p1",
        )

    def test_null_both_returns_empty(self):
        self.assertEqual(resolve_problem_id({"problem_id": None}), "")

    def test_empty_both_returns_empty(self):
        self.assertEqual(resolve_problem_id({}), "")

    def test_whitespace_metadata_returns_empty(self):
        self.assertEqual(
            resolve_problem_id({"problem_id": None, "metadata": {"problem_id": "  "}}),
            "",
        )


class TestResolveAnswer(unittest.TestCase):
    def test_plain_answer(self):
        self.assertEqual(resolve_answer({"answer": "42"}), "42")

    def test_gsm8k_hash_format(self):
        self.assertEqual(resolve_answer({"answer": "reasoning\n#### 42"}), "42")

    def test_gsm8k_hash_with_spaces(self):
        self.assertEqual(resolve_answer({"answer": "step 1\nstep 2\n#### 1234"}), "1234")

    def test_gsm8k_hash_multiple_takes_last(self):
        self.assertEqual(resolve_answer({"answer": "#### first\n#### second"}), "second")

    def test_metadata_fallback(self):
        self.assertEqual(
            resolve_answer({"metadata": {"answer": "99"}}),
            "99",
        )


class TestCheckStyleAdherence(unittest.TestCase):
    """Test the DPO v2 style template checker."""

    def _compliant(self, answer: str = "42") -> str:
        return (
            "<think>\n"
            "Solution:\n"
            "1. Compute 40 + 2 = 42\n"
            "2. Verify: 42 is correct\n"
            "</think>\n\n"
            f"Final: The answer is \\boxed{{{answer}}}."
        )

    def test_compliant_passes(self):
        ok, reasons = check_style_adherence(self._compliant())
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_compliant_with_fraction(self):
        ok, reasons = check_style_adherence(self._compliant(r"\frac{3}{4}"))
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_missing_solution_prefix(self):
        text = "<think>\n40 + 2 = 42\n</think>\n\nFinal: The answer is \\boxed{42}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("missing_solution_prefix", reasons)

    def test_steps_not_from_1(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "2. Second step first\n"
            "3. Third step\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("steps_not_contiguous", reasons)

    def test_steps_skip_number(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. First\n"
            "3. Skipped 2\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("steps_not_contiguous", reasons)

    def test_bad_final_template(self):
        text = "<think>\nSolution:\n1. Step one\n</think>\nThe answer is 42."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_missing_think_close(self):
        text = "<think>\nSolution:\n1. Step one\nFinal: The answer is \\boxed{42}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_think_tags", reasons)

    def test_duplicate_think_tag(self):
        text = "<think>\nSolution:\n1. Step\n<think>\n</think>\n\nFinal: The answer is \\boxed{42}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_think_tags", reasons)

    def test_tags_reversed(self):
        text = "Solution:\n1. Step\n</think>\n<think>\n\nFinal: The answer is \\boxed{42}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("missing_solution_prefix", reasons)

    def test_multiple_failures_reported(self):
        """Several independent failures → all reported."""
        text = "bad prefix\n1. Step\n"
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("missing_solution_prefix", reasons)
        self.assertIn("invalid_think_tags", reasons)
        self.assertIn("invalid_final_template", reasons)

    # --- Strict brace-depth boxed validation ---

    def test_malformed_boxed_extra_closing_brace(self):
        r"""\boxed{42}} — extra closing brace after boxed → invalid."""
        text = "<think>\nSolution:\n1. Step\n</think>\n\nFinal: The answer is \\boxed{42}}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_malformed_boxed_space_before_closing(self):
        r"""\boxed{42} } — space + brace after boxed → invalid."""
        text = "<think>\nSolution:\n1. Step\n</think>\n\nFinal: The answer is \\boxed{42} }."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_malformed_boxed_nested_extra_brace(self):
        r"""\boxed{\frac{3}{4}}} — extra brace after nested boxed → invalid."""
        text = "<think>\nSolution:\n1. Step\n</think>\n\nFinal: The answer is \\boxed{\\frac{3}{4}}}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_malformed_boxed_missing_closing_brace(self):
        r"""\boxed{{42}. — unbalanced braces → invalid."""
        text = "<think>\nSolution:\n1. Step\n</think>\n\nFinal: The answer is \\boxed{{42}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_empty_boxed_content(self):
        r"""\boxed{} — empty content → invalid."""
        text = "<think>\nSolution:\n1. Step\n</think>\n\nFinal: The answer is \\boxed{}."
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_multiple_boxed_in_final(self):
        r"""Two \\boxed in final segment → invalid."""
        text = (
            "<think>\nSolution:\n1. Step\n</think>\n\n"
            r"Final: The answer is \boxed{1} and \boxed{2}."
        )
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_residual_after_boxed(self):
        r"""Content after the boxed closing brace → invalid."""
        text = (
            "<think>\nSolution:\n1. Step\n</think>\n\n"
            r"Final: The answer is \boxed{42} extra text."
        )
        ok, reasons = check_style_adherence(text)
        self.assertFalse(ok)
        self.assertIn("invalid_final_template", reasons)

    def test_blank_lines_between_think_and_final(self):
        """Blank lines between </think> and Final: are allowed."""
        text = (
            "<think>\n"
            "Solution:\n"
            "1. Step one\n"
            "</think>\n\n\n\n"
            "Final: The answer is \\boxed{42}."
        )
        ok, reasons = check_style_adherence(text)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])


class TestEvaluateSample(unittest.TestCase):
    """Test per-sample evaluation combining correctness and style."""

    def _compliant_correct(self) -> str:
        return (
            "<think>\n"
            "Solution:\n"
            "1. Compute 40 + 2 = 42\n"
            "2. Verify: 42 is correct\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )

    def test_compliant_and_correct(self):
        r = evaluate_sample(self._compliant_correct(), "42")
        self.assertTrue(r["answer_extractable"])
        self.assertTrue(r["answer_correct"])
        self.assertTrue(r["style_adherent"])
        self.assertEqual(r["predicted_answer"], "42")
        self.assertEqual(r["step_count"], 2)
        self.assertGreater(r["char_count"], 0)
        self.assertEqual(r["style_failures"], [])

    def test_correct_but_gsm8k_format(self):
        """GSM8K gold format: has boxed answer but no DPO v2 style template."""
        text = "<think>\nLet me think.\n40 + 2 = 42\n</think>\n\n\\boxed{42}"
        r = evaluate_sample(text, "42")
        self.assertTrue(r["answer_correct"])
        self.assertFalse(r["style_adherent"])
        self.assertIn("missing_solution_prefix", r["style_failures"])
        self.assertIn("invalid_final_template", r["style_failures"])

    def test_wrong_answer_correct_flag_false(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. Compute 40 + 2 = 44\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{44}."
        )
        r = evaluate_sample(text, "42")
        self.assertFalse(r["answer_correct"])
        self.assertTrue(r["style_adherent"])

    def test_step_count_and_char_count(self):
        text = (
            "<think>\n"
            "Solution:\n"
            "1. First step\n"
            "2. Second step\n"
            "3. Third step\n"
            "</think>\n\n"
            "Final: The answer is \\boxed{42}."
        )
        r = evaluate_sample(text, "42")
        self.assertEqual(r["step_count"], 3)
        self.assertGreater(r["char_count"], 50)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests using temp directories and patched sys.argv."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _run_main(self, predictions_path, references_path, output_dir) -> tuple[int, str]:
        buf = io.StringIO()
        with patch(
            "sys.argv",
            [
                "eval_style_dpo.py",
                "--predictions", str(predictions_path),
                "--references", str(references_path),
                "--output-dir", str(output_dir),
            ],
        ), redirect_stdout(buf):
            result = main()
        return result, buf.getvalue()

    def _compliant_prediction(self, answer: str = "42") -> str:
        return (
            "<think>\n"
            "Solution:\n"
            "1. Compute 40 + 2 = 42\n"
            "2. Verify: 42 is correct\n"
            "</think>\n\n"
            f"Final: The answer is \\boxed{{{answer}}}."
        )

    def test_full_cli_run(self):
        """End-to-end: write inputs, run main, verify summary and per_sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            predictions = [
                {"problem_id": "p1", "prediction": self._compliant_prediction("42")},
                {"problem_id": "p2", "prediction": "<think>\nThinking...\n</think>\n\n\\boxed{42}"},
                {"problem_id": "p3", "prediction": self._compliant_prediction("99")},
            ]
            references = [
                {"problem_id": "p1", "answer": "42"},
                {"problem_id": "p2", "answer": "42"},
                {"problem_id": "p3", "answer": "42"},
            ]
            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 0)

            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(summary["total_samples"], 3)
            self.assertEqual(summary["answer_correct_count"], 2)
            self.assertEqual(summary["style_adherent_count"], 2)
            self.assertEqual(summary["correct_and_adherent_count"], 1)
            self.assertIn("token_estimate_method", summary)
            self.assertIn("avg_token_estimate", summary)

            with (out_dir / "per_sample.jsonl").open() as f:
                samples = [json.loads(line) for line in f]
            self.assertEqual(len(samples), 3)

            p1 = next(s for s in samples if s["problem_id"] == "p1")
            self.assertTrue(p1["answer_correct"])
            self.assertTrue(p1["style_adherent"])

            p2 = next(s for s in samples if s["problem_id"] == "p2")
            self.assertTrue(p2["answer_correct"])
            self.assertFalse(p2["style_adherent"])

            p3 = next(s for s in samples if s["problem_id"] == "p3")
            self.assertFalse(p3["answer_correct"])
            self.assertTrue(p3["style_adherent"])

    def test_output_dir_already_exists_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [{"problem_id": "p1", "prediction": self._compliant_prediction()}])
            self._write_jsonl(ref_path, [{"problem_id": "p1", "answer": "42"}])
            out_dir.mkdir()

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("already exists", stdout)
            self.assertEqual(list(out_dir.iterdir()), [])

    def test_references_from_metadata_problem_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            predictions = [
                {"problem_id": "gsm8k_train_d5_0001", "prediction": self._compliant_prediction()},
            ]
            references = [
                {"id": "dpo_style_abc123", "answer": "42", "metadata": {"problem_id": "gsm8k_train_d5_0001"}},
            ]
            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, _ = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 0)

            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(summary["answer_correct_count"], 1)

    def test_empty_both_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [])
            self._write_jsonl(ref_path, [])

            result, _ = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 0)

            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(summary["total_samples"], 0)

    def test_empty_predictions_nonempty_references_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [])
            self._write_jsonl(ref_path, [{"problem_id": "p1", "answer": "42"}])

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("empty", stdout)
            self.assertFalse(out_dir.exists())

    def test_duplicate_prediction_problem_id_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            predictions = [
                {"problem_id": "p1", "prediction": self._compliant_prediction()},
                {"problem_id": "p1", "prediction": self._compliant_prediction()},
            ]
            references = [{"problem_id": "p1", "answer": "42"}]
            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("duplicate prediction", stdout)
            self.assertFalse(out_dir.exists())

    def test_duplicate_reference_problem_id_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            predictions = [{"problem_id": "p1", "prediction": self._compliant_prediction()}]
            references = [
                {"problem_id": "p1", "answer": "42"},
                {"problem_id": "p1", "answer": "99"},
            ]
            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("duplicate reference", stdout)
            self.assertFalse(out_dir.exists())

    def test_missing_reference_for_prediction_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            predictions = [
                {"problem_id": "p1", "prediction": self._compliant_prediction()},
                {"problem_id": "p2", "prediction": self._compliant_prediction()},
            ]
            references = [{"problem_id": "p1", "answer": "42"}]
            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("no matching reference", stdout)
            self.assertIn("p2", stdout)
            self.assertFalse(out_dir.exists())

    def test_extra_reference_no_prediction_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            predictions = [{"problem_id": "p1", "prediction": self._compliant_prediction()}]
            references = [
                {"problem_id": "p1", "answer": "42"},
                {"problem_id": "p_extra", "answer": "99"},
            ]
            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("no matching prediction", stdout)
            self.assertIn("p_extra", stdout)
            self.assertFalse(out_dir.exists())

    def test_null_prediction_problem_id_fails(self):
        """problem_id: null in predictions → fail before output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [{"problem_id": None, "prediction": "\\boxed{1}"}])
            self._write_jsonl(ref_path, [{"problem_id": "p1", "answer": "1"}])

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("empty problem_id", stdout)
            self.assertFalse(out_dir.exists())

    def test_whitespace_prediction_problem_id_fails(self):
        """problem_id: '   ' in predictions → fail before output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [{"problem_id": "   ", "prediction": "\\boxed{1}"}])
            self._write_jsonl(ref_path, [{"problem_id": "p1", "answer": "1"}])

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("empty problem_id", stdout)
            self.assertFalse(out_dir.exists())

    def test_null_reference_problem_id_fails(self):
        """problem_id: null with no metadata fallback → fail before output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [{"problem_id": "p1", "prediction": self._compliant_prediction()}])
            self._write_jsonl(ref_path, [{"problem_id": None, "answer": "42"}])

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("empty problem_id", stdout)
            self.assertFalse(out_dir.exists())

    def test_whitespace_reference_problem_id_fails(self):
        """problem_id: '   ' with no metadata fallback → fail before output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            self._write_jsonl(pred_path, [{"problem_id": "p1", "prediction": self._compliant_prediction()}])
            self._write_jsonl(ref_path, [{"problem_id": "   ", "answer": "42"}])

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 1)
            self.assertIn("empty problem_id", stdout)
            self.assertFalse(out_dir.exists())

    def test_stress_set_compatibility(self):
        """dpo_v2_style_stress_50.jsonl chosen → predictions: 50/50 correct, 50/50 adherent."""
        stress_path = Path("data/math/splits/dpo_v2_style_stress_50.jsonl")
        if not stress_path.exists():
            self.skipTest("stress split not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pred_path = tmpdir / "predictions.jsonl"
            ref_path = tmpdir / "references.jsonl"
            out_dir = tmpdir / "results"

            with stress_path.open() as f:
                records = [json.loads(line) for line in f]

            predictions = []
            references = []
            for rec in records:
                pid = rec["metadata"]["problem_id"]
                predictions.append({"problem_id": pid, "prediction": rec["chosen"]})
                references.append({"problem_id": pid, "answer": rec["answer"]})

            self._write_jsonl(pred_path, predictions)
            self._write_jsonl(ref_path, references)

            result, stdout = self._run_main(pred_path, ref_path, out_dir)
            self.assertEqual(result, 0, f"main() failed: {stdout}")

            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(summary["total_samples"], 50)
            self.assertEqual(summary["answer_correct_count"], 50)
            self.assertEqual(summary["style_adherent_count"], 50)


if __name__ == "__main__":
    unittest.main()
