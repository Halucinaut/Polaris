"""Tests for prepare_dpo_v2_style_eval_data.py — stress-50 eval data conversion."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_dpo_v2_style_eval_data import (
    convert_record,
    extract_user_content,
    main,
    validate_inputs,
)


class TestExtractUserContent(unittest.TestCase):
    def test_single_user_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        self.assertEqual(extract_user_content(messages), "What is 2+2?")

    def test_no_user_message_raises(self):
        with self.assertRaises(ValueError):
            extract_user_content([{"role": "system", "content": "hi"}])

    def test_multiple_user_messages_raises(self):
        with self.assertRaises(ValueError):
            extract_user_content([
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ])

    def test_null_content_raises(self):
        with self.assertRaises(ValueError) as ctx:
            extract_user_content([{"role": "user", "content": None}])
        self.assertIn("must be str", str(ctx.exception))

    def test_whitespace_content_raises(self):
        with self.assertRaises(ValueError) as ctx:
            extract_user_content([{"role": "user", "content": "   "}])
        self.assertIn("empty or whitespace", str(ctx.exception))

    def test_numeric_content_raises(self):
        with self.assertRaises(ValueError) as ctx:
            extract_user_content([{"role": "user", "content": 42}])
        self.assertIn("must be str", str(ctx.exception))


class TestValidateInputs(unittest.TestCase):
    def _make_record(self, problem_id="gsm8k_train_d5_0001", answer="42",
                     content="What is 40+2?") -> dict:
        return {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": content},
            ],
            "answer": answer,
            "metadata": {"problem_id": problem_id},
        }

    def test_valid_50_records(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(50)]
        validate_inputs(records)

    def test_wrong_count_raises(self):
        with self.assertRaises(ValueError) as ctx:
            validate_inputs([self._make_record()])
        self.assertIn("expected 50", str(ctx.exception))

    def test_empty_problem_id_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record(problem_id=""))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("empty or whitespace", str(ctx.exception))

    def test_null_problem_id_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record(problem_id=None))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("must be str", str(ctx.exception))

    def test_whitespace_problem_id_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record(problem_id="   \t  "))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("empty or whitespace", str(ctx.exception))

    def test_numeric_problem_id_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record(problem_id=12345))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("must be str", str(ctx.exception))

    def test_duplicate_problem_id_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record(problem_id="gsm8k_train_d5_0000"))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("duplicate problem_id", str(ctx.exception))

    def test_null_content_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record("gsm8k_train_d5_0049", content=None))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("must be str", str(ctx.exception))

    def test_whitespace_content_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record("gsm8k_train_d5_0049", content="   "))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("empty or whitespace", str(ctx.exception))

    def test_empty_content_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record("gsm8k_train_d5_0049", content=""))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("empty or whitespace", str(ctx.exception))

    def test_null_answer_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record("gsm8k_train_d5_0049", answer=None))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("must be str", str(ctx.exception))

    def test_whitespace_answer_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record("gsm8k_train_d5_0049", answer="   "))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("empty or whitespace", str(ctx.exception))

    def test_empty_answer_raises(self):
        records = [self._make_record(f"gsm8k_train_d5_{i:04d}") for i in range(49)]
        records.append(self._make_record("gsm8k_train_d5_0049", answer=""))
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(records)
        self.assertIn("empty or whitespace", str(ctx.exception))


class TestConvertRecord(unittest.TestCase):
    def test_converts_correctly(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 40+2?"},
            ],
            "answer": "42",
            "metadata": {"problem_id": "gsm8k_train_d5_0001"},
        }
        result = convert_record(record)
        self.assertEqual(result, {
            "problem_id": "gsm8k_train_d5_0001",
            "problem": "What is 40+2?",
            "answer": "42",
            "source": "dpo_v2_style_stress",
        })

    def test_preserves_original_text_not_stripped(self):
        """Output keeps original problem/answer text; only pid is stripped."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "  What is 40+2?  "},
            ],
            "answer": "  42  ",
            "metadata": {"problem_id": "  gsm8k_train_d5_0001  "},
        }
        result = convert_record(record)
        # problem_id is stripped
        self.assertEqual(result["problem_id"], "gsm8k_train_d5_0001")
        # problem and answer preserve original text
        self.assertEqual(result["problem"], "  What is 40+2?  ")
        self.assertEqual(result["answer"], "  42  ")


class TestCLIIntegration(unittest.TestCase):
    def _make_records(self, n: int = 50) -> list[dict]:
        return [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": f"What is {i}+1?"},
                ],
                "answer": str(i + 1),
                "metadata": {"problem_id": f"gsm8k_train_d5_{i:04d}"},
            }
            for i in range(n)
        ]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _run_main(self, input_path, output_path) -> tuple[int, str]:
        buf = io.StringIO()
        with patch(
            "sys.argv",
            ["prepare_dpo_v2_style_eval_data.py", "--input", str(input_path), "--output", str(output_path)],
        ), redirect_stdout(buf):
            result = main()
        return result, buf.getvalue()

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "stress_50.jsonl"
            output_path = tmpdir / "stress_eval_50.jsonl"

            self._write_jsonl(input_path, self._make_records(50))

            result, stdout = self._run_main(input_path, output_path)
            self.assertEqual(result, 0)
            self.assertTrue(output_path.exists())

            with output_path.open() as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 50)
            self.assertEqual(lines[0]["source"], "dpo_v2_style_stress")
            self.assertIn("problem_id", lines[0])
            self.assertIn("problem", lines[0])
            self.assertIn("answer", lines[0])

    def test_output_already_exists_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "stress_50.jsonl"
            output_path = tmpdir / "stress_eval_50.jsonl"

            self._write_jsonl(input_path, self._make_records(50))
            output_path.write_text('{"sentinel": true}\n')

            result, stdout = self._run_main(input_path, output_path)
            self.assertEqual(result, 1)
            self.assertIn("already exists", stdout)
            self.assertEqual(output_path.read_text(), '{"sentinel": true}\n')

    def test_invalid_input_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "bad_input.jsonl"
            output_path = tmpdir / "should_not_exist.jsonl"

            self._write_jsonl(input_path, self._make_records(3))

            result, stdout = self._run_main(input_path, output_path)
            self.assertEqual(result, 1)
            self.assertIn("expected 50", stdout)
            self.assertFalse(output_path.exists())

    def test_null_id_no_output(self):
        """Null problem_id must fail and leave no output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "bad_input.jsonl"
            output_path = tmpdir / "should_not_exist.jsonl"

            records = self._make_records(49)
            records.append({
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q"},
                ],
                "answer": "42",
                "metadata": {"problem_id": None},
            })
            self._write_jsonl(input_path, records)

            result, stdout = self._run_main(input_path, output_path)
            self.assertEqual(result, 1)
            self.assertIn("must be str", stdout)
            self.assertFalse(output_path.exists())

    def test_real_stress_file_produces_50_records(self):
        """The real stress-50 file must convert to exactly 50 eval records with matching IDs."""
        real_input = Path("data/math/splits/dpo_v2_style_stress_50.jsonl")
        if not real_input.exists():
            self.skipTest("real stress file not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / "stress_eval_50.jsonl"

            result, _ = self._run_main(real_input, output_path)
            self.assertEqual(result, 0)

            with output_path.open() as f:
                records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 50)

            # Verify IDs match original
            with real_input.open() as f:
                original_ids = [json.loads(line)["metadata"]["problem_id"] for line in f]
            converted_ids = [r["problem_id"] for r in records]
            self.assertEqual(original_ids, converted_ids)


if __name__ == "__main__":
    unittest.main()
