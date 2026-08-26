"""Tests for prepare_sft_v2_style_control_data.py"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_sft_v2_style_control_data import (
    _EXPECTED_INPUT_COUNT,
    convert_record,
    load_input,
    main,
)


def _make_dpo_record(
    problem_id: str = "p1",
    content: str = "What is 1+1?",
    chosen: str = "<think>\nSolution:\n1. Add.\n</think>\n\nFinal: The answer is \\boxed{2}.",
    answer: str = "2",
    pair_id: str = "dpo_style_abc",
) -> dict:
    return {
        "id": pair_id,
        "messages": [
            {"role": "system", "content": "You are a math assistant."},
            {"role": "user", "content": content},
        ],
        "chosen": chosen,
        "rejected": "<think>\n2\n</think>\n\\boxed{2}",
        "answer": answer,
        "metadata": {
            "problem_id": problem_id,
            "source_dataset": "gsm8k",
        },
    }


class TestConvertRecord(unittest.TestCase):
    def test_basic(self):
        r = _make_dpo_record()
        c = convert_record(r)
        self.assertEqual(c["metadata"]["problem_id"], "p1")
        self.assertEqual(c["metadata"]["source"], "dpo_v2_style_chosen")
        self.assertEqual(c["metadata"]["source_pair_id"], "dpo_style_abc")
        self.assertEqual(c["metadata"]["answer"], "2")
        self.assertIn("Solution:", c["target"])

    def test_target_equals_chosen_bytes(self):
        chosen_text = "<think>\nSolution:\n1. Step.\n</think>\n\nFinal: \\boxed{42}."
        r = _make_dpo_record(chosen=chosen_text)
        c = convert_record(r)
        self.assertEqual(c["target"], chosen_text)

    def test_messages_preserved(self):
        r = _make_dpo_record(content="Test problem?")
        c = convert_record(r)
        roles = [m["role"] for m in c["messages"]]
        self.assertIn("user", roles)
        self.assertIn("system", roles)

    def test_empty_chosen_raises(self):
        r = _make_dpo_record(chosen="")
        with self.assertRaises(ValueError):
            convert_record(r)

    def test_empty_problem_id_raises(self):
        r = _make_dpo_record(problem_id="  ")
        with self.assertRaises(ValueError):
            convert_record(r)

    def test_empty_content_raises(self):
        r = _make_dpo_record(content="  ")
        with self.assertRaises(ValueError):
            convert_record(r)

    def test_problem_id_from_metadata(self):
        r = _make_dpo_record(problem_id="meta_pid")
        c = convert_record(r)
        self.assertEqual(c["metadata"]["problem_id"], "meta_pid")

    def test_missing_pair_id_ok(self):
        r = _make_dpo_record()
        del r["id"]
        c = convert_record(r)
        self.assertEqual(c["metadata"]["source_pair_id"], "")


class TestLoadInput(unittest.TestCase):
    def test_loads_records(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_dpo_record("r1")) + "\n")
            f.write(json.dumps(_make_dpo_record("r2")) + "\n")
            path = f.name
        try:
            records = load_input(path)
            self.assertEqual(len(records), 2)
        finally:
            os.unlink(path)

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_dpo_record("r1")) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_dpo_record("r2")) + "\n")
            path = f.name
        try:
            records = load_input(path)
            self.assertEqual(len(records), 2)
        finally:
            os.unlink(path)


class TestRealData(unittest.TestCase):
    INPUT_PATH = "data/math/splits/dpo_v2_style_train_449.jsonl"

    def _load(self):
        if not os.path.exists(self.INPUT_PATH):
            self.skipTest("Real data not present")
        return load_input(self.INPUT_PATH)

    def test_count_is_449(self):
        self.assertEqual(len(self._load()), _EXPECTED_INPUT_COUNT)

    def test_all_convert(self):
        for r in self._load():
            c = convert_record(r)
            self.assertIn("messages", c)
            self.assertIn("target", c)
            self.assertIn("metadata", c)
            self.assertEqual(c["metadata"]["source"], "dpo_v2_style_chosen")


class TestCli(unittest.TestCase):
    def test_output_exists_raises(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "in.jsonl")
            with open(input_path, "w") as f:
                f.write(json.dumps(_make_dpo_record()) + "\n")
            output_path = os.path.join(td, "out.jsonl")
            with open(output_path, "w") as f:
                f.write("existing")
            sys.argv = [
                "prepare_sft_v2_style_control_data.py",
                "--input", input_path,
                "--output", output_path,
                "--report", os.path.join(td, "report.json"),
            ]
            with self.assertRaises(FileExistsError):
                main()

    def test_full_run(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "in.jsonl")
            with open(input_path, "w") as f:
                for i in range(3):
                    f.write(json.dumps(_make_dpo_record(problem_id=f"p{i}")) + "\n")
            output_path = os.path.join(td, "out.jsonl")
            report_path = os.path.join(td, "report.json")
            sys.argv = [
                "prepare_sft_v2_style_control_data.py",
                "--input", input_path,
                "--output", output_path,
                "--report", report_path,
                "--expected-count", "3",
            ]
            main()
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.exists(report_path))
            with open(output_path) as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
