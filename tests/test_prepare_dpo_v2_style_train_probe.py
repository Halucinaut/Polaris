"""Tests for prepare_dpo_v2_style_train_probe.py"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_dpo_v2_style_train_probe import (
    _EXPECTED_INPUT_COUNT,
    _EXPECTED_TOTAL,
    _PER_BIN,
    _sha256_hex,
    convert_record,
    main,
    select_probe,
)


def _make_record(
    problem_id: str = "p1",
    content: str = "What is 1+1?",
    answer: str = "2",
    ratio: float = 1.0,
) -> dict:
    """Build a record matching real DPO v2 data format."""
    return {
        "id": f"dpo_style_{problem_id}",
        "messages": [
            {"role": "user", "content": content},
        ],
        "answer": answer,
        "metadata": {
            "problem_id": problem_id,
            "token_length_ratio": ratio,
            "source_dataset": "gsm8k",
        },
    }


def _make_records(count: int, start: int = 1, ratio: float = 1.0) -> list[dict]:
    return [_make_record(problem_id=f"p{i}", ratio=ratio) for i in range(start, start + count)]


class TestSha256Hex(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(_sha256_hex("abc"), _sha256_hex("abc"))

    def test_different_for_different_ids(self):
        self.assertNotEqual(_sha256_hex("abc"), _sha256_hex("def"))

    def test_64_char_hex(self):
        self.assertEqual(len(_sha256_hex("test")), 64)


class TestConvertRecord(unittest.TestCase):
    def test_basic(self):
        r = _make_record(ratio=0.8)
        c = convert_record(r)
        self.assertEqual(c["problem_id"], "p1")
        self.assertEqual(c["source"], "dpo_v2_style_train_probe")
        self.assertEqual(c["source_length_ratio"], 0.8)
        self.assertEqual(c["probe_bin"], "short")

    def test_short_bin(self):
        self.assertEqual(convert_record(_make_record(ratio=0.5))["probe_bin"], "short")

    def test_balanced_bin(self):
        self.assertEqual(convert_record(_make_record(ratio=1.0))["probe_bin"], "balanced")

    def test_long_bin(self):
        self.assertEqual(convert_record(_make_record(ratio=1.5))["probe_bin"], "long")

    def test_boundary_09_goes_to_balanced(self):
        self.assertEqual(convert_record(_make_record(ratio=0.9))["probe_bin"], "balanced")

    def test_boundary_12_goes_to_long(self):
        self.assertEqual(convert_record(_make_record(ratio=1.2))["probe_bin"], "long")

    def test_missing_ratio_raises(self):
        r = _make_record()
        del r["metadata"]["token_length_ratio"]
        with self.assertRaises(ValueError):
            convert_record(r)

    def test_empty_problem_id_raises(self):
        r = _make_record(problem_id="  ")
        with self.assertRaises(ValueError):
            convert_record(r)

    def test_problem_id_from_metadata(self):
        r = _make_record(problem_id="from_meta")
        c = convert_record(r)
        self.assertEqual(c["problem_id"], "from_meta")


class TestSelectProbe(unittest.TestCase):
    def _mixed_records(self):
        return (
            _make_records(149, start=1, ratio=0.5)
            + _make_records(150, start=150, ratio=1.0)
            + _make_records(150, start=300, ratio=1.5)
        )

    def test_exact_449_records(self):
        probe = select_probe(self._mixed_records())
        self.assertEqual(len(probe), _EXPECTED_TOTAL)

    def test_wrong_count_raises(self):
        with self.assertRaises(ValueError):
            select_probe(_make_records(100))

    def test_duplicate_ids_raises(self):
        with self.assertRaises(ValueError):
            select_probe([_make_record(problem_id="dup")] * 449)

    def test_each_bin_has_exactly_10(self):
        probe = select_probe(self._mixed_records())
        bins = {}
        for r in probe:
            bins.setdefault(r["probe_bin"], []).append(r)
        self.assertEqual(len(bins.get("short", [])), _PER_BIN)
        self.assertEqual(len(bins.get("balanced", [])), _PER_BIN)
        self.assertEqual(len(bins.get("long", [])), _PER_BIN)

    def test_reproducible(self):
        records = self._mixed_records()
        p1 = [r["problem_id"] for r in select_probe(records)]
        p2 = [r["problem_id"] for r in select_probe(records)]
        self.assertEqual(p1, p2)

    def test_bin_insufficient_raises(self):
        records = (
            _make_records(5, start=1, ratio=0.5)
            + _make_records(150, start=6, ratio=1.0)
            + _make_records(294, start=156, ratio=1.5)
        )
        with self.assertRaises(ValueError):
            select_probe(records)


class TestRealData(unittest.TestCase):
    TRAIN_PATH = "data/math/splits/dpo_v2_style_train_449.jsonl"

    def _load(self):
        if not os.path.exists(self.TRAIN_PATH):
            self.skipTest("Real train data not present")
        records = []
        with open(self.TRAIN_PATH, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def test_three_bins_have_at_least_10(self):
        records = self._load()
        bins = {}
        for r in records:
            c = convert_record(r)
            bins.setdefault(c["probe_bin"], []).append(c["problem_id"])
        self.assertGreaterEqual(len(bins.get("short", [])), _PER_BIN)
        self.assertGreaterEqual(len(bins.get("balanced", [])), _PER_BIN)
        self.assertGreaterEqual(len(bins.get("long", [])), _PER_BIN)

    def test_output_is_10_10_10(self):
        probe = select_probe(self._load())
        bins = {}
        for r in probe:
            bins.setdefault(r["probe_bin"], []).append(r)
        self.assertEqual(len(bins["short"]), _PER_BIN)
        self.assertEqual(len(bins["balanced"]), _PER_BIN)
        self.assertEqual(len(bins["long"]), _PER_BIN)


class TestCli(unittest.TestCase):
    def test_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as td:
            sys.argv = [
                "prepare_dpo_v2_style_train_probe.py",
                "--input", os.path.join(td, "missing.jsonl"),
                "--output", os.path.join(td, "out.jsonl"),
                "--report", os.path.join(td, "report.json"),
            ]
            with self.assertRaises(FileNotFoundError):
                main()

    def test_output_exists_raises(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "in.jsonl")
            records = (
                _make_records(149, start=1, ratio=0.5)
                + _make_records(150, start=150, ratio=1.0)
                + _make_records(150, start=300, ratio=1.5)
            )
            with open(input_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            output_path = os.path.join(td, "out.jsonl")
            with open(output_path, "w") as f:
                f.write("existing")
            sys.argv = [
                "prepare_dpo_v2_style_train_probe.py",
                "--input", input_path,
                "--output", output_path,
                "--report", os.path.join(td, "report.json"),
            ]
            with self.assertRaises(FileExistsError):
                main()

    def test_full_run(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "in.jsonl")
            records = (
                _make_records(149, start=1, ratio=0.5)
                + _make_records(150, start=150, ratio=1.0)
                + _make_records(150, start=300, ratio=1.5)
            )
            with open(input_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            output_path = os.path.join(td, "out.jsonl")
            report_path = os.path.join(td, "report.json")
            sys.argv = [
                "prepare_dpo_v2_style_train_probe.py",
                "--input", input_path,
                "--output", output_path,
                "--report", report_path,
            ]
            main()
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.exists(report_path))
            with open(output_path) as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), _EXPECTED_TOTAL)
            with open(report_path) as f:
                report = json.load(f)
            self.assertEqual(report["total"], _EXPECTED_TOTAL)


if __name__ == "__main__":
    unittest.main()
