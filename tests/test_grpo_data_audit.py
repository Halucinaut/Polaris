"""Tests for M2.5 GRPO data source audit script."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.data.audit_grpo_candidates import (
    _extract_problem_id,
    _extract_problem_text,
    _extract_answer,
    _check_grpo_compatibility,
    canonical_question_hash,
    _text_hash,
    audit_file,
    load_canonical_hashes,
    load_ids_and_hashes,
    compute_overlap,
    run_audit,
)


class TestCanonicalQuestionHash(unittest.TestCase):
    """Test canonical question hashing."""

    def test_deterministic(self):
        self.assertEqual(canonical_question_hash("Hello World"), canonical_question_hash("Hello World"))

    def test_normalizes_whitespace(self):
        self.assertEqual(canonical_question_hash("hello  world"), canonical_question_hash(" hello world "))

    def test_case_insensitive(self):
        self.assertEqual(canonical_question_hash("Hello"), canonical_question_hash("hello"))

    def test_different_texts_differ(self):
        self.assertNotEqual(canonical_question_hash("hello"), canonical_question_hash("world"))

    def test_alias_is_same_function(self):
        self.assertIs(_text_hash, canonical_question_hash)


class TestExtractProblemId(unittest.TestCase):
    """Test problem_id extraction from various schema formats."""

    def test_top_level_problem_id(self):
        r = {"problem_id": "gsm8k_train_0001", "problem": "2+2=?"}
        self.assertEqual(_extract_problem_id(r), "gsm8k_train_0001")

    def test_metadata_problem_id(self):
        r = {"messages": [], "metadata": {"problem_id": "gsm8k_train_d5_0001"}}
        self.assertEqual(_extract_problem_id(r), "gsm8k_train_d5_0001")

    def test_phash_and_index(self):
        r = {"phash": "abc123", "problem_index": 42, "problem": "x=1"}
        self.assertEqual(_extract_problem_id(r), "phash:abc123:idx:42")

    def test_no_id(self):
        r = {"problem": "x=1", "answer": "1"}
        self.assertIsNone(_extract_problem_id(r))

    def test_metadata_takes_precedence_over_top_level(self):
        r = {"problem_id": "top", "metadata": {"problem_id": "meta"}}
        self.assertEqual(_extract_problem_id(r), "top")


class TestExtractProblemText(unittest.TestCase):
    """Test problem text extraction."""

    def test_messages_format(self):
        r = {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]}
        self.assertEqual(_extract_problem_text(r), "What is 2+2?")

    def test_raw_problem_field(self):
        r = {"problem": "What is 2+2?"}
        self.assertEqual(_extract_problem_text(r), "What is 2+2?")

    def test_raw_question_field(self):
        r = {"question": "What is 2+2?"}
        self.assertEqual(_extract_problem_text(r), "What is 2+2?")

    def test_messages_takes_precedence(self):
        r = {
            "messages": [{"role": "user", "content": "from_messages"}],
            "problem": "from_problem",
        }
        self.assertEqual(_extract_problem_text(r), "from_messages")

    def test_no_text(self):
        r = {"answer": "4"}
        self.assertIsNone(_extract_problem_text(r))


class TestExtractAnswer(unittest.TestCase):
    """Test answer extraction."""

    def test_metadata_answer(self):
        r = {"metadata": {"answer": "42"}}
        self.assertEqual(_extract_answer(r), "42")

    def test_top_level_answer(self):
        r = {"answer": "42"}
        self.assertEqual(_extract_answer(r), "42")

    def test_metadata_takes_precedence(self):
        r = {"answer": "top", "metadata": {"answer": "meta"}}
        self.assertEqual(_extract_answer(r), "meta")

    def test_no_answer(self):
        r = {"problem": "x=1"}
        self.assertIsNone(_extract_answer(r))


class TestGrpoCompatibility(unittest.TestCase):
    """Test GRPO input schema compatibility check (mirrors train_grpo.py)."""

    def test_valid_record(self):
        r = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            "metadata": {"answer": "42"},
        }
        issues = _check_grpo_compatibility(r)
        self.assertEqual(issues, [])

    def test_missing_messages(self):
        r = {"metadata": {"answer": "42"}}
        issues = _check_grpo_compatibility(r)
        self.assertIn("missing or invalid 'messages' field", issues)

    def test_messages_not_list(self):
        r = {"messages": "not_a_list", "metadata": {"answer": "42"}}
        issues = _check_grpo_compatibility(r)
        self.assertIn("missing or invalid 'messages' field", issues)

    def test_messages_too_short(self):
        r = {"messages": [{"role": "user", "content": "q"}], "metadata": {"answer": "42"}}
        issues = _check_grpo_compatibility(r)
        self.assertIn("missing or invalid 'messages' field", issues)

    def test_missing_metadata_answer(self):
        r = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            "metadata": {},
        }
        issues = _check_grpo_compatibility(r)
        self.assertIn("missing metadata.answer", issues)

    def test_top_level_answer_not_grpo_compatible(self):
        """DPO-style files with top-level answer should fail GRPO check."""
        r = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            "answer": "42",
        }
        issues = _check_grpo_compatibility(r)
        self.assertIn("missing metadata.answer", issues)

    def test_matches_trainer_validate_data_schema(self):
        """Audit schema check must match train_grpo.py validate_data_schema() logic."""
        # Valid: messages is list, metadata.answer exists
        valid = {
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "metadata": {"answer": "42"},
        }
        self.assertEqual(_check_grpo_compatibility(valid), [])

        # Invalid: no messages
        self.assertTrue(len(_check_grpo_compatibility({"metadata": {"answer": "1"}})) > 0)

        # Invalid: messages not list
        self.assertTrue(len(_check_grpo_compatibility({"messages": "x", "metadata": {"answer": "1"}})) > 0)

        # Invalid: no metadata.answer (top-level answer doesn't count)
        self.assertTrue(len(_check_grpo_compatibility({
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "answer": "42",
        })) > 0)


class TestComputeOverlap(unittest.TestCase):
    """Test overlap computation with canonical deduplication."""

    def test_no_overlap(self):
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes={"ha", "hb"},
            candidate_ids={"a", "b"},
            excluded_hashes={"hc", "hd"},
            excluded_ids={"c", "d"},
        )
        self.assertEqual(canon, 0)
        self.assertEqual(id_cnt, 0)
        self.assertEqual(hash_cnt, 0)

    def test_hash_overlap_only(self):
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes={"ha", "hb"},
            candidate_ids=set(),
            excluded_hashes={"hb", "hc"},
            excluded_ids=set(),
        )
        self.assertEqual(canon, 1)
        self.assertEqual(hash_cnt, 1)
        self.assertEqual(id_cnt, 0)

    def test_id_overlap_only(self):
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes=set(),
            candidate_ids={"a", "b"},
            excluded_hashes=set(),
            excluded_ids={"b", "c"},
        )
        self.assertEqual(canon, 0)  # no hash match = no canonical match
        self.assertEqual(id_cnt, 1)
        self.assertEqual(hash_cnt, 0)

    def test_same_question_id_and_hash_both_match_counted_once(self):
        """When a question matches by both ID and hash, canonical_overlap_count = 1."""
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes={"ha", "hb"},
            candidate_ids={"a", "b"},
            excluded_hashes={"hb"},
            excluded_ids={"b"},
        )
        # canonical_overlap_count is based on hash matches = 1
        self.assertEqual(canon, 1)
        # id_match_count and text_hash_match_count are diagnostic breakdowns
        self.assertEqual(id_cnt, 1)
        self.assertEqual(hash_cnt, 1)

    def test_multiple_questions(self):
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes={"h1", "h2", "h3", "h4"},
            candidate_ids={"id1", "id2", "id3", "id4"},
            excluded_hashes={"h2", "h4", "h5"},
            excluded_ids={"id2", "id4", "id5"},
        )
        self.assertEqual(canon, 2)
        self.assertEqual(id_cnt, 2)
        self.assertEqual(hash_cnt, 2)


class TestAuditFile(unittest.TestCase):
    """Test file-level audit on real project files."""

    def test_sft_v1_is_grpo_compatible(self):
        p = Path("data/math/splits/sft_v1.jsonl")
        if not p.exists():
            self.skipTest("sft_v1.jsonl not found")
        fa = audit_file(p)
        self.assertTrue(fa.grpo_compatible)
        self.assertGreater(fa.record_count, 0)
        self.assertTrue(fa.has_messages)
        self.assertTrue(fa.has_metadata_answer)
        self.assertEqual(fa.grpo_schema.valid_count, fa.record_count)
        self.assertEqual(fa.grpo_schema.invalid_count, 0)

    def test_dpo_v1_is_not_grpo_compatible(self):
        p = Path("data/math/splits/dpo_v1.jsonl")
        if not p.exists():
            self.skipTest("dpo_v1.jsonl not found")
        fa = audit_file(p)
        self.assertFalse(fa.grpo_compatible)
        self.assertGreater(fa.record_count, 0)
        # All DPO v1 records have messages but lack metadata.answer
        self.assertEqual(fa.grpo_schema.invalid_count, fa.record_count)
        self.assertIn("missing metadata.answer", fa.grpo_schema.invalid_reasons)

    def test_train_converted_d5_500_not_grpo_compatible(self):
        """Converted GSM8K files lack messages/metadata.answer."""
        p = Path("data/math/gsm8k/split/train_converted_d5_500.jsonl")
        if not p.exists():
            self.skipTest("train_converted_d5_500.jsonl not found")
        fa = audit_file(p)
        self.assertFalse(fa.grpo_compatible)
        self.assertTrue(fa.has_problem_id)
        self.assertTrue(fa.has_answer)
        self.assertFalse(fa.has_messages)
        # All records should be invalid
        self.assertEqual(fa.grpo_schema.invalid_count, fa.record_count)

    def test_sft_d5_500_is_grpo_compatible(self):
        p = Path("data/math/splits/sft_d5_500.jsonl")
        if not p.exists():
            self.skipTest("sft_d5_500.jsonl not found")
        fa = audit_file(p)
        self.assertTrue(fa.grpo_compatible)
        self.assertEqual(fa.record_count, 500)
        self.assertEqual(fa.grpo_schema.valid_count, 500)

    def test_nonexistent_file_returns_error(self):
        fa = audit_file(Path("/nonexistent/file.jsonl"))
        self.assertEqual(fa.record_count, 0)
        self.assertTrue(any("LOAD ERROR" in n for n in fa.notes))

    def test_subsequent_invalid_records_detected(self):
        """If later records have bad schema, audit_file reports invalid_count > 0."""
        import tempfile, json, os
        # Create a temp file: first record valid, second invalid
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                "metadata": {"answer": "1"},
            }) + "\n")
            f.write(json.dumps({
                "messages": [{"role": "user", "content": "q only"}],
                "metadata": {},
            }) + "\n")
            tmppath = f.name
        try:
            fa = audit_file(Path(tmppath))
            self.assertFalse(fa.grpo_compatible)
            self.assertEqual(fa.grpo_schema.valid_count, 1)
            self.assertEqual(fa.grpo_schema.invalid_count, 1)
        finally:
            os.unlink(tmppath)

    def test_raw_gsm8k_is_convertible(self):
        """Raw GSM8K train is not GRPO-compatible but is convertible."""
        p = Path("data/math/gsm8k/train.jsonl")
        if not p.exists():
            self.skipTest("train.jsonl not found")
        fa = audit_file(p)
        self.assertFalse(fa.grpo_compatible)
        self.assertFalse(fa.has_messages)
        self.assertTrue(fa.has_answer)  # raw answer field exists
        # All records invalid (no messages)
        self.assertEqual(fa.grpo_schema.invalid_count, fa.record_count)


class TestLoadIdsAndHashes(unittest.TestCase):
    """Test ID and hash loading from real files."""

    def test_sft_d5_500_loads500_ids(self):
        p = Path("data/math/splits/sft_d5_500.jsonl")
        if not p.exists():
            self.skipTest("sft_d5_500.jsonl not found")
        ids, hashes = load_ids_and_hashes(p)
        self.assertEqual(len(ids), 500)
        self.assertEqual(len(hashes), 500)
        self.assertTrue(all(i.startswith("gsm8k_train_d5_") for i in ids))

    def test_dpo_v1_loads329_ids(self):
        p = Path("data/math/splits/dpo_v1.jsonl")
        if not p.exists():
            self.skipTest("dpo_v1.jsonl not found")
        ids, hashes = load_ids_and_hashes(p)
        self.assertEqual(len(ids), 329)

    def test_raw_gsm8k_loads_hashes(self):
        p = Path("data/math/gsm8k/train.jsonl")
        if not p.exists():
            self.skipTest("train.jsonl not found")
        ids, hashes = load_ids_and_hashes(p)
        # Raw GSM8K has no problem_id, so ids should be empty
        self.assertEqual(len(ids), 0)
        self.assertEqual(len(hashes), 7473)


class TestRunAudit(unittest.TestCase):
    """Integration test: run full audit on project."""

    def test_run_audit_returns_report(self):
        root = Path(__file__).parent.parent
        report = run_audit(root)
        self.assertGreater(len(report.files), 0)
        self.assertIsInstance(report.summary, dict)
        self.assertIn("total_files_audited", report.summary)
        self.assertIn("grpo_direct_compatible_files", report.summary)
        self.assertIn("math_level_3_5_file_exists", report.summary)

    def test_grpo_compatible_files_exist(self):
        root = Path(__file__).parent.parent
        report = run_audit(root)
        grpo_ready = [f for f in report.files if f.grpo_compatible and f.record_count > 0]
        self.assertGreater(len(grpo_ready), 0, "Should have at least one GRPO-compatible file")

    def test_sft_d5_500_is_grpo_compatible_in_full_audit(self):
        root = Path(__file__).parent.parent
        report = run_audit(root)
        sft_entry = next(
            (f for f in report.files if "sft_d5_500" in f.path),
            None,
        )
        self.assertIsNotNone(sft_entry)
        self.assertTrue(sft_entry.grpo_compatible)
        self.assertEqual(sft_entry.record_count, 500)

    def test_math_level_3_5_does_not_exist(self):
        root = Path(__file__).parent.parent
        report = run_audit(root)
        self.assertFalse(report.summary["math_level_3_5_file_exists"])

    def test_raw_gsm8k_in_candidate_audits(self):
        """Raw GSM8K train must appear in candidate audits."""
        root = Path(__file__).parent.parent
        report = run_audit(root)
        raw_audit = next(
            (c for c in report.candidate_audits if "train.jsonl" in c.candidate_name),
            None,
        )
        self.assertIsNotNone(raw_audit, "Raw GSM8K train must be in candidate audits")
        self.assertEqual(raw_audit.total_records, 7473)

    def test_raw_gsm8k_d5_only_exclusion_gives6973(self):
        """Excluding only d5_500 from raw GSM8K train gives6973."""
        root = Path(__file__).parent.parent
        report = run_audit(root)
        raw_audit = next(
            c for c in report.candidate_audits
            if c.candidate_name == "data/math/gsm8k/train.jsonl"
        )
        # m1_sft_train = d5_500 = 500 questions
        d5_overlap = next(
            ov for ov in raw_audit.per_excluded_set
            if ov.excluded_name == "m1_sft_train"
        )
        self.assertEqual(d5_overlap.canonical_overlap_count, 500)
        self.assertEqual(raw_audit.total_records - d5_overlap.canonical_overlap_count, 6973)

    def test_raw_gsm8k_all_exclusions_gives6043(self):
        """Excluding ALL sets from raw GSM8K train gives6043 clean remaining."""
        root = Path(__file__).parent.parent
        report = run_audit(root)
        raw_audit = next(
            c for c in report.candidate_audits
            if c.candidate_name == "data/math/gsm8k/train.jsonl"
        )
        self.assertEqual(raw_audit.clean_remaining, 6043)

    def test_overlap_no_double_counting(self):
        """Same question matching by both ID and hash counts as1 in canonical_overlap."""
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes={"h1"},
            candidate_ids={"id1"},
            excluded_hashes={"h1"},
            excluded_ids={"id1"},
        )
        self.assertEqual(canon, 1, "Canonical overlap must count each question once")


class TestTrainerSchemaConsistency(unittest.TestCase):
    """Verify audit schema check matches train_grpo.py validate_data_schema()."""

    def test_valid_record_passes_both(self):
        """A record passing audit check should also pass trainer validation."""
        from scripts.train_grpo import validate_data_schema
        record = {
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "metadata": {"answer": "42"},
        }
        audit_issues = _check_grpo_compatibility(record)
        trainer_errors = validate_data_schema([record])
        self.assertEqual(audit_issues, [])
        self.assertEqual(trainer_errors, [])

    def test_missing_messages_fails_both(self):
        """Missing messages should fail both audit and trainer validation."""
        from scripts.train_grpo import validate_data_schema
        record = {"metadata": {"answer": "42"}}
        audit_issues = _check_grpo_compatibility(record)
        trainer_errors = validate_data_schema([record])
        self.assertGreater(len(audit_issues), 0)
        self.assertGreater(len(trainer_errors), 0)

    def test_top_level_answer_fails_both(self):
        """Top-level answer (no metadata.answer) should fail both checks."""
        from scripts.train_grpo import validate_data_schema
        record = {
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "answer": "42",
        }
        audit_issues = _check_grpo_compatibility(record)
        trainer_errors = validate_data_schema([record])
        self.assertGreater(len(audit_issues), 0)
        self.assertGreater(len(trainer_errors), 0)

    def test_sft_v1_passes_trainer_validation(self):
        """Real sft_v1.jsonl should pass trainer validate_data_schema()."""
        from scripts.train_grpo import validate_data_schema
        from polaris.json_records import load_json_record_stream
        p = Path("data/math/splits/sft_v1.jsonl")
        if not p.exists():
            self.skipTest("sft_v1.jsonl not found")
        records = load_json_record_stream(p)
        errors = validate_data_schema(records)
        self.assertEqual(errors, [], f"sft_v1 should pass trainer validation, got: {errors}")


if __name__ == "__main__":
    unittest.main()
