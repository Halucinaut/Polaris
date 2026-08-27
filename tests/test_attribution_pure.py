"""Pure (no-MLX) unit tests for polaris.attribution.

These tests run under system Python with ``make test``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polaris.attribution import (
    assemble_solution_metrics,
    build_style_patterns,
    build_summary,
    classify_response_positions,
    compute_exact_margin,
    find_divergence_position,
    find_token_in_response,
    lookup_entry,
    lookup_logprob,
    sum_masked_logprob,
)


# ---------------------------------------------------------------------------
# Stub tokenizer
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Deterministic 1-char-per-token tokenizer for tests."""

    def encode(self, text, **_kw):
        return [ord(c) for c in text]

    def decode(self, ids, **_kw):
        return "".join(chr(i) for i in ids if 0 < i < 128)


# ---------------------------------------------------------------------------
# classify_response_positions
# ---------------------------------------------------------------------------

class TestClassifyResponsePositions(unittest.TestCase):

    def _call(self, full_ids, mask, patterns=None):
        tok = _StubTokenizer()
        if patterns is None:
            patterns = build_style_patterns(tok)
        return classify_response_positions(full_ids, mask, patterns)

    def test_solution_keyword(self):
        tok = _StubTokenizer()
        sol = tok.encode("Solution")
        prompt = [1, 2]
        full = prompt + sol + [99]
        mask = [0, 0, 1, 1, 1]
        result = self._call(full, mask)
        for i in range(len(sol)):
            self.assertEqual(result[2 + i], "solution_keyword")
        self.assertEqual(result[2 + len(sol)], "unclassified")

    def test_final_wrapper(self):
        tok = _StubTokenizer()
        final = tok.encode("Final: The answer is ")
        prompt = [1]
        prefix = tok.encode("Solution:\n1. X.\n")
        full = prompt + prefix + final + [88]
        mask = [0] + [1] * (len(prefix) + len(final) + 1)
        result = self._call(full, mask)
        base = 1 + len(prefix)
        for i in range(len(final)):
            self.assertEqual(result[base + i], "final_wrapper")

    def test_boxed_answer(self):
        tok = _StubTokenizer()
        boxed = tok.encode("\\boxed{")
        prompt = [1]
        prefix = tok.encode("Solution:\nFinal: The answer is ")
        full = prompt + prefix + boxed + [125, 46]
        mask = [0] + [1] * (len(prefix) + len(boxed) + 2)
        result = self._call(full, mask)
        base = 1 + len(prefix)
        for i in range(len(boxed)):
            self.assertEqual(result[base + i], "boxed_answer")

    def test_numbered_step_prefix(self):
        tok = _StubTokenizer()
        step = tok.encode("1. ")
        prompt = [1]
        sol = tok.encode("Solution:\n")
        full = prompt + sol + step + [50]
        mask = [0] + [1] * (len(sol) + len(step) + 1)
        result = self._call(full, mask)
        base = 1 + len(sol)
        for i in range(len(step)):
            self.assertEqual(result[base + i], "numbered_step_prefix")

    def test_unclassified_never_omitted(self):
        full = [1, 200, 201, 202]
        mask = [0, 1, 1, 1]
        result = self._call(full, mask)
        for i in range(1, 4):
            self.assertIn(i, result)
            self.assertEqual(result[i], "unclassified")

    def test_full_template(self):
        tok = _StubTokenizer()
        sol = tok.encode("Solution")
        step1 = tok.encode("1. ")
        step2 = tok.encode("2. ")
        final = tok.encode("Final: The answer is ")
        boxed = tok.encode("\\boxed{")
        rest = [125, 46]
        prompt = [1, 2, 3]
        chosen = sol + step1 + [50, 51] + step2 + [52, 53] + final + boxed + rest
        full = prompt + chosen
        mask = [0, 0, 0] + [1] * len(chosen)
        result = self._call(full, mask)

        r = 3  # prompt_len
        for i in range(len(sol)):
            self.assertEqual(result[r + i], "solution_keyword")
        off = r + len(sol)
        for i in range(len(step1)):
            self.assertEqual(result[off + i], "numbered_step_prefix")
        off += len(step1) + 2
        for i in range(len(step2)):
            self.assertEqual(result[off + i], "numbered_step_prefix")
        off += len(step2) + 2
        for i in range(len(final)):
            self.assertEqual(result[off + i], "final_wrapper")
        off += len(final)
        for i in range(len(boxed)):
            self.assertEqual(result[off + i], "boxed_answer")


# ---------------------------------------------------------------------------
# find_divergence_position / find_token_in_response
# ---------------------------------------------------------------------------

class TestFindHelpers(unittest.TestCase):

    def test_divergence_at_start(self):
        self.assertEqual(find_divergence_position([1, 2, 3], [9, 2, 3]), 0)

    def test_divergence_at_end(self):
        self.assertEqual(find_divergence_position([1, 2, 3], [1, 2, 9]), 2)

    def test_no_divergence(self):
        self.assertIsNone(find_divergence_position([1, 2], [1, 2]))

    def test_different_lengths(self):
        self.assertEqual(find_divergence_position([1, 2], [1]), None)

    def test_find_token_present(self):
        self.assertEqual(find_token_in_response([10, 20, 30], 20), 1)

    def test_find_token_absent(self):
        self.assertIsNone(find_token_in_response([10, 20], 99))

    def test_find_token_with_start(self):
        self.assertEqual(find_token_in_response([10, 20, 10, 30], 10, start=1), 2)


# ---------------------------------------------------------------------------
# sum_masked_logprob / lookup_logprob / lookup_entry
# ---------------------------------------------------------------------------

class TestLogprobHelpers(unittest.TestCase):

    def test_sum_masked(self):
        entries = [{"logprob": -1.0}, {"logprob": -2.5}]
        self.assertAlmostEqual(sum_masked_logprob(entries), -3.5)

    def test_sum_empty(self):
        self.assertAlmostEqual(sum_masked_logprob([]), 0.0)

    def test_lookup_found(self):
        entries = [{"abs_position": 5, "logprob": -1.23}]
        self.assertAlmostEqual(lookup_logprob(entries, 5), -1.23)

    def test_lookup_missing(self):
        self.assertIsNone(lookup_logprob([], 5))

    def test_lookup_entry(self):
        e = {"abs_position": 3, "token_id": 42}
        self.assertEqual(lookup_entry([e], 3), e)
        self.assertIsNone(lookup_entry([e], 4))


# ---------------------------------------------------------------------------
# compute_exact_margin
# ---------------------------------------------------------------------------

class TestExactMargin(unittest.TestCase):

    def test_basic(self):
        p_ch = [{"logprob": -1.0}, {"logprob": -2.0}]
        p_re = [{"logprob": -3.0}]
        r_ch = [{"logprob": -4.0}, {"logprob": -5.0}]
        r_re = [{"logprob": -6.0}]
        # (-1-2) - (-4-5) = -3 - -9 = 6
        # (-3) - (-6) = 3
        # margin = 6 - 3 = 3
        self.assertAlmostEqual(compute_exact_margin(p_ch, p_re, r_ch, r_re), 3.0)

    def test_zero_margin(self):
        e = [{"logprob": -1.0}]
        self.assertAlmostEqual(compute_exact_margin(e, e, e, e), 0.0)


# ---------------------------------------------------------------------------
# assemble_solution_metrics
# ---------------------------------------------------------------------------

class TestAssembleSolutionMetrics(unittest.TestCase):

    def _make_entry(self, abs_pos, token_id, logprob, rank=0, greedy_gap=0.0):
        return {
            "abs_position": abs_pos,
            "token_id": token_id,
            "token_text": f"tok{token_id}",
            "logprob": logprob,
            "rank": rank,
            "greedy_gap": greedy_gap,
        }

    def test_no_solution(self):
        result = assemble_solution_metrics(
            sol_abs_pos=None, div_pos_in_resp=None, prompt_len=5,
            policy_ch=[], policy_re=[], ref_ch=[], ref_re=[],
            exact_margin=1.0, expected_solution_id=10)
        self.assertIsNone(result["solution_chosen_shift"])
        self.assertIsNone(result["solution_analysis"])

    def test_with_solution_and_divergence(self):
        sol_id = 10
        p_ch = [self._make_entry(5, sol_id, -2.0, rank=1)]
        r_ch = [self._make_entry(5, sol_id, -5.0)]
        p_re = [self._make_entry(5, 99, -3.0, rank=2)]
        r_re = [self._make_entry(5, 99, -7.0)]

        result = assemble_solution_metrics(
            sol_abs_pos=5, div_pos_in_resp=0, prompt_len=5,
            policy_ch=p_ch, policy_re=p_re, ref_ch=r_ch, ref_re=r_re,
            exact_margin=3.0, expected_solution_id=sol_id)

        self.assertAlmostEqual(result["solution_chosen_shift"], 3.0)
        self.assertIsNotNone(result["solution_analysis"])
        self.assertEqual(result["solution_analysis"]["token_id"], sol_id)
        self.assertAlmostEqual(result["solution_analysis"]["chosen_shift"], 3.0)
        self.assertAlmostEqual(result["solution_analysis"]["rejected_shift"], 4.0)
        # shift_minus_rejected_div = 3.0 - 4.0 = -1.0
        self.assertAlmostEqual(
            result["solution_shift_minus_rejected_div"], -1.0)
        self.assertAlmostEqual(
            result["solution_analysis"]["shift_minus_rejected_div"], -1.0)


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary(unittest.TestCase):

    def test_basic(self):
        results = [
            {"exact_margin": 10.0, "solution_share_of_margin": 0.5,
             "solution_chosen_shift": 2.0,
             "solution_analysis": {"policy_chosen_rank": 1}},
            {"exact_margin": 20.0, "solution_share_of_margin": None,
             "solution_chosen_shift": None,
             "solution_analysis": None},
        ]
        summary = build_summary(results, [], expected_solution_id=42)
        self.assertEqual(summary["n_pairs"], 2)
        self.assertAlmostEqual(summary["aggregate"]["avg_exact_margin"], 15.0)
        self.assertAlmostEqual(
            summary["aggregate"]["avg_solution_share_of_margin"], 0.5)
        self.assertEqual(summary["aggregate"]["solution_token_id"], 42)


# ---------------------------------------------------------------------------
# build_style_patterns
# ---------------------------------------------------------------------------

class TestBuildStylePatterns(unittest.TestCase):

    def test_returns_named_patterns(self):
        tok = _StubTokenizer()
        patterns = build_style_patterns(tok)
        names = [name for name, _ in patterns]
        self.assertIn("solution_keyword", names)
        self.assertIn("final_wrapper", names)
        self.assertIn("boxed_answer", names)
        self.assertIn("numbered_step_prefix", names)
        # 10 numbered steps
        self.assertEqual(names.count("numbered_step_prefix"), 10)


if __name__ == "__main__":
    unittest.main()
