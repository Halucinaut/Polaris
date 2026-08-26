"""Tests for build_microbatch_groups (no MLX dependency)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_sft import build_microbatch_groups


class TestBuildMicrobatchGroups(unittest.TestCase):
    def test_449_samples_produces_57_groups(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        self.assertEqual(len(groups), 57)

    def test_449_samples_last_group_has_1_sample(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        last = groups[-1]
        self.assertEqual(last["samples"], 1)
        self.assertEqual(last["micro_batches"], 1)

    def test_449_samples_first_56_groups_have_8_samples(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        for i, g in enumerate(groups[:56]):
            self.assertEqual(g["samples"], 8, f"group {i} samples")
            self.assertEqual(g["micro_batches"], 4, f"group {i} micro_batches")

    def test_449_samples_total_coverage(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        total = sum(g["samples"] for g in groups)
        self.assertEqual(total, 449)

    def test_no_wrapping_indices(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        for i, g in enumerate(groups):
            self.assertGreaterEqual(g["start"], 0)
            self.assertLessEqual(g["end"], 449)
            self.assertEqual(g["end"] - g["start"], g["samples"])
        # Groups are contiguous and non-overlapping
        for i in range(1, len(groups)):
            self.assertEqual(groups[i]["start"], groups[i - 1]["end"])

    def test_max_steps_10_consumes_first_80_samples(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        first_10 = groups[:10]
        total = sum(g["samples"] for g in first_10)
        self.assertEqual(total, 80)

    def test_each_step_is_8_samples_for_full_groups(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        for g in groups[:56]:
            self.assertEqual(g["samples"], 2 * 4)

    def test_exact_division_case(self):
        # 16 samples, batch=2, grad_accum=4 → exactly 2 groups
        groups = build_microbatch_groups(16, batch_size=2, grad_accum=4)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(g["samples"] for g in groups), 16)
        for g in groups:
            self.assertEqual(g["samples"], 8)

    def test_single_sample(self):
        groups = build_microbatch_groups(1, batch_size=2, grad_accum=4)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["samples"], 1)
        self.assertEqual(groups[0]["micro_batches"], 1)

    def test_batch_size_larger_than_samples(self):
        groups = build_microbatch_groups(3, batch_size=4, grad_accum=4)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["samples"], 3)
        self.assertEqual(groups[0]["micro_batches"], 1)

    def test_500_samples_batch4_gradaccum4(self):
        # M1 config: 500 samples, batch_size=4, grad_accum=4
        groups = build_microbatch_groups(500, batch_size=4, grad_accum=4)
        # 500/4 = 125 micro-batches, 125/4 = 31 full + 1 partial = 32 groups
        self.assertEqual(len(groups), 32)
        self.assertEqual(groups[-1]["samples"], 4)  # last group: 1 micro-batch of 4
        self.assertEqual(groups[-1]["micro_batches"], 1)
        total = sum(g["samples"] for g in groups)
        self.assertEqual(total, 500)

    def test_negative_samples_raises(self):
        with self.assertRaises(ValueError):
            build_microbatch_groups(-1, 2, 4)

    def test_zero_batch_size_raises(self):
        with self.assertRaises(ValueError):
            build_microbatch_groups(10, 0, 4)

    def test_zero_grad_accum_raises(self):
        with self.assertRaises(ValueError):
            build_microbatch_groups(10, 2, 0)

    def test_groups_contain_contiguous_indices(self):
        groups = build_microbatch_groups(449, batch_size=2, grad_accum=4)
        flat_indices = []
        for g in groups:
            # Verify micro-batch boundaries within group
            for m in range(g["micro_batches"]):
                m_start = g["start"] + m * 2
                m_end = min(m_start + 2, g["end"])
                flat_indices.extend(range(m_start, m_end))
        self.assertEqual(flat_indices, list(range(449)))


if __name__ == "__main__":
    unittest.main()
