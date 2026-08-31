"""Tests for group-relative advantage computation."""

import unittest

from polaris.trainers.base import GroupStats, compute_group_relative_advantage


class TestGroupRelativeAdvantage(unittest.TestCase):

    def test_uniform_rewards_zero_variance(self):
        stats = compute_group_relative_advantage([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(stats.zero_variance)
        self.assertEqual(stats.group_size, 4)
        self.assertAlmostEqual(stats.reward_mean, 1.0)
        self.assertAlmostEqual(stats.reward_std, 0.0)
        self.assertEqual(stats.advantages, [0.0, 0.0, 0.0, 0.0])

    def test_single_element_zero_variance(self):
        stats = compute_group_relative_advantage([5.0])
        self.assertTrue(stats.zero_variance)
        self.assertEqual(stats.group_size, 1)
        self.assertAlmostEqual(stats.reward_mean, 5.0)
        self.assertEqual(stats.advantages, [0.0])

    def test_two_elements_normalization(self):
        stats = compute_group_relative_advantage([0.0, 2.0])
        self.assertFalse(stats.zero_variance)
        self.assertAlmostEqual(stats.reward_mean, 1.0)
        self.assertAlmostEqual(stats.advantages[0], -1.0)
        self.assertAlmostEqual(stats.advantages[1], 1.0)

    def test_mixed_rewards_normalization(self):
        stats = compute_group_relative_advantage([0.0, 0.5, 1.0])
        self.assertFalse(stats.zero_variance)
        self.assertAlmostEqual(stats.reward_mean, 0.5, places=5)
        # advantages should sum to ~0
        self.assertAlmostEqual(sum(stats.advantages), 0.0, places=5)

    def test_all_negative_rewards(self):
        stats = compute_group_relative_advantage([-1.0, -0.5, 0.0])
        self.assertFalse(stats.zero_variance)
        self.assertAlmostEqual(sum(stats.advantages), 0.0, places=5)

    def test_near_zero_std_threshold(self):
        # Rewards differ by < 1e-8 should be treated as zero variance
        stats = compute_group_relative_advantage([1.0, 1.0 + 1e-10, 1.0 - 1e-10])
        self.assertTrue(stats.zero_variance)
        self.assertEqual(stats.advantages, [0.0, 0.0, 0.0])

    def test_empty_group_raises(self):
        with self.assertRaises(ValueError):
            compute_group_relative_advantage([])

    def test_advantages_are_floats(self):
        stats = compute_group_relative_advantage([0.0, 1.0, 2.0])
        for a in stats.advantages:
            self.assertIsInstance(a, float)

    def test_large_group(self):
        rewards = [float(i) for i in range(16)]
        stats = compute_group_relative_advantage(rewards)
        self.assertFalse(stats.zero_variance)
        self.assertEqual(stats.group_size, 16)
        self.assertAlmostEqual(sum(stats.advantages), 0.0, places=5)

    def test_two_near_identical_rewards(self):
        stats = compute_group_relative_advantage([1.0, 1.0])
        self.assertTrue(stats.zero_variance)
        self.assertEqual(stats.advantages, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
