"""Tests for DPO v4 style minimal config."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polaris.config import build_config

CONFIG_PATH = "configs/qwen3_0_6b/dpo_v4_style_minimal.yaml"
BASE_PATH = "configs/base.yaml"


def _load_config():
    return build_config(BASE_PATH, CONFIG_PATH)


class TestDpoV4StyleMinimalConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CONFIG_PATH):
            raise unittest.SkipTest("Config not found")
        cls.cfg = _load_config()

    def test_run_name(self):
        self.assertEqual(self.cfg["run"]["name"], "qwen3_0_6b_dpo_v4_style_minimal")

    def test_learning_rate_is_v2_original(self):
        self.assertAlmostEqual(self.cfg["training"]["learning_rate"], 5.0e-7)

    def test_data_path_is_minimal(self):
        self.assertEqual(
            self.cfg["data"]["path"],
            "data/math/pilots/dpo_v4_minimal_449.jsonl",
        )

    def test_beta(self):
        self.assertAlmostEqual(self.cfg["dpo"]["beta"], 0.1)

    def test_batch_size(self):
        self.assertEqual(self.cfg["training"]["batch_size"], 2)

    def test_grad_accum(self):
        self.assertEqual(self.cfg["training"]["gradient_accumulation_steps"], 4)

    def test_num_epochs(self):
        self.assertEqual(self.cfg["training"]["num_epochs"], 1)

    def test_tags_include_v4_and_minimal(self):
        tags = self.cfg["run"]["tags"]
        self.assertIn("v4", tags)
        self.assertIn("minimal", tags)

    def test_lora_config(self):
        self.assertTrue(self.cfg["lora"]["enabled"])
        self.assertEqual(self.cfg["lora"]["r"], 32)
        self.assertEqual(self.cfg["lora"]["alpha"], 32)


if __name__ == "__main__":
    unittest.main()
