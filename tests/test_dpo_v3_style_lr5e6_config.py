"""Tests for DPO v3 style lr=5e-6 config."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polaris.config import build_config

CONFIG_PATH = "configs/qwen3_0_6b/dpo_v3_style_lr5e6.yaml"
BASE_PATH = "configs/base.yaml"


def _load_config():
    return build_config(BASE_PATH, CONFIG_PATH)


class TestDpoV3StyleLr5e6Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CONFIG_PATH):
            raise unittest.SkipTest("Config not found")
        cls.cfg = _load_config()

    def test_run_name(self):
        self.assertEqual(self.cfg["run"]["name"], "qwen3_0_6b_dpo_v3_style_lr5e6")

    def test_learning_rate_is_5e6(self):
        self.assertAlmostEqual(self.cfg["training"]["learning_rate"], 5.0e-6)

    def test_learning_rate_differs_from_v2(self):
        v2 = build_config(BASE_PATH, "configs/qwen3_0_6b/dpo_v2_style.yaml")
        self.assertNotAlmostEqual(
            self.cfg["training"]["learning_rate"],
            v2["training"]["learning_rate"],
        )

    def test_beta(self):
        self.assertAlmostEqual(self.cfg["dpo"]["beta"], 0.1)

    def test_batch_size(self):
        self.assertEqual(self.cfg["training"]["batch_size"], 2)

    def test_grad_accum(self):
        self.assertEqual(self.cfg["training"]["gradient_accumulation_steps"], 4)

    def test_num_epochs(self):
        self.assertEqual(self.cfg["training"]["num_epochs"], 1)

    def test_data_path(self):
        self.assertEqual(
            self.cfg["data"]["path"],
            "data/math/splits/dpo_v2_style_train_449.jsonl",
        )

    def test_policy_adapter_path(self):
        self.assertIn("000030", self.cfg["dpo"]["policy_adapter_path"])

    def test_ref_adapter_path(self):
        self.assertIn("000030", self.cfg["dpo"]["ref_adapter_path"])

    def test_lora_config(self):
        self.assertTrue(self.cfg["lora"]["enabled"])
        self.assertEqual(self.cfg["lora"]["r"], 32)
        self.assertEqual(self.cfg["lora"]["alpha"], 32)
        expected = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.assertEqual(self.cfg["lora"]["target_modules"], expected)

    def test_seed(self):
        self.assertEqual(self.cfg["training"]["seed"], 42)

    def test_max_seq_length(self):
        self.assertEqual(self.cfg["training"]["max_seq_length"], 2048)

    def test_tags_include_v3_and_lr5e6(self):
        tags = self.cfg["run"]["tags"]
        self.assertIn("v3", tags)
        self.assertIn("lr5e6", tags)


if __name__ == "__main__":
    unittest.main()
