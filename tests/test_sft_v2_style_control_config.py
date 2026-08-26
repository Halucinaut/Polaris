"""Tests for SFT v2 style control config."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polaris.config import build_config

CONFIG_PATH = "configs/qwen3_0_6b/sft_v2_style_control.yaml"
BASE_PATH = "configs/base.yaml"


def _load_config():
    return build_config(BASE_PATH, CONFIG_PATH)


class TestSftV2StyleControlConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CONFIG_PATH):
            raise unittest.SkipTest("Config not found")
        cls.cfg = _load_config()

    def test_run_name(self):
        self.assertEqual(self.cfg["run"]["name"], "qwen3_0_6b_sft_v2_style_control")

    def test_model_path(self):
        self.assertEqual(self.cfg["model"]["path"], "models/qwen3_0_6b/mlx")

    def test_data_path(self):
        self.assertEqual(
            self.cfg["data"]["path"],
            "data/math/splits/sft_v2_style_control_train_449.jsonl",
        )

    def test_init_adapter_path(self):
        self.assertEqual(
            self.cfg["sft"]["init_adapter_path"],
            "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
        )

    def test_training_method(self):
        self.assertEqual(self.cfg["training"]["method"], "sft")

    def test_num_epochs(self):
        self.assertEqual(self.cfg["training"]["num_epochs"], 1)

    def test_batch_size(self):
        self.assertEqual(self.cfg["training"]["batch_size"], 2)

    def test_grad_accum(self):
        self.assertEqual(self.cfg["training"]["gradient_accumulation_steps"], 4)

    def test_learning_rate(self):
        self.assertAlmostEqual(self.cfg["training"]["learning_rate"], 5.0e-5)

    def test_max_seq_length(self):
        self.assertEqual(self.cfg["training"]["max_seq_length"], 2048)

    def test_seed(self):
        self.assertEqual(self.cfg["training"]["seed"], 42)

    def test_lora_enabled(self):
        self.assertTrue(self.cfg["lora"]["enabled"])

    def test_lora_r(self):
        self.assertEqual(self.cfg["lora"]["r"], 32)

    def test_lora_alpha(self):
        self.assertEqual(self.cfg["lora"]["alpha"], 32)

    def test_lora_target_modules(self):
        expected = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.assertEqual(self.cfg["lora"]["target_modules"], expected)

    def test_tags_include_control(self):
        self.assertIn("control", self.cfg["run"]["tags"])


if __name__ == "__main__":
    unittest.main()
