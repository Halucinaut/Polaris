"""Tests for DPO v2 style-controlled config: merge, field values, and data integrity."""

import unittest
from pathlib import Path

import yaml

from polaris.config import build_config, load_yaml


BASE_PATH = Path("configs/base.yaml")
OVERRIDE_PATH = Path("configs/qwen3_0_6b/dpo_v2_style.yaml")
TRAIN_DATA_PATH = Path("data/math/splits/dpo_v2_style_train_449.jsonl")

# Canonical values from M1 SFT
M1_CHECKPOINT = "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final"
BASE_MODEL = "models/qwen3_0_6b/mlx"


class TestDpoV2StyleConfig(unittest.TestCase):
    """Validate merged config fields and data file integrity."""

    @classmethod
    def setUpClass(cls):
        cls.config = build_config(BASE_PATH, OVERRIDE_PATH)

    # --- Run metadata ---

    def test_run_name(self):
        self.assertEqual(self.config["run"]["name"], "qwen3_0_6b_dpo_v2_style")

    def test_tags(self):
        self.assertEqual(self.config["run"]["tags"], ["dpo", "math", "0.6b", "v2", "style"])

    # --- Data ---

    def test_data_name(self):
        self.assertEqual(self.config["data"]["name"], "gsm8k_dpo_v2_style_train_449")

    def test_data_path_not_v1(self):
        """Must never point at the DPO v1 dataset."""
        self.assertNotIn("dpo_v1", self.config["data"]["path"])

    def test_data_path_points_to_train_449(self):
        self.assertEqual(self.config["data"]["path"], str(TRAIN_DATA_PATH))

    def test_data_path_exists_and_has_449_records(self):
        import json
        path = Path(self.config["data"]["path"])
        self.assertTrue(path.exists(), f"data file not found: {path}")
        with path.open() as f:
            count = sum(1 for line in f if line.strip())
        self.assertEqual(count, 449)

    def test_max_samples_null(self):
        self.assertIsNone(self.config["data"]["max_samples"])

    # --- Training hyperparameters ---

    def test_training_method(self):
        self.assertEqual(self.config["training"]["method"], "dpo")

    def test_num_epochs(self):
        self.assertEqual(self.config["training"]["num_epochs"], 1)

    def test_batch_size(self):
        self.assertEqual(self.config["training"]["batch_size"], 2)

    def test_gradient_accumulation_steps(self):
        self.assertEqual(self.config["training"]["gradient_accumulation_steps"], 4)

    def test_learning_rate(self):
        self.assertAlmostEqual(self.config["training"]["learning_rate"], 5.0e-7)

    def test_max_seq_length(self):
        self.assertEqual(self.config["training"]["max_seq_length"], 2048)

    def test_seed(self):
        self.assertEqual(self.config["training"]["seed"], 42)

    # --- DPO specific ---

    def test_beta(self):
        self.assertAlmostEqual(self.config["dpo"]["beta"], 0.1)

    def test_policy_adapter_path(self):
        self.assertEqual(self.config["dpo"]["policy_adapter_path"], M1_CHECKPOINT)

    def test_ref_model_path(self):
        self.assertEqual(self.config["dpo"]["ref_model_path"], BASE_MODEL)

    def test_ref_adapter_path(self):
        self.assertEqual(self.config["dpo"]["ref_adapter_path"], M1_CHECKPOINT)

    # --- LoRA ---

    def test_lora_enabled(self):
        self.assertTrue(self.config["lora"]["enabled"])

    def test_lora_r(self):
        self.assertEqual(self.config["lora"]["r"], 32)

    def test_lora_alpha(self):
        self.assertEqual(self.config["lora"]["alpha"], 32)

    def test_lora_target_modules(self):
        self.assertEqual(
            self.config["lora"]["target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    # --- Model ---

    def test_model_path(self):
        self.assertEqual(self.config["model"]["path"], BASE_MODEL)

    # --- Override file itself ---

    def test_override_file_loads_cleanly(self):
        """The override YAML must parse without errors."""
        raw = load_yaml(OVERRIDE_PATH)
        self.assertIsInstance(raw, dict)
        self.assertIn("run", raw)
        self.assertIn("dpo", raw)


if __name__ == "__main__":
    unittest.main()
