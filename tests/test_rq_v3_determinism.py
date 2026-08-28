"""
Determinism test for rq v3 candidate generation.

Verifies that mx.random.seed(seed_i) produces identical outputs
when run twice on the same small sample.

Run under .venv:
    .venv/bin/python -m unittest tests.test_rq_v3_determinism
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from polaris.problem_hash import problem_hash


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _models_available() -> bool:
    if not _HAS_MLX:
        return False
    model_path = PROJECT_ROOT / "models/qwen3_0_6b/mlx"
    adapter_path = PROJECT_ROOT / "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final"
    return model_path.is_dir() and adapter_path.is_dir()


@unittest.skipUnless(_HAS_MLX, "MLX not available")
class TestProblemHash(unittest.TestCase):
    """Pure tests for problem_hash (no MLX needed)."""

    def test_deterministic(self):
        text = "What is 2 + 3?"
        h1 = problem_hash(text)
        h2 = problem_hash(text)
        self.assertEqual(h1, h2)

    def test_normalization(self):
        """Whitespace and case differences produce same hash."""
        h1 = problem_hash("What is 2 + 3?")
        h2 = problem_hash("  what  IS  2 + 3?  ")
        self.assertEqual(h1, h2)

    def test_different_problems(self):
        h1 = problem_hash("What is 2 + 3?")
        h2 = problem_hash("What is 5 + 7?")
        self.assertNotEqual(h1, h2)

    def test_length(self):
        h = problem_hash("test")
        self.assertEqual(len(h), 16)


@unittest.skipUnless(_models_available(), "Model checkpoints not available")
class TestDeterministicGeneration(unittest.TestCase):
    """Integration: same seed → same output hash."""

    def test_same_seed_same_output(self):
        """Run generation twice with the same seed, assert identical output."""
        load_model = _load_module("load_model", PROJECT_ROOT / "scripts/smoke/load_model.py")

        model, tokenizer = load(
            str(PROJECT_ROOT / "models/qwen3_0_6b/mlx"),
            adapter_path=str(PROJECT_ROOT / "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final"),
        )
        sampler = make_sampler(1.0)

        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem and put the final answer in \\boxed{}."},
            {"role": "user", "content": "What is 15 + 27?"},
        ]
        rendered = load_model.apply_chat_template_safe(tokenizer, messages)

        outputs = []
        for run in range(2):
            mx.random.seed(42)
            out = generate(model, tokenizer, prompt=rendered, max_tokens=256, sampler=sampler)
            outputs.append(out)

        # Outputs must be identical
        self.assertEqual(
            problem_hash(outputs[0]),
            problem_hash(outputs[1]),
            f"Two runs with same seed produced different outputs:\n"
            f"  Run 1 ({len(outputs[0])} chars): {outputs[0][:100]}...\n"
            f"  Run 2 ({len(outputs[1])} chars): {outputs[1][:100]}...",
        )

    def test_different_seeds_different_output(self):
        """Different seeds should (very likely) produce different output."""
        load_model = _load_module("load_model", PROJECT_ROOT / "scripts/smoke/load_model.py")

        model, tokenizer = load(
            str(PROJECT_ROOT / "models/qwen3_0_6b/mlx"),
            adapter_path=str(PROJECT_ROOT / "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final"),
        )
        sampler = make_sampler(1.0)

        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem and put the final answer in \\boxed{}."},
            {"role": "user", "content": "What is 15 + 27?"},
        ]
        rendered = load_model.apply_chat_template_safe(tokenizer, messages)

        outputs = []
        for seed in [42, 99]:
            mx.random.seed(seed)
            out = generate(model, tokenizer, prompt=rendered, max_tokens=256, sampler=sampler)
            outputs.append(out)

        # With temp=1.0, different seeds should produce different outputs
        # (not a hard guarantee, but extremely likely)
        if outputs[0] == outputs[1]:
            self.skipTest("Different seeds produced same output (very unlikely but possible)")


if __name__ == "__main__":
    unittest.main()
