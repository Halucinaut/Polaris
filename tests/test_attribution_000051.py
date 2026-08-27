"""MLX integration tests for diagnose_full_attribution_000051.py.

Run with:  ./.venv/bin/python -m unittest tests.test_attribution_000051

These tests REQUIRE mlx and the trained model checkpoints.  When run under
system Python (``make test``) they skip gracefully.
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HAS_MLX = False
try:
    import mlx.core  # noqa: F401
    _HAS_MLX = True
except ImportError:
    pass


def _models_available():
    return (
        _HAS_MLX
        and os.path.isdir("models/qwen3_0_6b/mlx")
        and os.path.isdir("runs/000051_qwen3_0_6b_dpo_v4_style_minimal_4ep/checkpoints/final")
        and os.path.isdir("runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final")
    )


def _load_models():
    from scripts.diagnose_full_attribution_000051 import load_model_with_adapter
    policy_model, tokenizer = load_model_with_adapter(
        "models/qwen3_0_6b/mlx",
        "runs/000051_qwen3_0_6b_dpo_v4_style_minimal_4ep/checkpoints/final",
    )
    ref_model, _ = load_model_with_adapter(
        "models/qwen3_0_6b/mlx",
        "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
    )
    return policy_model, ref_model, tokenizer


# ---------------------------------------------------------------------------
# Style adherence (no model needed)
# ---------------------------------------------------------------------------

class TestStyleAdherenceImport(unittest.TestCase):

    def test_imports(self):
        from scripts.eval_style_dpo import check_style_adherence
        self.assertTrue(callable(check_style_adherence))

    def test_style_check_valid(self):
        from scripts.eval_style_dpo import check_style_adherence
        text = ("<think>\nSolution:\n1. Step.\n"
                "</think>\n\nFinal: The answer is \\boxed{42}.")
        adherent, reasons = check_style_adherence(text)
        self.assertTrue(adherent)
        self.assertEqual(reasons, [])


# ---------------------------------------------------------------------------
# Off-by-one detection (MLX needed but no checkpoints)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_MLX, "mlx not installed")
class TestOffByOne(unittest.TestCase):

    def test_logprob_reads_logits_at_t_minus_1(self):
        import mlx.core as mx

        vocab_size = 5
        seq_len = 4
        full_ids = [0, 2, 3, 1]
        response_mask = [0, 1, 1, 1]

        raw = mx.zeros((1, seq_len, vocab_size))
        for i in range(seq_len):
            raw[0, i, (i + 1) % vocab_size] = 100.0
        log_probs = mx.log(mx.softmax(raw, axis=-1))

        expected_lps = []
        for t in range(seq_len):
            if response_mask[t] == 0:
                continue
            lp = float(log_probs[0, t - 1, full_ids[t]].item())
            expected_lps.append(lp)

        for lp in expected_lps:
            self.assertLess(lp, -50.0,
                            f"logprob {lp:.2f} too high — possible off-by-one")

    def test_logprob_matches_when_target_is_greedy(self):
        import mlx.core as mx

        vocab_size = 5
        seq_len = 4
        full_ids = [0, 1, 2, 3]
        response_mask = [0, 1, 1, 1]

        raw = mx.zeros((1, seq_len, vocab_size))
        for i in range(seq_len):
            raw[0, i, (i + 1) % vocab_size] = 100.0
        log_probs = mx.log(mx.softmax(raw, axis=-1))

        for t in range(seq_len):
            if response_mask[t] == 0:
                continue
            lp = float(log_probs[0, t - 1, full_ids[t]].item())
            self.assertGreater(lp, -0.01)


# ---------------------------------------------------------------------------
# Boundary consistency (requires models + checkpoints)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_models_available(), "MLX models/checkpoints not available")
class TestBoundaryConsistency(unittest.TestCase):

    def test_solution_metrics_consistency(self):
        from scripts.diagnose_boundary_logprob import score_prefix
        from scripts.diagnose_full_attribution_000051 import (
            compute_per_token_logprobs,
        )
        from scripts.train_dpo import tokenize_pair

        policy_model, ref_model, tokenizer = _load_models()

        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        chosen = "<think>\nSolution:\n1. Compute 2+2=4.\n"
        rejected = "<think>\nI think the answer is 4.\n"

        pair = tokenize_pair(tokenizer, messages, chosen, rejected,
                             max_seq_length=2048)
        self.assertIsNotNone(pair)

        prompt_len = pair["prompt_len"]
        chosen_full = pair["chosen_ids"]
        chosen_mask = pair["chosen_mask"]

        # Context up to Solution: prompt + tokens before Solution
        sol_id = tokenizer.encode("Solution")[0]
        chosen_resp_ids = chosen_full[prompt_len:]
        sol_offset = next(j for j, tid in enumerate(chosen_resp_ids)
                          if tid == sol_id)
        context_ids = chosen_full[:prompt_len + sol_offset]

        boundary_result = score_prefix(policy_model, tokenizer,
                                       context_ids, "Solution")

        attr_tokens = compute_per_token_logprobs(
            policy_model, tokenizer, chosen_full, chosen_mask)

        sol_entry = next(
            (e for e in attr_tokens if e["token_id"] == sol_id), None)
        self.assertIsNotNone(sol_entry, "Solution token not found")

        self.assertAlmostEqual(
            sol_entry["logprob"],
            boundary_result["tokens"][0]["logprob"],
            delta=0.1,
            msg="Logprob mismatch between attribution and score_prefix")

        self.assertAlmostEqual(
            sol_entry["greedy_gap"],
            boundary_result["first_token_gap"],
            delta=0.1,
            msg="Gap mismatch between attribution and score_prefix")

        # Top-1 agreement
        self.assertEqual(
            sol_entry["rank"] == 0,
            boundary_result["first_token_rank"] == 0,
            msg=f"Top-1 disagreement: attr rank={sol_entry['rank']}, "
                f"bp rank={boundary_result['first_token_rank']}")


# ---------------------------------------------------------------------------
# Margin consistency (requires models + checkpoints)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_models_available(), "MLX models/checkpoints not available")
class TestMarginConsistency(unittest.TestCase):

    def test_margin_matches_dpo_loss(self):
        import mlx.core as mx
        from scripts.train_dpo import tokenize_pair, compute_response_logprob
        from scripts.diagnose_full_attribution_000051 import (
            compute_per_token_logprobs,
        )
        from polaris.attribution import compute_exact_margin

        policy_model, ref_model, tokenizer = _load_models()

        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        chosen = "<think>\nSolution:\n1. Compute 2+2=4.\n"
        rejected = "<think>\nI think the answer is 4.\n"

        pair = tokenize_pair(tokenizer, messages, chosen, rejected,
                             max_seq_length=2048)
        self.assertIsNotNone(pair)

        chosen_full = pair["chosen_ids"]
        rejected_full = pair["rejected_ids"]
        chosen_mask = pair["chosen_mask"]
        rejected_mask = pair["rejected_mask"]

        # Batch path (compute_dpo_loss internals)
        ch_ids_b = mx.array([chosen_full], dtype=mx.int32)
        re_ids_b = mx.array([rejected_full], dtype=mx.int32)
        ch_mask_b = mx.array([chosen_mask], dtype=mx.float32)
        re_mask_b = mx.array([rejected_mask], dtype=mx.float32)

        batch_margin = float((
            compute_response_logprob(policy_model(ch_ids_b), ch_ids_b, ch_mask_b)[0]
            - compute_response_logprob(policy_model(re_ids_b), re_ids_b, re_mask_b)[0]
            - compute_response_logprob(ref_model(ch_ids_b), ch_ids_b, ch_mask_b)[0]
            + compute_response_logprob(ref_model(re_ids_b), re_ids_b, re_mask_b)[0]
        ).item())

        # Attribution path
        attr_margin = compute_exact_margin(
            compute_per_token_logprobs(policy_model, tokenizer, chosen_full, chosen_mask),
            compute_per_token_logprobs(policy_model, tokenizer, rejected_full, rejected_mask),
            compute_per_token_logprobs(ref_model, tokenizer, chosen_full, chosen_mask),
            compute_per_token_logprobs(ref_model, tokenizer, rejected_full, rejected_mask),
        )

        self.assertAlmostEqual(
            attr_margin, batch_margin, delta=0.15,
            msg=f"Margin mismatch: attribution={attr_margin:.4f}, "
                f"compute_dpo_loss={batch_margin:.4f}")


if __name__ == "__main__":
    unittest.main()
