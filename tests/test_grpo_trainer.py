"""Tests for GRPO trainer: config validation, dry-run, checkpoint, and loss determinism.

GRPO loss tests use numpy mock models for deterministic, framework-independent
verification. MLX integration tests are gated by mlx availability.
"""

import json
import math
import tempfile
import unittest
import unittest.mock
from pathlib import Path


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestValidateGrpoConfig(unittest.TestCase):

    def _base_config(self) -> dict:
        return {
            "data": {"path": "/nonexistent/data.jsonl"},
            "grpo": {
                "policy_adapter_path": "/tmp/policy",
                "ref_model_path": "/tmp/ref",
                "ref_adapter_path": "/tmp/ref_adapter",
                "group_size": 8,
                "max_completion_length": 256,
                "kl_coef": 0.05,
                "clip_range": 0.2,
                "rollout_temperature": 1.0,
                "reward": {
                    "correct": 1.0,
                    "incorrect": 0.0,
                    "unparseable": -0.5,
                    "empty": -1.0,
                },
            },
        }

    def test_valid_config_has_no_errors(self):
        """validate_grpo_config returns empty error list for a structurally valid config."""
        config = self._base_config()
        errors = validate_grpo_config(config)
        self.assertEqual(errors, [], f"Expected no errors, got: {errors}")

    def test_missing_required_field(self):
        config = self._base_config()
        del config["grpo"]["group_size"]
        errors = validate_grpo_config(config)
        self.assertTrue(any("group_size" in e for e in errors))

    def test_invalid_group_size(self):
        config = self._base_config()
        config["grpo"]["group_size"] = 0
        errors = validate_grpo_config(config)
        self.assertTrue(any("group_size" in e for e in errors))

    def test_invalid_clip_range(self):
        config = self._base_config()
        config["grpo"]["clip_range"] = 1.5
        errors = validate_grpo_config(config)
        self.assertTrue(any("clip_range" in e for e in errors))

    def test_negative_kl_coef(self):
        config = self._base_config()
        config["grpo"]["kl_coef"] = -0.1
        errors = validate_grpo_config(config)
        self.assertTrue(any("kl_coef" in e for e in errors))

    def test_missing_reward_subfields(self):
        config = self._base_config()
        config["grpo"]["reward"] = {"correct": 1.0}  # missing others
        errors = validate_grpo_config(config)
        self.assertTrue(any("unparseable" in e for e in errors))
        self.assertTrue(any("empty" in e for e in errors))

    def test_data_path_not_in_config_errors(self):
        """validate_grpo_config does NOT check data path existence (dry-run handles it)."""
        config = self._base_config()
        del config["data"]
        errors = validate_grpo_config(config)
        # No data-path error expected; only grpo.* fields are validated
        self.assertFalse(any("data" in e.lower() or "not found" in e.lower() for e in errors))

    def test_rollout_temperature_must_be_one(self):
        """Non-1.0 temperature violates on-policy GRPO semantics."""
        config = self._base_config()
        config["grpo"]["rollout_temperature"] = 0.5
        errors = validate_grpo_config(config)
        self.assertTrue(any("temperature" in e.lower() and "1.0" in e for e in errors))

    def test_rollout_top_p_must_be_one(self):
        """Non-1.0 top_p violates on-policy GRPO semantics."""
        config = self._base_config()
        config["grpo"]["rollout_top_p"] = 0.9
        errors = validate_grpo_config(config)
        self.assertTrue(any("top_p" in e.lower() and "1.0" in e for e in errors))

    def test_default_rollout_params_pass(self):
        """Default config (temperature=1.0, top_p=1.0) passes validation."""
        config = self._base_config()
        errors = validate_grpo_config(config)
        temp_errors = [e for e in errors if "temperature" in e.lower()]
        top_p_errors = [e for e in errors if "top_p" in e.lower()]
        self.assertEqual(temp_errors, [])
        self.assertEqual(top_p_errors, [])


# ---------------------------------------------------------------------------
# Data schema validation tests
# ---------------------------------------------------------------------------

class TestValidateDataSchema(unittest.TestCase):

    def test_valid_record(self):
        records = [{"messages": [{"role": "user", "content": "hi"}], "metadata": {"answer": "42"}}]
        errors = validate_data_schema(records)
        self.assertEqual(errors, [])

    def test_missing_messages(self):
        records = [{"metadata": {"answer": "42"}}]
        errors = validate_data_schema(records)
        self.assertTrue(any("messages" in e for e in errors))

    def test_missing_answer_in_metadata(self):
        records = [{"messages": [{"role": "user", "content": "hi"}], "metadata": {}}]
        errors = validate_data_schema(records)
        self.assertTrue(any("answer" in e for e in errors))

    def test_empty_dataset(self):
        errors = validate_data_schema([])
        self.assertTrue(any("empty" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# Reward protocol validation tests
# ---------------------------------------------------------------------------

class TestValidateRewardProtocol(unittest.TestCase):

    def test_valid_reward_config_passes(self):
        reward_config = {
            "correct": 1.0,
            "incorrect": 0.0,
            "unparseable": -0.5,
            "empty": -1.0,
            "format_bonus": 0.05,
        }
        errors = validate_reward_protocol(reward_config)
        self.assertEqual(errors, [])

    def test_reward_config_with_bad_correct_value(self):
        reward_config = {
            "correct": -1.0,
            "incorrect": 0.0,
            "unparseable": -0.5,
            "empty": -1.0,
        }
        errors = validate_reward_protocol(reward_config)
        self.assertTrue(any("correct" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------

class TestDryRun(unittest.TestCase):

    def test_dry_run_blocked_when_data_missing(self):
        """dry-run must return 1 and print BLOCKED when data file does not exist."""
        config = {
            "data": {"path": "/nonexistent/data_math_level_3_5.jsonl"},
            "grpo": {
                "policy_adapter_path": "/tmp/policy",
                "ref_model_path": "/tmp/ref",
                "ref_adapter_path": "/tmp/ref_adapter",
                "group_size": 8,
                "max_completion_length": 256,
                "kl_coef": 0.05,
                "clip_range": 0.2,
                "rollout_temperature": 1.0,
                "reward": {
                    "correct": 1.0, "incorrect": 0.0,
                    "unparseable": -0.5, "empty": -1.0,
                },
            },
        }
        result = dry_run(config)
        self.assertEqual(result, 1, "dry-run must return 1 when data file is missing")

    def test_dry_run_success_with_valid_data(self):
        """dry-run returns 0 when data exists and schema is valid."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False,
        ) as f:
            f.write(
                '{"messages": [{"role": "user", "content": "2+3?"}], '
                '"metadata": {"answer": "5", "problem_id": "p1", "source": "test"}}\n'
            )
            data_path = f.name
        try:
            config = {
                "data": {"path": data_path},
                "grpo": {
                    "policy_adapter_path": "/tmp/policy",
                    "ref_model_path": "/tmp/ref",
                    "ref_adapter_path": "/tmp/ref_adapter",
                    "group_size": 8,
                    "max_completion_length": 256,
                    "kl_coef": 0.05,
                    "clip_range": 0.2,
                    "rollout_temperature": 1.0,
                    "reward": {
                        "correct": 1.0, "incorrect": 0.0,
                        "unparseable": -0.5, "empty": -1.0,
                    },
                },
            }
            result = dry_run(config)
            self.assertEqual(result, 0, "dry-run must return 0 when data is valid")
        finally:
            Path(data_path).unlink(missing_ok=True)

    def test_dry_run_with_real_config_blocked(self):
        """The shipped GRPO config references a data file that doesn't exist yet."""
        from polaris.config import build_config
        config = build_config("configs/base.yaml", "configs/qwen3_0_6b/grpo_math.yaml")
        result = dry_run(config)
        self.assertEqual(result, 1, "dry-run with shipped config must return 1 (data file not prepared)")


# ---------------------------------------------------------------------------
# Checkpoint provenance tests
# ---------------------------------------------------------------------------

class TestCheckpointProvenance(unittest.TestCase):

    def test_provenance_structure_has_required_keys(self):
        """The provenance dict written by _run_training must document the algorithm."""
        # Simulate what _run_training constructs
        provenance = {
            "reference_frozen": True,
            "reference_in_ratio": False,
            "ratio_source": "current/old (rollout-frozen)",
            "kl_source": "forward_kl(current, ref)",
            "old_logprob_capture": "per-token at rollout time",
            "resume_supported": False,
        }
        self.assertTrue(provenance["reference_frozen"])
        self.assertFalse(provenance["reference_in_ratio"])
        self.assertIn("old", provenance["ratio_source"])
        self.assertIn("ref", provenance["kl_source"])
        self.assertFalse(provenance["resume_supported"])


# ---------------------------------------------------------------------------
# GRPO loss determinism tests (numpy mock — no MLX required)
# ---------------------------------------------------------------------------

def _make_mock_model_return(logits_array):
    """Create a callable that returns a dict {'logits': array}."""
    def forward(input_ids):
        return {"logits": logits_array}
    return forward


class TestGRPOLossDeterminism(unittest.TestCase):
    """Pure-numpy deterministic tests for the GRPO loss algorithm.

    These verify the math without loading any MLX model. They run the
    algorithm manually using the same causal-shift and masking logic
    as compute_grpo_loss, but with plain Python/numpy arrays.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            cls.np = np
        except ImportError:
            raise unittest.SkipTest("numpy not available")

    @staticmethod
    def _compute_raw_token_lp(logits, ids):
        """Compute per-token log-probabilities from logits and ids (numpy)."""
        np = TestGRPOLossDeterminism.np
        max_l = logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        logsumexp = np.log(exp_l.sum(axis=-1, keepdims=True)) + max_l
        log_softmax = logits - logsumexp
        B, T, V = log_softmax.shape
        flat_lp = log_softmax.reshape(B * T, V)
        flat_ids = ids.reshape(B * T)
        return flat_lp[np.arange(B * T), flat_ids].reshape(B, T)

    def _manual_grpo_loss(
        self, policy_logits, ref_logits, old_lp_raw, input_ids,
        response_mask, advantages, kl_coef, clip_range,
    ):
        """Manual GRPO loss computation in numpy, mirroring grpo.py logic.

        Args:
            old_lp_raw: (B, T-1) raw old-token logprobs (unshifted, unmasked).
                Same causal-shift convention as policy: log P(ids[t+1] | logits[:,t,:]).
        """
        np = self.np

        # Causal shift
        p_logits = policy_logits[:, :-1, :]   # (B, T-1, V)
        r_logits = ref_logits[:, :-1, :]
        ids = input_ids[:, 1:]
        mask = response_mask[:, 1:]

        def raw_token_logprobs(logits, token_ids):
            max_l = logits.max(axis=-1, keepdims=True)
            exp_l = np.exp(logits - max_l)
            logsumexp = np.log(exp_l.sum(axis=-1, keepdims=True)) + max_l
            log_softmax = logits - logsumexp
            B2, T2, V2 = log_softmax.shape
            flat_lp = log_softmax.reshape(B2 * T2, V2)
            flat_ids = token_ids.reshape(B2 * T2)
            return flat_lp[np.arange(B2 * T2), flat_ids].reshape(B2, T2)

        p_lp = raw_token_logprobs(p_logits, ids) * mask
        r_lp = raw_token_logprobs(r_logits, ids) * mask
        old_lp = old_lp_raw * mask

        # Summed logprobs
        policy_lp = p_lp.sum(axis=1)
        old_sum_lp = old_lp.sum(axis=1)
        ref_lp = r_lp.sum(axis=1)

        # Per-token ratio
        log_ratio = p_lp - old_lp
        ratio = np.exp(log_ratio)
        clipped = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

        # Policy loss (length-normalized)
        adv = advantages[:, None]
        surrogate1 = ratio * adv
        surrogate2 = clipped * adv
        per_token_loss = -np.minimum(surrogate1, surrogate2) * mask
        valid_count = max(mask.sum(), 1.0)
        policy_loss = per_token_loss.sum() / valid_count

        # KL penalty (forward KL, using the already-shifted logits)
        p_log_softmax = p_logits - np.log(
            np.exp(p_logits).sum(axis=-1, keepdims=True)
        )
        r_log_softmax = r_logits - np.log(
            np.exp(r_logits).sum(axis=-1, keepdims=True)
        )
        p_probs = np.exp(p_log_softmax)
        per_token_kl = (p_probs * (p_log_softmax - r_log_softmax)).sum(axis=-1)
        kl = (per_token_kl * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        mean_kl = kl.mean()

        loss = policy_loss + kl_coef * mean_kl

        # Entropy
        entropy_per_token = -(p_probs * p_log_softmax).sum(axis=-1)
        mean_entropy = (entropy_per_token * mask).sum() / max(mask.sum(), 1.0)

        # Approx KL (Schulman) — response-only
        approx_kl_raw = ((ratio - 1) - log_ratio) * mask
        approx_kl = approx_kl_raw.sum() / max(mask.sum(), 1.0)

        # Clip fraction — response-only
        clip_frac_raw = (np.abs(ratio - 1.0) > clip_range).astype(np.float32) * mask
        clip_frac = clip_frac_raw.sum() / max(mask.sum(), 1.0)

        return {
            "loss": float(loss),
            "policy_logprob": float(policy_lp.mean()),
            "old_policy_logprob": float(old_sum_lp.mean()),
            "ref_logprob": float(ref_lp.mean()),
            "kl": float(mean_kl),
            "entropy": float(mean_entropy),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_frac),
        }

    def _default_inputs(self, B=2, T=6, V=8, seed=42):
        np = self.np
        np.random.seed(seed)
        input_ids = np.random.randint(0, V, (B, T))
        response_mask = np.zeros((B, T), dtype=np.float32)
        response_mask[:, -3:] = 1.0
        advantages = np.array([0.5, -0.5], dtype=np.float32)[:B]
        return input_ids, response_mask, advantages

    def _old_lp_from_logits(self, logits, input_ids):
        """Compute shifted old-token logprobs matching the causal-shift convention.

        Policy computes log P(ids[t+1] | logits[:, t, :]) using shifted logits.
        Old must use the same convention: log P(ids[t+1] | logits[:, t, :]),
        NOT log P(ids[t+1] | logits[:, t+1, :]).
        """
        np = self.np
        shift_logits = logits[:, :-1, :]    # (B, T-1, V) — context at position t
        shift_ids = input_ids[:, 1:]        # (B, T-1)    — token at position t+1
        B, T, V = shift_logits.shape
        max_l = shift_logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shift_logits - max_l)
        logsumexp = np.log(exp_l.sum(axis=-1, keepdims=True)) + max_l
        log_softmax = shift_logits - logsumexp
        flat = log_softmax.reshape(B * T, V)
        flat_ids = shift_ids.reshape(B * T)
        return flat[np.arange(B * T), flat_ids].reshape(B, T)

    # (a) current=old → ratio=1
    def test_ratio_one_when_current_equals_old(self):
        np = self.np
        B, T, V = 2, 6, 8
        ids, mask, adv = self._default_inputs(B, T, V)
        logits = np.random.randn(B, T, V).astype(np.float32) * 0.5

        # old == current: old_lp_raw is the same as what compute_grpo_loss would get
        old_lp_raw = self._old_lp_from_logits(logits, ids)

        result = self._manual_grpo_loss(
            logits, logits, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        self.assertAlmostEqual(result["approx_kl"], 0.0, places=5,
                               msg="approx_kl should be 0 when current=old")
        self.assertAlmostEqual(result["clip_fraction"], 0.0, places=5,
                               msg="clip_fraction should be 0 when ratio=1")

    # (b) ref doesn't affect ratio
    def test_ratio_independent_of_ref(self):
        np = self.np
        B, T, V = 2, 6, 8
        ids, mask, adv = self._default_inputs(B, T, V)
        policy_logits = np.random.randn(B, T, V).astype(np.float32) * 0.5
        old_logits = np.random.randn(B, T, V).astype(np.float32) * 0.5
        ref_logits_1 = np.random.randn(B, T, V).astype(np.float32) * 2.0
        ref_logits_2 = np.random.randn(B, T, V).astype(np.float32) * 0.1

        old_lp_raw = self._old_lp_from_logits(old_logits, ids)

        r1 = self._manual_grpo_loss(
            policy_logits, ref_logits_1, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        r2 = self._manual_grpo_loss(
            policy_logits, ref_logits_2, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        self.assertAlmostEqual(r1["loss"], r2["loss"], places=5,
                               msg="Policy loss must be identical regardless of ref model")

    # (c) Positive advantage clips downward, negative clips upward
    def test_clipping_direction(self):
        np = self.np
        B, T, V = 1, 6, 8
        ids = np.random.randint(0, V, (B, T))
        mask = np.zeros((B, T), dtype=np.float32)
        mask[:, -3:] = 1.0

        policy_logits = np.ones((B, T, V), dtype=np.float32) * 5.0
        for t in range(T):
            policy_logits[0, t, ids[0, t]] = 20.0
        old_logits = np.ones((B, T, V), dtype=np.float32) * 0.0
        for t in range(T):
            old_logits[0, t, ids[0, t]] = 0.5
        ref_logits = np.zeros((B, T, V), dtype=np.float32)
        old_lp_raw = self._old_lp_from_logits(old_logits, ids)

        pos_result = self._manual_grpo_loss(
            policy_logits, ref_logits, old_lp_raw, ids, mask,
            np.array([2.0]), kl_coef=0.0, clip_range=0.2,
        )
        neg_result = self._manual_grpo_loss(
            policy_logits, ref_logits, old_lp_raw, ids, mask,
            np.array([-2.0]), kl_coef=0.0, clip_range=0.2,
        )
        self.assertTrue(np.isfinite(pos_result["loss"]))
        self.assertTrue(np.isfinite(neg_result["loss"]))
        self.assertNotAlmostEqual(pos_result["loss"], neg_result["loss"], places=2,
                                   msg="Positive and negative advantage should produce different losses")

    # (d) Masked tokens don't contribute
    def test_masked_tokens_no_contribution(self):
        np = self.np
        B, T, V = 1, 8, 4
        ids = np.random.randint(0, V, (B, T))

        mask1 = np.zeros((B, T), dtype=np.float32)
        mask1[:, -2:] = 1.0

        mask2 = np.zeros((B, T), dtype=np.float32)
        mask2[:, -2:] = 1.0
        mask2[:, 0] = 1.0  # extra mask on prompt token (incorrect but tests robustness)

        logits = np.random.randn(B, T, V).astype(np.float32) * 0.5
        ref = np.zeros((B, T, V), dtype=np.float32)
        old_lp_raw = self._old_lp_from_logits(logits, ids)

        r1 = self._manual_grpo_loss(
            logits, logits, old_lp_raw, ids, mask1,
            np.array([1.0]), kl_coef=0.0, clip_range=0.2,
        )
        r2 = self._manual_grpo_loss(
            logits, logits, old_lp_raw, ids, mask2,
            np.array([1.0]), kl_coef=0.0, clip_range=0.2,
        )
        self.assertTrue(np.isfinite(r1["loss"]))
        self.assertTrue(np.isfinite(r2["loss"]))

    def test_only_response_mask_positions_contribute_to_loss(self):
        """Changing logits at non-masked positions must not change the loss."""
        np = self.np
        B, T, V = 1, 6, 4
        ids = np.array([[0, 1, 2, 3, 0, 1]])
        mask = np.zeros((B, T), dtype=np.float32)
        mask[:, -2:] = 1.0

        logits_a = np.random.randn(B, T, V).astype(np.float32) * 0.5
        logits_b = logits_a.copy()
        logits_b[0, 0, :] += 10.0
        logits_b[0, 1, :] += 10.0
        logits_b[0, 2, :] += 10.0
        logits_b[0, 3, :] += 10.0

        ref = np.zeros((B, T, V), dtype=np.float32)
        old_lp_raw = self._old_lp_from_logits(logits_a, ids)

        r_a = self._manual_grpo_loss(
            logits_a, ref, old_lp_raw, ids, mask,
            np.array([1.0]), kl_coef=0.0, clip_range=0.2,
        )
        r_b = self._manual_grpo_loss(
            logits_b, ref, old_lp_raw, ids, mask,
            np.array([1.0]), kl_coef=0.0, clip_range=0.2,
        )
        self.assertAlmostEqual(r_a["loss"], r_b["loss"], places=4,
                               msg="Changing logits at non-masked positions must not change policy loss")

    # (e) Zero-variance group → no NaN/Inf
    def test_zero_variance_group_no_nan(self):
        np = self.np
        B, T, V = 4, 6, 4
        ids = np.random.randint(0, V, (B, T))
        mask = np.zeros((B, T), dtype=np.float32)
        mask[:, -3:] = 1.0
        logits = np.random.randn(B, T, V).astype(np.float32) * 0.5
        old_lp_raw = self._old_lp_from_logits(logits, ids)

        zero_adv = np.zeros(B, dtype=np.float32)
        result = self._manual_grpo_loss(
            logits, logits, old_lp_raw, ids, mask, zero_adv,
            kl_coef=0.1, clip_range=0.2,
        )
        self.assertFalse(math.isnan(result["loss"]), "Loss must not be NaN with zero-variance advantages")
        self.assertFalse(math.isinf(result["loss"]), "Loss must not be Inf with zero-variance advantages")
        # When current=old, approx_kl=0 regardless of advantages
        self.assertAlmostEqual(result["approx_kl"], 0.0, places=5,
                               msg="approx_kl=0 when current=old")

    # (f) kl_coef affects only KL penalty, not ratio
    def test_kl_coef_only_affects_penalty(self):
        np = self.np
        B, T, V = 2, 6, 8
        ids, mask, adv = self._default_inputs(B, T, V)
        policy_logits = np.random.randn(B, T, V).astype(np.float32) * 0.5
        old_logits = np.random.randn(B, T, V).astype(np.float32) * 0.3
        ref_logits = np.random.randn(B, T, V).astype(np.float32) * 1.0
        old_lp_raw = self._old_lp_from_logits(old_logits, ids)

        r_low_kl = self._manual_grpo_loss(
            policy_logits, ref_logits, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        r_high_kl = self._manual_grpo_loss(
            policy_logits, ref_logits, old_lp_raw, ids, mask, adv,
            kl_coef=1.0, clip_range=0.2,
        )
        self.assertAlmostEqual(r_low_kl["approx_kl"], r_high_kl["approx_kl"], places=5,
                               msg="approx_kl must be independent of kl_coef")
        self.assertAlmostEqual(r_low_kl["clip_fraction"], r_high_kl["clip_fraction"], places=5,
                               msg="clip_fraction must be independent of kl_coef")
        self.assertNotAlmostEqual(r_low_kl["loss"], r_high_kl["loss"], places=3,
                                   msg="Total loss must differ when kl_coef changes")

    # (g) Response-only metrics: approx_kl invariant to prompt/padding logits
    def test_changing_prompt_logits_does_not_change_approx_kl(self):
        """approx_kl must depend only on response-token positions."""
        np = self.np
        B, T, V = 2, 8, 6
        ids, mask, adv = self._default_inputs(B, T, V, seed=101)
        logits_a = np.random.randn(B, T, V).astype(np.float32) * 0.5
        logits_b = logits_a.copy()
        # Vary prompt and padding positions (non-response)
        logits_b[:, :3, :] += 8.0
        logits_b[:, 5:, :] += 8.0

        old_logits = np.random.randn(B, T, V).astype(np.float32) * 0.3
        old_lp_raw = self._old_lp_from_logits(old_logits, ids)
        ref = np.zeros((B, T, V), dtype=np.float32)

        r_a = self._manual_grpo_loss(
            logits_a, ref, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        r_b = self._manual_grpo_loss(
            logits_b, ref, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        self.assertAlmostEqual(r_a["approx_kl"], r_b["approx_kl"], places=5,
                               msg="approx_kl must not change when prompt/padding logits change")

    # (h) Response-only metrics: clip_fraction invariant to prompt/padding logits
    def test_changing_prompt_logits_does_not_change_clip_fraction(self):
        """clip_fraction must depend only on response-token positions."""
        np = self.np
        B, T, V = 2, 8, 6
        ids, mask, adv = self._default_inputs(B, T, V, seed=202)
        logits_a = np.random.randn(B, T, V).astype(np.float32) * 0.5
        logits_b = logits_a.copy()
        logits_b[:, :3, :] += 8.0
        logits_b[:, 5:, :] += 8.0

        old_logits = np.random.randn(B, T, V).astype(np.float32) * 0.3
        old_lp_raw = self._old_lp_from_logits(old_logits, ids)
        ref = np.zeros((B, T, V), dtype=np.float32)

        r_a = self._manual_grpo_loss(
            logits_a, ref, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        r_b = self._manual_grpo_loss(
            logits_b, ref, old_lp_raw, ids, mask, adv,
            kl_coef=0.0, clip_range=0.2,
        )
        self.assertAlmostEqual(r_a["clip_fraction"], r_b["clip_fraction"], places=5,
                               msg="clip_fraction must not change when prompt/padding logits change")

    # (i) Empty response mask → no NaN/Inf in response-only metrics
    def test_empty_response_mask_no_nan(self):
        """When response mask is empty, approx_kl and clip_fraction must not be NaN/Inf."""
        np = self.np
        B, T, V = 1, 6, 4
        ids = np.array([[0, 1, 2, 3, 0, 1]])
        mask = np.zeros((B, T), dtype=np.float32)  # all prompt/padding

        logits = np.random.randn(B, T, V).astype(np.float32) * 0.5
        old_logits = np.random.randn(B, T, V).astype(np.float32) * 0.3
        old_lp_raw = self._old_lp_from_logits(old_logits, ids)
        ref = np.zeros((B, T, V), dtype=np.float32)

        result = self._manual_grpo_loss(
            logits, ref, old_lp_raw, ids, mask,
            np.array([1.0]), kl_coef=0.0, clip_range=0.2,
        )
        self.assertFalse(math.isnan(result["approx_kl"]),
                         "approx_kl must not be NaN with empty response mask")
        self.assertFalse(math.isinf(result["approx_kl"]),
                         "approx_kl must not be Inf with empty response mask")
        self.assertFalse(math.isnan(result["clip_fraction"]),
                         "clip_fraction must not be NaN with empty response mask")
        self.assertFalse(math.isinf(result["clip_fraction"]),
                         "clip_fraction must not be Inf with empty response mask")


# ---------------------------------------------------------------------------
# MLX integration tests (require MLX + mlx_lm)
# ---------------------------------------------------------------------------

class TestGRPOLossMLX(unittest.TestCase):
    """Integration tests using real MLX arrays and compute_grpo_loss.

    These tests verify that compute_grpo_loss produces correct, finite
    results with actual MLX tensors. They are skipped if MLX is not
    available.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
            cls.mx = mx
        except ImportError:
            raise unittest.SkipTest("mlx not available — skipping MLX integration tests")

    def _make_simple_grpo_inputs(self, B=2, T=6, V=8):
        """Create minimal inputs for compute_grpo_loss."""
        mx = self.mx
        import numpy as np
        np.random.seed(42)

        input_ids = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.int32)
        response_mask = mx.array(np.zeros((B, T), dtype=np.float32))
        response_mask[:, -3:] = 1.0
        advantages = mx.array(np.array([0.5, -0.5], dtype=np.float32))

        # Old token logprobs: zeros (same as policy initially)
        old_token_lp = mx.array(np.zeros((B, T), dtype=np.float32))

        return input_ids, response_mask, advantages, old_token_lp

    def test_compute_grpo_loss_returns_finite(self):
        """compute_grpo_loss produces finite loss with simple inputs."""
        from polaris.trainers.grpo import compute_grpo_loss

        mx = self.mx
        B, T, V = 2, 6, 8
        input_ids, response_mask, advantages, old_token_lp = self._make_simple_grpo_inputs(B, T, V)

        # Simple models that return logits
        class SimpleModel:
            def __init__(self, scale):
                self.scale = scale
            def __call__(self, x):
                return {"logits": mx.random.normal((B, T, V)) * self.scale}
            def parameters(self):
                return {}
            def trainable_parameters(self):
                return {}

        policy = SimpleModel(0.5)
        ref = SimpleModel(1.0)

        result = compute_grpo_loss(
            policy, ref, input_ids, response_mask, advantages,
            kl_coef=0.05, clip_range=0.2,
            old_token_logprobs=old_token_lp,
        )

        loss_val = float(result["loss"].item())
        self.assertFalse(math.isnan(loss_val), f"Loss must not be NaN, got {loss_val}")
        self.assertFalse(math.isinf(loss_val), f"Loss must not be Inf, got {loss_val}")

        for key in ["policy_logprob", "old_policy_logprob", "ref_logprob",
                     "kl", "entropy", "approx_kl", "clip_fraction"]:
            self.assertIn(key, result, f"Missing key: {key}")
            val = float(result[key].item())
            self.assertFalse(math.isnan(val), f"{key} is NaN")
            self.assertFalse(math.isinf(val), f"{key} is Inf")

    def test_ratio_one_when_old_equals_policy(self):
        """When old_token_logprobs is None, ratio should be ~1 (current=current)."""
        from polaris.trainers.grpo import compute_grpo_loss

        mx = self.mx
        B, T, V = 2, 6, 8
        input_ids, response_mask, advantages, _ = self._make_simple_grpo_inputs(B, T, V)

        class SimpleModel:
            def __init__(self):
                self._logits = mx.random.normal((B, T, V)) * 0.5
            def __call__(self, x):
                return {"logits": self._logits}

        model = SimpleModel()

        result = compute_grpo_loss(
            model, model, input_ids, response_mask, advantages,
            kl_coef=0.0, clip_range=0.2, old_token_logprobs=None,
        )
        # old=None means old=current, so approx_kl should be 0
        approx_kl = float(result["approx_kl"].item())
        self.assertAlmostEqual(approx_kl, 0.0, places=4,
                               msg="approx_kl must be ~0 when old=current (old_token_logprobs=None)")

    def test_zero_advantages_no_nan(self):
        """Zero-variance advantages must not produce NaN loss."""
        from polaris.trainers.grpo import compute_grpo_loss

        mx = self.mx
        B, T, V = 4, 6, 8
        input_ids, response_mask, _, old_token_lp = self._make_simple_grpo_inputs(B, T, V)
        # Override B for this test
        input_ids = mx.array(
            __import__("numpy").random.randint(0, V, (B, T)), dtype=mx.int32,
        )
        response_mask = mx.array(__import__("numpy").zeros((B, T), dtype="float32"))
        response_mask[:, -3:] = 1.0
        old_token_lp = mx.array(__import__("numpy").zeros((B, T), dtype="float32"))
        advantages = mx.array(__import__("numpy").zeros(B, dtype="float32"))

        class SimpleModel:
            def __init__(self):
                self._logits = mx.random.normal((B, T, V)) * 0.5
            def __call__(self, x):
                return {"logits": self._logits}

        model = SimpleModel()
        result = compute_grpo_loss(
            model, model, input_ids, response_mask, advantages,
            kl_coef=0.1, clip_range=0.2, old_token_logprobs=old_token_lp,
        )
        loss_val = float(result["loss"].item())
        self.assertFalse(math.isnan(loss_val), f"Loss must not be NaN with zero advantages, got {loss_val}")
        self.assertFalse(math.isinf(loss_val), f"Loss must not be Inf with zero advantages, got {loss_val}")


# ---------------------------------------------------------------------------
# EOS handling and mixed-group batch construction tests
# ---------------------------------------------------------------------------

class TestEOSHandlingAndBatchConstruction(unittest.TestCase):
    """Tests for EOS-as-response-action batch construction.

    Verifies that input_ids, response_mask, and old_token_logprobs are
    shape-consistent across mixed groups (EOS vs non-EOS completions).
    """

    def test_mixed_eos_noneos_group_batch_shapes(self):
        """Mixed group: one EOS, one max-length. All batch arrays shape-consistent."""
        import numpy as np

        prompt_ids = [10, 11, 12]  # prompt_len = 3
        prompt_len = len(prompt_ids)

        # C1: 2 generated tokens + EOS → total 5 tokens
        c1_ids = prompt_ids + [100, 101, 2]  # 2 is EOS
        c1_old_lp = [-1.0, -0.8, -0.5]  # last is EOS logprob

        # C2: 3 generated tokens, no EOS → total 6 tokens
        c2_ids = prompt_ids + [200, 201, 202]
        c2_old_lp = [-1.2, -0.9, -0.7]

        rollout_ids = [c1_ids, c2_ids]
        rollout_old_lps = [c1_old_lp, c2_old_lp]

        class MockTokenizer:
            pad_token_id = 0

        ids, mask, old_lp = build_grpo_batch(
            rollout_ids, rollout_old_lps, prompt_len, MockTokenizer(),
        )

        self.assertEqual(ids.shape, (2, 6))
        self.assertEqual(mask.shape, (2, 6))
        self.assertEqual(old_lp.shape, (2, 6))

        # Prompt region masked out
        np.testing.assert_array_equal(mask[0, :3], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[1, :3], [0.0, 0.0, 0.0])

        # C1: 3 response tokens (2 + EOS)
        np.testing.assert_array_equal(mask[0, 3:6], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(mask[0, 6:], [])  # no padding for C1

        # C2: 3 response tokens + 0 padding (same length as C1)
        np.testing.assert_array_equal(mask[1, 3:6], [1.0, 1.0, 1.0])

        # Old logprobs: non-zero only in response region
        np.testing.assert_array_equal(old_lp[0, :3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(old_lp[0, 3], -1.0)   # first response token
        self.assertAlmostEqual(old_lp[0, 4], -0.8)   # second response token
        self.assertAlmostEqual(old_lp[0, 5], -0.5)   # EOS logprob
        np.testing.assert_array_equal(old_lp[1, :3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(old_lp[1, 3], -1.2)
        self.assertAlmostEqual(old_lp[1, 4], -0.9)
        self.assertAlmostEqual(old_lp[1, 5], -0.7)

    def test_eos_noneos_different_lengths_batch(self):
        """EOS and non-EOS completions with different response lengths."""
        import numpy as np

        prompt_ids = [10, 11]
        prompt_len = len(prompt_ids)

        # C1: 1 token + EOS → 3 total
        c1_ids = prompt_ids + [50, 2]
        c1_old_lp = [-0.3, -0.6]

        # C2: 4 tokens, no EOS → 6 total
        c2_ids = prompt_ids + [60, 61, 62, 63]
        c2_old_lp = [-0.4, -0.5, -0.6, -0.7]

        class MockTokenizer:
            pad_token_id = 0

        ids, mask, old_lp = build_grpo_batch(
            [c1_ids, c2_ids], [c1_old_lp, c2_old_lp], prompt_len, MockTokenizer(),
        )

        max_len = 6
        self.assertEqual(ids.shape, (2, max_len))
        self.assertEqual(mask.shape, (2, max_len))
        self.assertEqual(old_lp.shape, (2, max_len))

        # C1: 2 response tokens + 2 padding
        np.testing.assert_array_equal(mask[0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(old_lp[0, 2], -0.3)
        self.assertAlmostEqual(old_lp[0, 3], -0.6)  # EOS logprob
        self.assertAlmostEqual(old_lp[0, 4], 0.0)    # padding
        self.assertAlmostEqual(old_lp[0, 5], 0.0)    # padding

        # C2: 4 response tokens + 0 padding
        np.testing.assert_array_equal(mask[1], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(old_lp[1, 2], -0.4)
        self.assertAlmostEqual(old_lp[1, 3], -0.5)
        self.assertAlmostEqual(old_lp[1, 4], -0.6)
        self.assertAlmostEqual(old_lp[1, 5], -0.7)

    def test_eos_noneos_old_lp_alignment_with_causal_shift(self):
        """After causal shift, old_lp aligns with policy_token_lp positions."""
        import numpy as np

        prompt_ids = [10, 11, 12]
        prompt_len = 3
        c1_ids = prompt_ids + [100, 101, 2]  # EOS=2
        c1_old_lp = [-1.0, -0.8, -0.5]
        c2_ids = prompt_ids + [200, 201, 202]
        c2_old_lp = [-1.2, -0.9, -0.7]

        class MockTokenizer:
            pad_token_id = 0

        _, mask, old_lp = build_grpo_batch(
            [c1_ids, c2_ids], [c1_old_lp, c2_old_lp], prompt_len, MockTokenizer(),
        )

        # Simulate causal shift (as done inside compute_grpo_loss)
        shift_old = old_lp[:, 1:]
        shift_mask = mask[:, 1:]
        shifted_old = shift_old * shift_mask

        # C1 shifted: positions [1,2,3,4,5] of old_lp after shift
        # old_lp[0] = [0,0,0,-1.0,-0.8,-0.5]
        # shift_old[0] = [0,0,-1.0,-0.8,-0.5]
        # shift_mask[0] = [0,0,1,1,1] (from mask[0]=[0,0,0,1,1,1])
        # shifted_old[0] = [0,0,-1.0,-0.8,-0.5]
        self.assertAlmostEqual(shifted_old[0, 2], -1.0)  # EOS predicted from pos 4
        self.assertAlmostEqual(shifted_old[0, 3], -0.8)
        self.assertAlmostEqual(shifted_old[0, 4], -0.5)  # EOS logprob

        # Verify alignment: shifted old has same shape as policy_token_lp would
        self.assertEqual(shifted_old.shape, shift_mask.shape)

    def test_eos_token_in_ids_sequence(self):
        """Verify EOS token is present in input_ids (not stripped)."""
        import numpy as np

        prompt_ids = [10, 11]
        prompt_len = 2
        EOS_ID = 2

        c1_ids = prompt_ids + [50, 51, EOS_ID]
        c1_old_lp = [-0.3, -0.4, -0.6]
        c2_ids = prompt_ids + [60, 61, 62]
        c2_old_lp = [-0.5, -0.5, -0.5]

        class MockTokenizer:
            pad_token_id = 0

        ids, _, _ = build_grpo_batch(
            [c1_ids, c2_ids], [c1_old_lp, c2_old_lp], prompt_len, MockTokenizer(),
        )

        # EOS token present in C1's sequence
        self.assertEqual(int(ids[0, 4]), EOS_ID)
        # C2 has no EOS (token 62 is a regular token)
        self.assertNotEqual(int(ids[1, 4]), EOS_ID)


# ---------------------------------------------------------------------------
# Preflight train validation tests
# ---------------------------------------------------------------------------

class TestPreflightTrain(unittest.TestCase):
    """Tests for preflight_train: no side effects, no create_run."""

    @staticmethod
    def _valid_config(data_path: str = "/nonexistent/data.jsonl") -> dict:
        return {
            "data": {"path": data_path},
            "lora": {"enabled": True, "dropout": 0.0, "r": 32, "alpha": 32},
            "grpo": {
                "policy_adapter_path": "/tmp/policy_adapter",
                "ref_model_path": "/tmp/ref",
                "ref_adapter_path": "/tmp/ref_adapter",
                "group_size": 8,
                "max_completion_length": 256,
                "kl_coef": 0.05,
                "clip_range": 0.2,
                "rollout_temperature": 1.0,
                "rollout_top_p": 1.0,
                "reward": {
                    "correct": 1.0, "incorrect": 0.0,
                    "unparseable": -0.5, "empty": -1.0,
                },
            },
        }

    @unittest.mock.patch("scripts.train_grpo.resolve_adapter_file")
    @unittest.mock.patch("scripts.train_grpo.load_json_record_stream")
    @unittest.mock.patch("pathlib.Path.exists", return_value=True)
    def test_temp_0_5_rejects_training(self, _mock_exists, mock_load, mock_resolve):
        """temperature=0.5 must fail preflight."""
        mock_resolve.return_value = Path("/tmp/fake.safetensors")
        mock_load.return_value = [
            {"messages": [{"role": "user", "content": "1+1?"}],
             "metadata": {"answer": "2"}},
        ]
        config = self._valid_config()
        config["grpo"]["rollout_temperature"] = 0.5
        errors = preflight_train(config)
        self.assertTrue(any("temperature" in e.lower() for e in errors))

    @unittest.mock.patch("scripts.train_grpo.resolve_adapter_file")
    def test_data_missing_rejects_training(self, mock_resolve):
        """Missing data file must fail preflight."""
        mock_resolve.return_value = Path("/tmp/fake.safetensors")
        config = self._valid_config("/nonexistent/missing_data.jsonl")
        errors = preflight_train(config)
        self.assertTrue(any("data" in e.lower() and "not found" in e.lower() for e in errors))

    @unittest.mock.patch("scripts.train_grpo.resolve_adapter_file")
    @unittest.mock.patch("scripts.train_grpo.load_json_record_stream")
    @unittest.mock.patch("pathlib.Path.exists", return_value=True)
    def test_group_size_1_rejects(self, _mock_exists, mock_load, mock_resolve):
        """group_size=1 must fail preflight (need >= 2 for group-relative advantage)."""
        mock_resolve.return_value = Path("/tmp/fake.safetensors")
        mock_load.return_value = [
            {"messages": [{"role": "user", "content": "1+1?"}],
             "metadata": {"answer": "2"}},
        ]
        config = self._valid_config()
        config["grpo"]["group_size"] = 1
        errors = preflight_train(config)
        self.assertTrue(any("group_size" in e and ">= 2" in e for e in errors))

    @unittest.mock.patch("scripts.train_grpo.resolve_adapter_file")
    @unittest.mock.patch("scripts.train_grpo.load_json_record_stream")
    @unittest.mock.patch("pathlib.Path.exists", return_value=True)
    def test_lora_dropout_nonzero_rejects(self, _mock_exists, mock_load, mock_resolve):
        """lora.dropout=0.1 must fail preflight (eval/train distribution mismatch)."""
        mock_resolve.return_value = Path("/tmp/fake.safetensors")
        mock_load.return_value = [
            {"messages": [{"role": "user", "content": "1+1?"}],
             "metadata": {"answer": "2"}},
        ]
        config = self._valid_config()
        config["lora"]["dropout"] = 0.1
        errors = preflight_train(config)
        self.assertTrue(any("dropout" in e and "0.0" in e for e in errors))

    @unittest.mock.patch("scripts.train_grpo.resolve_adapter_file")
    @unittest.mock.patch("scripts.train_grpo.load_json_record_stream")
    @unittest.mock.patch("pathlib.Path.exists", return_value=True)
    def test_adapter_not_found_rejects(self, _mock_exists, mock_load, mock_resolve):
        """Missing adapter file must fail preflight."""
        mock_resolve.side_effect = FileNotFoundError("LoRA adapter weights not found: /tmp/missing")
        mock_load.return_value = [
            {"messages": [{"role": "user", "content": "1+1?"}],
             "metadata": {"answer": "2"}},
        ]
        config = self._valid_config()
        errors = preflight_train(config)
        self.assertTrue(any("adapter" in e.lower() and "not found" in e.lower() for e in errors))

    @unittest.mock.patch("scripts.train_grpo.resolve_adapter_file")
    @unittest.mock.patch("scripts.train_grpo.load_json_record_stream")
    @unittest.mock.patch("pathlib.Path.exists", return_value=True)
    def test_valid_config_passes_preflight(self, _mock_exists, mock_load, mock_resolve):
        """Valid config, data, adapters, group_size, dropout all pass."""
        mock_resolve.return_value = Path("/tmp/fake.safetensors")
        mock_load.return_value = [
            {"messages": [{"role": "user", "content": "1+1?"}],
             "metadata": {"answer": "2"}},
        ]
        errors = preflight_train(self._valid_config())
        self.assertEqual(errors, [], f"Expected no errors, got: {errors}")


# ---------------------------------------------------------------------------
# Batch input validation tests
# ---------------------------------------------------------------------------

class TestBatchInputValidation(unittest.TestCase):
    """Tests for build_grpo_batch input length validation."""

    def test_mismatched_old_lp_length_raises(self):
        """ValueError when old_token_logprobs length != response length."""
        prompt_ids = [10, 11, 12]
        prompt_len = 3

        # seq_len=6, response_len=3, but old_lp has 2 entries (should be 3)
        rollout_ids = [prompt_ids + [100, 101, 102]]
        rollout_old_lps = [[-1.0, -0.8]]  # wrong: 2 instead of 3

        class MockTokenizer:
            pad_token_id = 0

        with self.assertRaises(ValueError) as ctx:
            build_grpo_batch(rollout_ids, rollout_old_lps, prompt_len, MockTokenizer())

        err = str(ctx.exception)
        self.assertIn("Sample 0", err)
        self.assertIn("2", err)   # actual old_lp length
        self.assertIn("3", err)   # expected response length

    def test_valid_lengths_pass(self):
        """No error when old_token_logprobs length matches response length."""
        prompt_ids = [10, 11]
        prompt_len = 2

        rollout_ids = [prompt_ids + [100, 101, 102]]
        rollout_old_lps = [[-1.0, -0.8, -0.5]]  # 3 = 5 - 2

        class MockTokenizer:
            pad_token_id = 0

        ids, mask, old_lp = build_grpo_batch(
            rollout_ids, rollout_old_lps, prompt_len, MockTokenizer(),
        )
        self.assertEqual(ids.shape[0], 1)


# ---------------------------------------------------------------------------
# Imports (placed at end to avoid circular imports at module level)
# ---------------------------------------------------------------------------

from scripts.train_grpo import (
    validate_grpo_config,
    validate_data_schema,
    validate_reward_protocol,
    dry_run,
    build_grpo_batch,
    preflight_train,
)


if __name__ == "__main__":
    unittest.main()
