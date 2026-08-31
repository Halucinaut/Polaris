"""GRPO trainer core logic.

Implements the clipped policy objective with group-relative advantage
and KL penalty, operating on response-only token log-probabilities
with proper causal shift alignment.

This module contains pure computation; the training loop, rollout,
and I/O live in scripts/train_grpo.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GRPOStepMetrics:
    """Metrics produced by one GRPO training step."""
    loss: float
    policy_logprob: float
    ref_logprob: float
    kl: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    reward_mean: float
    reward_std: float
    zero_variance_group_count: int
    valid_advantage_count: int
    total_advantage_count: int
    avg_completion_length: float


def compute_response_logprob(logits, ids, mask):
    """Compute per-sample response log-probability with next-token alignment.

    Uses the same causal-shift convention as train_dpo.py:
    logits[:, :-1] predicts ids[:, 1:]; mask[:, 1:] selects response positions.

    Args:
        logits: (B, T, V) raw model output.
        ids:    (B, T) token ids.
        mask:   (B, T) float mask — 1.0 for response token *targets*.

    Returns:
        (B,) per-sample summed log-probability over response tokens.
    """
    import mlx.nn.losses as losses

    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_ids = ids[:, 1:]             # (B, T-1)
    shift_mask = mask[:, 1:]           # (B, T-1)

    B, T, V = shift_logits.shape
    flat_logits = shift_logits.reshape(B * T, V)
    flat_ids = shift_ids.reshape(B * T)

    token_nll = losses.cross_entropy(flat_logits, flat_ids)  # (B*T,)
    token_nll = token_nll.reshape(B, T)

    masked_nll = token_nll * shift_mask
    sample_logprob = -masked_nll.sum(axis=1)
    return sample_logprob


def compute_grpo_loss(
    policy_model,
    ref_model,
    input_ids,
    response_mask,
    advantages,
    kl_coef: float,
    clip_range: float,
) -> dict:
    """Compute GRPO clipped policy objective with KL penalty.

    Args:
        policy_model: Current (trainable) policy model.
        ref_model:    Frozen reference model.
        input_ids:    (B, T) token ids for prompt + completion.
        response_mask: (B, T) float mask — 1.0 for completion tokens.
        advantages:   (B,) pre-computed group-normalized advantages.
        kl_coef:      KL penalty coefficient.
        clip_range:   PPO-style clip epsilon.

    Returns:
        Dict with loss, policy_logprob, ref_logprob, kl, entropy,
        approx_kl, clip_fraction.
    """
    import mlx.core as mx
    import mlx.nn.losses as losses

    # Forward passes
    policy_logits = policy_model(input_ids)
    ref_logits = ref_model(input_ids)

    # Response log-probs (causal shift applied inside)
    policy_lp = compute_response_logprob(policy_logits, input_ids, response_mask)
    ref_lp = compute_response_logprob(ref_logits, input_ids, response_mask)

    # Per-token log-probs for KL computation (shifted)
    shift_policy_logits = policy_logits[:, :-1, :]
    shift_ref_logits = ref_logits[:, :-1, :]
    shift_mask = response_mask[:, 1:]

    # log_softmax for per-token KL
    policy_log_softmax = shift_policy_logits - mx.logsumexp(shift_policy_logits, axis=-1, keepdims=True)
    ref_log_softmax = shift_ref_logits - mx.logsumexp(shift_ref_logits, axis=-1, keepdims=True)

    # Per-token KL: sum over vocab of pi * (log_pi - log_ref)
    policy_probs = mx.exp(policy_log_softmax)
    per_token_kl = (policy_probs * (policy_log_softmax - ref_log_softmax)).sum(axis=-1)
    kl = (per_token_kl * shift_mask).sum(axis=1) / mx.maximum(shift_mask.sum(axis=1), 1.0)
    mean_kl = kl.mean()

    # Ratio for clipped objective
    log_ratio = policy_lp - ref_lp
    ratio = mx.exp(log_ratio)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

    advantages_arr = advantages
    policy_loss = -mx.minimum(ratio * advantages_arr, clipped_ratio * advantages_arr).mean()

    # KL penalty
    loss = policy_loss + kl_coef * mean_kl

    # Entropy (approximate from policy logits, response tokens only)
    # H = -sum p * log_p over vocab, averaged over response positions
    entropy_per_token = -(mx.exp(policy_log_softmax) * policy_log_softmax).sum(axis=-1)
    mean_entropy = (entropy_per_token * shift_mask).sum() / mx.maximum(shift_mask.sum(), 1.0)

    # Approximate KL for logging (Schulman blog approx)
    approx_kl = ((ratio - 1) - log_ratio).mean()

    # Clip fraction
    clip_frac = ((mx.abs(ratio - 1.0) > clip_range).astype(mx.float32)).mean()

    return {
        "loss": loss,
        "policy_logprob": policy_lp.mean(),
        "ref_logprob": ref_lp.mean(),
        "kl": mean_kl,
        "entropy": mean_entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
    }
