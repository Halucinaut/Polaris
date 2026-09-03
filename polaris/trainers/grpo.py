"""GRPO trainer core logic.

Implements the correct GRPO/PPO clipped policy objective with
group-relative advantage and reference KL penalty.

Key invariants:
- Ratio is current_policy / OLD_policy (frozen at rollout time).
- Reference model computes ONLY response-level KL penalty;
  it never enters the ratio.
- Per-token logprobs with causal shift: logits[t] predicts ids[t+1].
- Response mask aligns with input_ids (pre-shift).

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
    old_policy_logprob: float
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


# ---------------------------------------------------------------------------
# Token log-probability helpers
# ---------------------------------------------------------------------------

def _compute_token_logprobs(model_or_logits, ids, mask):
    """Compute per-token log π(token|context) with causal shift.

    Accepts either:
    - a pre-computed logits tensor, or
    - a dict with key 'logits' (as returned by MLX modules).

    Causal alignment: logprob[t] = log π(ids[t] | ids[:t])
    implemented as: logits[:, :-1, :] predicts ids[:, 1:].
    Mask[:, 1:] selects positions where the token is a response token.

    Returns:
        (B, T) array: log-probability for each response-token position.
        Positions outside the mask are set to 0.0 (ignored by downstream sums).
    """
    # Resolve logits from model output
    if isinstance(model_or_logits, dict):
        logits = model_or_logits["logits"]
    else:
        logits = model_or_logits

    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_ids = ids[:, 1:]             # (B, T-1)
    shift_mask = mask[:, 1:]           # (B, T-1)

    B, T_m1, V = shift_logits.shape
    flat_logits = shift_logits.reshape(B * T_m1, V)
    flat_ids = shift_ids.reshape(B * T_m1)

    import mlx.nn.losses as losses
    token_nll = losses.cross_entropy(flat_logits, flat_ids)  # (B*T_m1,)
    token_nll = token_nll.reshape(B, T_m1)

    token_lp = -token_nll * shift_mask  # zero out non-response positions
    return token_lp


def compute_response_logprob(model_or_logits, ids, mask):
    """Per-sample summed log-probability over response tokens.

    Returns (B,) array of summed response log-probabilities.
    """
    token_lp = _compute_token_logprobs(model_or_logits, ids, mask)
    return token_lp.sum(axis=1)


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    policy_model,
    ref_model,
    input_ids,
    response_mask,
    advantages,
    kl_coef: float,
    clip_range: float,
    old_token_logprobs=None,
) -> dict:
    """Compute GRPO clipped policy objective with reference KL penalty.

    Args:
        policy_model: Current (trainable) policy model.
        ref_model:    Frozen reference model.
        input_ids:    (B, T) token ids for prompt + completion.
        response_mask: (B, T) float mask — 1.0 for completion token TARGETS.
                      (aligned with input_ids; causal shift applied internally)
        advantages:   (B,) pre-computed group-normalized advantages.
        kl_coef:      KL penalty coefficient.
        clip_range:   PPO-style clip epsilon.
        old_token_logprobs: (B, T) per-token log-probs from rollout policy.
            When None, falls back to using policy_model(input_ids) as both
            current and old (correct only when current == old, i.e., before
            any gradient update).

    Returns:
        Dict with: loss, policy_logprob, old_policy_logprob, ref_logprob,
        kl, entropy, approx_kl, clip_fraction.
    """
    import mlx.core as mx

    # ---- Forward passes ----
    policy_out = policy_model(input_ids)
    ref_out = ref_model(input_ids)

    policy_logits = policy_out["logits"] if isinstance(policy_out, dict) else policy_out
    ref_logits = ref_out["logits"] if isinstance(ref_out, dict) else ref_out

    # ---- Per-token log-probs (causal shift applied inside) ----
    policy_token_lp = _compute_token_logprobs(policy_logits, input_ids, response_mask)
    ref_token_lp = _compute_token_logprobs(ref_logits, input_ids, response_mask)

    # ---- Shifted response mask (used throughout) ----
    shift_mask = response_mask[:, 1:]

    if old_token_logprobs is not None:
        # old_token_logprobs: (B, T) raw per-token logprobs from rollout;
        # apply same causal shift and mask as policy/ref so positions align.
        old_token_lp = old_token_logprobs[:, 1:] * shift_mask
    else:
        # Before any gradient update, old == current
        old_token_lp = policy_token_lp

    # ---- Response logprobs (summed, per-sample) ----
    policy_lp = policy_token_lp.sum(axis=1)
    old_lp = old_token_lp.sum(axis=1)
    ref_lp = ref_token_lp.sum(axis=1)

    # ---- Per-token ratio: π_current / π_old ----
    log_ratio = policy_token_lp - old_token_lp
    ratio = mx.exp(log_ratio)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # ---- Policy objective (length-normalized over response tokens) ----
    adv_broad = advantages[:, None]       # (B, 1)
    surrogate1 = ratio * adv_broad
    surrogate2 = clipped_ratio * adv_broad
    per_token_loss = -mx.minimum(surrogate1, surrogate2) * shift_mask

    valid_token_count = mx.maximum(shift_mask.sum(), 1.0)
    policy_loss = per_token_loss.sum() / valid_token_count

    # ---- Reference KL penalty (forward KL: Σ π_current · (log π_current - log π_ref)) ----
    shift_policy_logits = policy_logits[:, :-1, :]
    shift_ref_logits = ref_logits[:, :-1, :]
    policy_log_softmax = shift_policy_logits - mx.logsumexp(
        shift_policy_logits, axis=-1, keepdims=True,
    )
    ref_log_softmax = shift_ref_logits - mx.logsumexp(
        shift_ref_logits, axis=-1, keepdims=True,
    )
    policy_probs = mx.exp(policy_log_softmax)
    per_token_kl = (policy_probs * (policy_log_softmax - ref_log_softmax)).sum(axis=-1)
    kl = (per_token_kl * shift_mask).sum(axis=1) / mx.maximum(shift_mask.sum(axis=1), 1.0)
    mean_kl = kl.mean()

    # ---- Total loss ----
    loss = policy_loss + kl_coef * mean_kl

    # ---- Entropy (from current policy logits, response positions only) ----
    entropy_per_token = -(policy_probs * policy_log_softmax).sum(axis=-1)
    mean_entropy = (entropy_per_token * shift_mask).sum() / mx.maximum(shift_mask.sum(), 1.0)

    # ---- Approximate KL for logging (Schulman blog: (r-1) - log r) ----
    approx_kl = ((ratio - 1) - log_ratio) * shift_mask
    approx_kl = approx_kl.sum() / mx.maximum(shift_mask.sum(), 1.0)

    # ---- Clip fraction (response-only) ----
    clip_frac = (mx.abs(ratio - 1.0) > clip_range).astype(mx.float32) * shift_mask
    clip_frac = clip_frac.sum() / mx.maximum(shift_mask.sum(), 1.0)

    return {
        "loss": loss,
        "policy_logprob": policy_lp.mean(),
        "old_policy_logprob": old_lp.mean(),
        "ref_logprob": ref_lp.mean(),
        "kl": mean_kl,
        "entropy": mean_entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
    }
