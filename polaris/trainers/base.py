"""Shared trainer utilities for Polaris: group-relative advantage and helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GroupStats:
    """Statistics for one prompt's rollout group."""
    group_size: int
    reward_mean: float
    reward_std: float
    zero_variance: bool
    advantages: list[float]


def compute_group_relative_advantage(
    rewards: list[float],
) -> GroupStats:
    """Compute normalized advantages within a rollout group.

    Standardizes rewards by subtracting the group mean and dividing by
    the group standard deviation.  When all rewards are identical (zero
    variance), every advantage is set to 0.0 and *zero_variance* is True.
    This prevents NaN/Inf from propagating into the policy gradient.

    Args:
        rewards: Per-completion rewards for a single prompt.

    Returns:
        GroupStats with mean, std, zero_variance flag, and per-completion
        advantages.

    Raises:
        ValueError: If *rewards* is empty.
    """
    n = len(rewards)
    if n == 0:
        raise ValueError("Cannot compute advantage for an empty group")

    mean = sum(rewards) / n
    if n == 1:
        return GroupStats(
            group_size=1,
            reward_mean=mean,
            reward_std=0.0,
            zero_variance=True,
            advantages=[0.0],
        )

    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)

    if std < 1e-8:
        return GroupStats(
            group_size=n,
            reward_mean=mean,
            reward_std=std,
            zero_variance=True,
            advantages=[0.0] * n,
        )

    advantages = [(r - mean) / std for r in rewards]
    return GroupStats(
        group_size=n,
        reward_mean=mean,
        reward_std=std,
        zero_variance=False,
        advantages=advantages,
    )
