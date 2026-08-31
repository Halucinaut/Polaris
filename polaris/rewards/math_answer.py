"""Math answer reward for GRPO training.

Reuses the same answer extraction and comparison logic from eval_math.py
to ensure training and evaluation share identical parsing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Import the canonical extraction/comparison from eval_math.
# This ensures training reward and eval metrics always agree.
from scripts.eval_math import (
    answers_match,
    extract_predicted_answer,
    has_m1_format_adherence,
)


@dataclass
class RewardResult:
    """Structured reward output for one rollout completion."""
    reward: float
    answer_correct: Optional[bool]  # None if extraction failed
    extracted_answer: Optional[str]
    extraction_method: str
    format_adherent: bool
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    invalid_reason: Optional[str] = None


def compute_math_reward(
    completion: str,
    reference_answer: str,
    reward_config: dict,
) -> RewardResult:
    """Compute reward for a single rollout completion against a reference answer.

    Args:
        completion: Raw model output (may contain <think> blocks).
        reference_answer: Ground-truth answer string.
        reward_config: Dict with keys: correct, incorrect, unparseable, empty,
                       format_bonus.

    Returns:
        RewardResult with structured breakdown.
    """
    correct_reward = reward_config.get("correct", 1.0)
    incorrect_reward = reward_config.get("incorrect", 0.0)
    unparseable_reward = reward_config.get("unparseable", -0.5)
    empty_reward = reward_config.get("empty", -1.0)
    format_bonus = reward_config.get("format_bonus", 0.05)

    # Handle empty completion
    if not completion or not completion.strip():
        return RewardResult(
            reward=empty_reward,
            answer_correct=None,
            extracted_answer=None,
            extraction_method="none",
            format_adherent=False,
            reward_breakdown={"base": empty_reward, "format_bonus": 0.0},
            invalid_reason="empty_output",
        )

    # Extract answer using the same logic as eval
    extracted, method = extract_predicted_answer(completion)

    # Check format adherence
    format_ok = has_m1_format_adherence(completion)

    # Build breakdown
    breakdown: dict[str, float] = {}

    if extracted is None:
        # Unparseable: output exists but no answer could be extracted
        base = unparseable_reward
        breakdown["base"] = base
        breakdown["format_bonus"] = format_bonus if format_ok else 0.0
        return RewardResult(
            reward=base + (format_bonus if format_ok else 0.0),
            answer_correct=None,
            extracted_answer=None,
            extraction_method=method,
            format_adherent=format_ok,
            reward_breakdown=breakdown,
            invalid_reason="unparseable",
        )

    # Answer was extracted — check correctness
    is_correct = answers_match(extracted, reference_answer)
    base = correct_reward if is_correct else incorrect_reward
    breakdown["base"] = base
    breakdown["format_bonus"] = format_bonus if format_ok else 0.0

    return RewardResult(
        reward=base + (format_bonus if format_ok else 0.0),
        answer_correct=is_correct,
        extracted_answer=extracted,
        extraction_method=method,
        format_adherent=format_ok,
        reward_breakdown=breakdown,
        invalid_reason=None,
    )
