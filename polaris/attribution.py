"""
Pure (no-MLX) helpers for full-sequence DPO attribution.

Every function in this module is importable under the system Python with no
mlx / mlx-lm dependency.  MLX model inference lives in the diagnosis script.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Token span classification
# ---------------------------------------------------------------------------

def classify_response_positions(
    full_ids: list[int],
    chosen_mask: list[int],
    encoded_patterns: list[tuple[str, list[int]]],
) -> dict[int, str]:
    """Classify each response-token position into style-template categories.

    Parameters
    ----------
    full_ids : list[int]
        Full token-ID sequence (prompt + response).
    chosen_mask : list[int]
        0 for prompt positions, 1 for response positions.
    encoded_patterns : list[tuple[str, list[int]]]
        ``(category_name, token_ids)`` pairs.  Evaluated in order; first
        match wins for overlapping spans.

    Returns
    -------
    dict[int, str]
        absolute_position → category name.  Every response position appears;
        unmatched positions are ``"unclassified"``.
    """
    prompt_len = sum(1 for m in chosen_mask if m == 0)
    resp_ids = full_ids[prompt_len:]
    n_resp = len(resp_ids)

    classified: dict[int, str] = {}   # rel_pos → category

    for cat_name, pat_ids in encoded_patterns:
        plen = len(pat_ids)
        if plen == 0:
            continue
        for i in range(n_resp - plen + 1):
            if any((i + j) in classified for j in range(plen)):
                continue
            if resp_ids[i:i + plen] == pat_ids:
                for j in range(plen):
                    classified[i + j] = cat_name

    result: dict[int, str] = {}
    for i in range(n_resp):
        abs_pos = prompt_len + i
        result[abs_pos] = classified.get(i, "unclassified")
    return result


def build_style_patterns(tokenizer) -> list[tuple[str, list[int]]]:
    """Build the standard style-template pattern list for a tokenizer."""
    patterns: list[tuple[str, list[int]]] = [
        ("final_wrapper",    tokenizer.encode("Final: The answer is ")),
        ("boxed_answer",     tokenizer.encode("\\boxed{")),
        ("solution_keyword", tokenizer.encode("Solution")),
    ]
    for k in range(1, 11):
        patterns.append(("numbered_step_prefix", tokenizer.encode(f"{k}. ")))
    return patterns


# ---------------------------------------------------------------------------
# Sequence-level arithmetic (all pure — no model calls)
# ---------------------------------------------------------------------------

def find_divergence_position(
    chosen_resp_ids: list[int],
    rejected_resp_ids: list[int],
) -> int | None:
    """Index of first differing token in the two response-ID lists."""
    for j in range(min(len(chosen_resp_ids), len(rejected_resp_ids))):
        if chosen_resp_ids[j] != rejected_resp_ids[j]:
            return j
    return None


def find_token_in_response(
    resp_ids: list[int],
    target_token_id: int,
    start: int = 0,
) -> int | None:
    """Relative position of *target_token_id* in *resp_ids*, or None."""
    for j in range(start, len(resp_ids)):
        if resp_ids[j] == target_token_id:
            return j
    return None


def sum_masked_logprob(token_logprobs: list[dict]) -> float:
    """Sum ``logprob`` values over entries returned by per-token computation."""
    return sum(entry["logprob"] for entry in token_logprobs)


def lookup_logprob(
    token_logprobs: list[dict],
    abs_position: int,
) -> float | None:
    """Return logprob at *abs_position*, or None if absent."""
    for e in token_logprobs:
        if e["abs_position"] == abs_position:
            return e["logprob"]
    return None


def lookup_entry(
    token_logprobs: list[dict],
    abs_position: int,
) -> dict | None:
    """Return the full entry dict at *abs_position*, or None."""
    for e in token_logprobs:
        if e["abs_position"] == abs_position:
            return e
    return None


def compute_exact_margin(
    policy_ch: list[dict],
    policy_re: list[dict],
    ref_ch: list[dict],
    ref_re: list[dict],
) -> float:
    """Full-sequence DPO margin (no common-length truncation).

    ``sum_masked(π_ch - πref_ch) - sum_masked(π_re - πref_re)``
    """
    return (
        (sum_masked_logprob(policy_ch) - sum_masked_logprob(ref_ch))
        - (sum_masked_logprob(policy_re) - sum_masked_logprob(ref_re))
    )


def assemble_solution_metrics(
    *,
    sol_abs_pos: int | None,
    div_pos_in_resp: int | None,
    prompt_len: int,
    policy_ch: list[dict],
    policy_re: list[dict],
    ref_ch: list[dict],
    ref_re: list[dict],
    exact_margin: float,
    expected_solution_id: int,
) -> dict[str, Any]:
    """Compute all Solution-related metrics from pre-computed logprob lists.

    Returns a dict with keys: ``solution_abs_position``,
    ``solution_chosen_shift``, ``solution_shift_minus_rejected_div``,
    ``solution_contribution``, ``solution_share_of_margin``,
    ``solution_analysis``.
    """
    result: dict[str, Any] = {
        "solution_abs_position": sol_abs_pos,
        "solution_token_id": expected_solution_id,
        "solution_chosen_shift": None,
        "solution_shift_minus_rejected_div": None,
        "solution_contribution": None,
        "solution_share_of_margin": None,
        "solution_analysis": None,
    }
    if sol_abs_pos is None:
        return result

    p_ch_sol = lookup_logprob(policy_ch, sol_abs_pos)
    r_ch_sol = lookup_logprob(ref_ch, sol_abs_pos)
    if p_ch_sol is None or r_ch_sol is None:
        return result

    sol_chosen_shift = round(p_ch_sol - r_ch_sol, 6)
    result["solution_chosen_shift"] = sol_chosen_shift

    # Rejected divergence shift
    rej_div_abs = prompt_len + div_pos_in_resp if div_pos_in_resp is not None else None
    p_re_div = lookup_logprob(policy_re, rej_div_abs) if rej_div_abs is not None else None
    r_re_div = lookup_logprob(ref_re, rej_div_abs) if rej_div_abs is not None else None
    if p_re_div is not None and r_re_div is not None:
        result["solution_shift_minus_rejected_div"] = round(
            sol_chosen_shift - (p_re_div - r_re_div), 6)

    # Contribution to total margin
    sol_idx_ch = lookup_entry(policy_ch, sol_abs_pos)
    sol_idx_ref = lookup_entry(ref_ch, sol_abs_pos)
    sol_div_re = lookup_entry(policy_re, rej_div_abs) if rej_div_abs is not None else None
    sol_div_ref_re = lookup_entry(ref_re, rej_div_abs) if rej_div_abs is not None else None
    if sol_idx_ch and sol_idx_ref and sol_div_re and sol_div_ref_re:
        ch_contrib = sol_idx_ch["logprob"] - sol_idx_ref["logprob"]
        re_contrib = sol_div_re["logprob"] - sol_div_ref_re["logprob"]
        sol_contribution = round(ch_contrib - re_contrib, 6)
        result["solution_contribution"] = sol_contribution
        if abs(exact_margin) > 1e-8:
            result["solution_share_of_margin"] = round(
                sol_contribution / exact_margin, 6)

    # Rich solution_analysis
    analysis: dict[str, Any] = {
        "abs_position": sol_abs_pos,
        "token_id": sol_idx_ch["token_id"],
        "token_text": sol_idx_ch["token_text"],
        "policy_chosen_lp": sol_idx_ch["logprob"],
        "ref_chosen_lp": sol_idx_ref["logprob"],
        "policy_chosen_rank": sol_idx_ch["rank"],
        "policy_chosen_greedy_gap": sol_idx_ch["greedy_gap"],
    }
    if sol_div_re and sol_div_ref_re:
        analysis["rejected_div_abs_position"] = rej_div_abs
        analysis["rejected_token_id"] = sol_div_re["token_id"]
        analysis["rejected_token_text"] = sol_div_re["token_text"]
        analysis["policy_rejected_lp"] = sol_div_re["logprob"]
        analysis["ref_rejected_lp"] = sol_div_ref_re["logprob"]
        analysis["policy_rejected_rank"] = sol_div_re["rank"]
        analysis["chosen_shift"] = sol_chosen_shift
        analysis["rejected_shift"] = round(
            sol_div_re["logprob"] - sol_div_ref_re["logprob"], 6)
        analysis["shift_minus_rejected_div"] = result["solution_shift_minus_rejected_div"]
    result["solution_analysis"] = analysis
    return result


# ---------------------------------------------------------------------------
# Summary aggregation (pure)
# ---------------------------------------------------------------------------

def aggregate_probe_categories(
    probe_results: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate per-category logprob-change statistics across probe samples."""
    cat_names = ["solution_keyword", "numbered_step_prefix",
                 "final_wrapper", "boxed_answer", "unclassified", "other"]
    out: dict[str, dict[str, float]] = {}
    for cat in cat_names:
        all_changes: list[float] = []
        for pr in probe_results:
            cat_data = pr["categories"].get(cat, {})
            for tok in cat_data.get("changes", []):
                all_changes.append(tok["change"])
        if all_changes:
            out[cat] = {
                "count": len(all_changes),
                "avg_change": round(sum(all_changes) / len(all_changes), 6),
                "min_change": round(min(all_changes), 6),
                "max_change": round(max(all_changes), 6),
            }
    return out


def compute_classification_coverage(
    probe_results: list[dict],
) -> dict[str, Any]:
    """Compute per-category token counts across all probe samples.

    Returns a dict with ``total_tokens``, ``classified_count``,
    ``unclassified_count``, ``coverage_ratio``, and per-category counts.
    """
    cat_counts: dict[str, int] = {}
    for pr in probe_results:
        for cat_name, cat_data in pr.get("categories", {}).items():
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + cat_data.get("count", 0)

    total = sum(cat_counts.values())
    classified = total - cat_counts.get("unclassified", 0)
    return {
        "total_tokens": total,
        "classified_count": classified,
        "unclassified_count": cat_counts.get("unclassified", 0),
        "coverage_ratio": round(classified / total, 4) if total > 0 else 0.0,
        "per_category": cat_counts,
    }


def build_summary(
    all_results: list[dict],
    probe_results: list[dict],
    expected_solution_id: int,
) -> dict[str, Any]:
    """Build the top-level summary dict from per-sample results."""
    n = len(all_results)
    avg_margin = sum(r["exact_margin"] for r in all_results) / n
    sol_shares = [r["solution_share_of_margin"] for r in all_results
                  if r["solution_share_of_margin"] is not None]
    avg_sol_share = (sum(sol_shares) / len(sol_shares)) if sol_shares else 0.0
    sol_ranks = [r["solution_analysis"]["policy_chosen_rank"]
                 for r in all_results if r["solution_analysis"]]
    avg_sol_rank = (sum(sol_ranks) / len(sol_ranks)) if sol_ranks else 0.0
    sol_shifts = [r["solution_chosen_shift"] for r in all_results
                  if r["solution_chosen_shift"] is not None]
    avg_sol_shift = (sum(sol_shifts) / len(sol_shifts)) if sol_shifts else 0.0

    summary: dict[str, Any] = {
        "description": "Full-sequence attribution for DPO v4 minimal 4-epoch (000051)",
        "method": "uses train_dpo.tokenize_pair(); logits[t-1] for target position t",
        "n_pairs": n,
        "aggregate": {
            "avg_exact_margin": round(avg_margin, 4),
            "avg_solution_chosen_shift": round(avg_sol_shift, 4),
            "avg_solution_share_of_margin": round(avg_sol_share, 4),
            "avg_solution_rank": round(avg_sol_rank, 1),
            "median_solution_rank": (
                sorted(sol_ranks)[len(sol_ranks) // 2] if sol_ranks else None),
            "solution_token_id": expected_solution_id,
        },
        "probe_position_analysis": aggregate_probe_categories(probe_results),
        "classification_coverage": compute_classification_coverage(probe_results),
    }
    return summary
