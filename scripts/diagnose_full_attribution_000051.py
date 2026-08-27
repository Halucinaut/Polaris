"""
Full-sequence attribution for DPO run 000051 (v4 minimal 4-epoch).

Read-only analysis. For all 449 v4 pairs:
  1. Uses train_dpo.tokenize_pair() for correct prompt/response masking.
  2. Off-by-one fixed: target position t reads logits[t-1].
  3. Full-sequence margin = sum_masked(policy_ch - ref_ch) - sum_masked(policy_re - ref_re).
  4. Solution divergence: two quantities (chosen shift, chosen shift minus rejected shift).
  5. Position classification verifies each token span; unclassified if no match.
  6. Outputs: summary JSON, per-sample JSONL, 3 representative audits.

Usage:
    ./.venv/bin/python scripts/diagnose_full_attribution_000051.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polaris.attribution import (
    assemble_solution_metrics,
    build_style_patterns,
    build_summary,
    classify_response_positions,
    compute_exact_margin,
    find_divergence_position,
    find_token_in_response,
    lookup_entry,
    sum_masked_logprob,
)


# ---------------------------------------------------------------------------
# Per-token logprob computation (MLX-dependent)
# ---------------------------------------------------------------------------

def compute_per_token_logprobs(
    model,
    tokenizer,
    full_ids: list[int],
    response_mask: list[int],
) -> list[dict]:
    """Compute per-token logprobs for response positions only.

    Off-by-one rule: target position *t* reads ``logits[t-1]``.
    """
    import mlx.core as mx

    x = mx.array([full_ids])
    logits = model(x)                          # (1, T, V)
    # Numerically stable log-softmax (matches losses.cross_entropy path)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    results = []
    for t in range(len(full_ids)):
        if response_mask[t] == 0:
            continue
        # target position t → logits at position t-1
        lp_vec = log_probs[0, t - 1, :]
        target_id = full_ids[t]
        target_lp = float(lp_vec[target_id].item())

        greedy_lp = float(mx.max(lp_vec).item())
        rank = int(mx.sum(lp_vec > lp_vec[target_id]).item())
        greedy_id = int(mx.argmax(logits[0, t - 1, :]).item())

        results.append({
            "abs_position": t,
            "token_id": target_id,
            "token_text": tokenizer.decode([target_id]),
            "logprob": round(target_lp, 6),
            "rank": rank,
            "greedy_id": greedy_id,
            "greedy_text": tokenizer.decode([greedy_id]),
            "greedy_gap": round(target_lp - greedy_lp, 6),
        })
    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_with_adapter(base_path: str, adapter_path: str | None = None):
    from mlx_lm import load as mlx_load
    from mlx_lm.utils import load_adapters
    import mlx.core as mx

    model, tokenizer = mlx_load(base_path)
    if adapter_path:
        model = load_adapters(model, str(adapter_path))
        mx.eval(model.parameters())
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-sequence attribution for 000051 (off-by-one corrected)")
    parser.add_argument("--v4-data",
                        default="data/math/pilots/dpo_v4_minimal_449.jsonl")
    parser.add_argument("--probe-data",
                        default="data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl")
    parser.add_argument("--output-summary",
                        default="reports/attribution_000051_summary.json")
    parser.add_argument("--output-samples",
                        default="reports/attribution_000051_samples.jsonl")
    parser.add_argument("--output-audit",
                        default="reports/attribution_000051_audit.json")
    args = parser.parse_args()

    # Import tokenize_pair from train_dpo (canonical prompt/response masking)
    from scripts.train_dpo import tokenize_pair

    # Load data
    v4_records: list[dict] = []
    with open(args.v4_data, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                v4_records.append(json.loads(stripped))

    probe_ids: set[str] = set()
    with open(args.probe_data, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            probe_ids.add(r.get("metadata", {}).get("problem_id",
                                                     r.get("problem_id", "")))

    print(f"Loaded {len(v4_records)} v4 pairs, {len(probe_ids)} probe IDs")

    # Load models
    print("Loading policy (000051)...")
    policy_model, tokenizer = load_model_with_adapter(
        "models/qwen3_0_6b/mlx",
        "runs/000051_qwen3_0_6b_dpo_v4_style_minimal_4ep/checkpoints/final",
    )
    print("Loading reference (M1)...")
    ref_model, _ = load_model_with_adapter(
        "models/qwen3_0_6b/mlx",
        "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
    )

    # Pre-compute patterns and verify Solution token ID
    expected_solution_id = tokenizer.encode("Solution")[0]
    style_patterns = build_style_patterns(tokenizer)
    print(f"Solution token_id = {expected_solution_id}")

    # Process all pairs
    all_results: list[dict] = []
    probe_results: list[dict] = []

    for idx, rec in enumerate(v4_records):
        pid = rec["metadata"]["problem_id"]
        messages = rec["messages"]

        # Use train_dpo.tokenize_pair for canonical masking
        pair = tokenize_pair(
            tokenizer, messages,
            rec["chosen"], rec["rejected"],
            max_seq_length=2048,
            prompt_suffix=rec.get("prompt_suffix", ""),
            no_eos=rec.get("no_eos", False),
        )
        if pair is None:
            print(f"  SKIP {pid}: tokenize_pair returned None")
            continue

        chosen_full   = pair["chosen_ids"]
        rejected_full = pair["rejected_ids"]
        chosen_mask   = pair["chosen_mask"]
        rejected_mask = pair["rejected_mask"]
        prompt_len    = pair["prompt_len"]

        # --- Per-token logprobs (response positions only, logits[t-1]) ---
        policy_ch = compute_per_token_logprobs(
            policy_model, tokenizer, chosen_full, chosen_mask)
        policy_re = compute_per_token_logprobs(
            policy_model, tokenizer, rejected_full, rejected_mask)
        ref_ch = compute_per_token_logprobs(
            ref_model, tokenizer, chosen_full, chosen_mask)
        ref_re = compute_per_token_logprobs(
            ref_model, tokenizer, rejected_full, rejected_mask)

        # --- Pure helpers from polaris.attribution ---
        chosen_resp_ids = chosen_full[prompt_len:]
        rejected_resp_ids = rejected_full[prompt_len:]
        div_pos_in_resp = find_divergence_position(
            chosen_resp_ids, rejected_resp_ids)
        sol_rel = find_token_in_response(chosen_resp_ids, expected_solution_id)
        sol_abs_pos = (prompt_len + sol_rel) if sol_rel is not None else None

        exact_margin = compute_exact_margin(policy_ch, policy_re, ref_ch, ref_re)

        sol_metrics = assemble_solution_metrics(
            sol_abs_pos=sol_abs_pos,
            div_pos_in_resp=div_pos_in_resp,
            prompt_len=prompt_len,
            policy_ch=policy_ch,
            policy_re=policy_re,
            ref_ch=ref_ch,
            ref_re=ref_re,
            exact_margin=exact_margin,
            expected_solution_id=expected_solution_id,
        )

        # Assert Solution token_id at the attributed position
        if sol_abs_pos is not None:
            sol_entry = lookup_entry(policy_ch, sol_abs_pos)
            if sol_entry is not None:
                assert sol_entry["token_id"] == expected_solution_id, (
                    f"Solution token_id mismatch at position {sol_abs_pos}: "
                    f"expected {expected_solution_id}, got {sol_entry['token_id']}")

        result = {
            "problem_id": pid,
            "chosen_response_tokens": pair["chosen_len"],
            "rejected_response_tokens": pair["rejected_len"],
            "exact_margin": round(exact_margin, 6),
            "policy_chosen_seq_lp": round(sum_masked_logprob(policy_ch), 6),
            "policy_rejected_seq_lp": round(sum_masked_logprob(policy_re), 6),
            "ref_chosen_seq_lp": round(sum_masked_logprob(ref_ch), 6),
            "ref_rejected_seq_lp": round(sum_masked_logprob(ref_re), 6),
            "divergence_position_in_response": div_pos_in_resp,
            **sol_metrics,
        }
        all_results.append(result)

        # --- Probe-30: position classification ---
        if pid in probe_ids:
            categories = classify_response_positions(
                chosen_full, chosen_mask, style_patterns)
            probe_token_detail: dict[str, dict] = {}
            for abs_pos, cat_name in sorted(categories.items()):
                p_entry = lookup_entry(policy_ch, abs_pos)
                r_entry = lookup_entry(ref_ch, abs_pos)
                if p_entry is None or r_entry is None:
                    continue
                change = round(p_entry["logprob"] - r_entry["logprob"], 6)
                if cat_name not in probe_token_detail:
                    probe_token_detail[cat_name] = {
                        "count": 0, "changes": [], "avg_change": 0.0}
                probe_token_detail[cat_name]["count"] += 1
                probe_token_detail[cat_name]["changes"].append({
                    "position": abs_pos,
                    "token_text": p_entry["token_text"],
                    "policy_lp": p_entry["logprob"],
                    "ref_lp": r_entry["logprob"],
                    "change": change,
                    "rank": p_entry["rank"],
                    "greedy_gap": p_entry["greedy_gap"],
                })
            for cat_name, cat_data in probe_token_detail.items():
                changes = cat_data["changes"]
                if changes:
                    cat_data["avg_change"] = round(
                        sum(c["change"] for c in changes) / len(changes), 6)
            probe_results.append({
                "problem_id": pid,
                "categories": probe_token_detail,
            })

        if (idx + 1) % 50 == 0 or idx == 0:
            sol_share = sol_metrics.get("solution_share_of_margin") or 0
            print(f"  [{idx+1:3d}/{len(v4_records)}] {pid}: "
                  f"margin={exact_margin:8.3f}  sol_share={sol_share:6.1%}")

    # ---- Summary ----
    summary = build_summary(all_results, probe_results, expected_solution_id)
    summary["policy"] = "runs/000051_qwen3_0_6b_dpo_v4_style_minimal_4ep"
    summary["reference"] = "runs/000030_qwen3_0_6b_sft_gsm8k_500"

    n = len(all_results)
    avg_margin = summary["aggregate"]["avg_exact_margin"]
    avg_sol_shift = summary["aggregate"]["avg_solution_chosen_shift"]
    avg_sol_share = summary["aggregate"]["avg_solution_share_of_margin"]
    avg_sol_rank = summary["aggregate"]["avg_solution_rank"]

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n} pairs)")
    print(f"  Avg exact margin:               {avg_margin:10.4f}")
    print(f"  Avg Solution chosen-shift:       {avg_sol_shift:10.4f}")
    print(f"  Avg Solution share of margin:    {avg_sol_share:10.1%}")
    print(f"  Avg Solution rank (policy):      {avg_sol_rank:10.1f}")

    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    # Per-sample JSONL
    samples_path = Path(args.output_samples)
    with open(samples_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Samples: {samples_path}")

    # 3 representative audits: worst / median / best policy-ref shift
    sol_shift_sorted = sorted(
        [r for r in all_results if r["solution_chosen_shift"] is not None],
        key=lambda r: r["solution_chosen_shift"])
    if len(sol_shift_sorted) >= 3:
        audit_indices = [0, len(sol_shift_sorted) // 2, len(sol_shift_sorted) - 1]
        audit_samples = [sol_shift_sorted[i] for i in audit_indices]
    else:
        audit_samples = sol_shift_sorted

    probe_map = {pr["problem_id"]: pr for pr in probe_results}
    for audit in audit_samples:
        pid = audit["problem_id"]
        if pid in probe_map:
            audit["probe_token_detail"] = probe_map[pid]["categories"]

    audit_path = Path(args.output_audit)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": (
                "3 representative samples: worst/median/best "
                "chosen Solution policy-ref shift"),
            "samples": audit_samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
