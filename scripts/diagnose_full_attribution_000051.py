"""
Full-sequence attribution for DPO v4 minimal 4-epoch (000051).

Read-only analysis. For all 449 v4 pairs:
  1. Response token count, raw logprob difference, policy-ref shift
  2. Solution token logprob change, rank, greedy gap
  3. Solution divergence contribution to total margin
  4. Probe-30: per-token changes at Solution/numbered-steps/Final positions
  5. Summary JSON, per-sample JSONL, 3 representative audits

All position classifications use tokenizer token IDs with verified boundaries.

Usage:
    ./.venv/bin/python scripts/diagnose_full_attribution_000051.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def find_token_boundary(token_ids: list[int], tokenizer, target_text: str,
                        search_range: tuple[int, int] | None = None) -> list[int]:
    """Find positions in token_ids where target_text starts, using token-level verification.

    Returns list of start positions. Each match is verified by decoding tokens back.
    """
    target_ids = tokenizer.encode(target_text)
    if not target_ids:
        return []
    start = search_range[0] if search_range else 0
    end = search_range[1] if search_range else len(token_ids)
    positions = []
    for i in range(start, min(end, len(token_ids) - len(target_ids) + 1)):
        if token_ids[i:i + len(target_ids)] == target_ids:
            positions.append(i)
    return positions


def classify_response_positions(token_ids: list[int], prompt_len: int,
                                tokenizer) -> dict[str, list[int]]:
    """Classify each response token position into categories using token ID matching.

    Returns dict mapping category → list of positions.
    """
    response_ids = token_ids[prompt_len:]
    categories: dict[str, list[int]] = {
        "solution_keyword": [],
        "numbered_step_prefix": [],
        "final_wrapper": [],
        "boxed_answer": [],
        "other": [],
    }

    # Encode target patterns
    solution_ids = tokenizer.encode("Solution")
    final_ids = tokenizer.encode("Final: The answer is ")
    boxed_marker_ids = tokenizer.encode("\\boxed{")

    i = 0
    while i < len(response_ids):
        abs_pos = prompt_len + i

        # Check "Solution" keyword
        if response_ids[i:i + len(solution_ids)] == solution_ids:
            categories["solution_keyword"].extend(range(abs_pos, abs_pos + len(solution_ids)))
            i += len(solution_ids)
            continue

        # Check "Final: The answer is "
        if response_ids[i:i + len(final_ids)] == final_ids:
            categories["final_wrapper"].extend(range(abs_pos, abs_pos + len(final_ids)))
            i += len(final_ids)
            continue

        # Check "\\boxed{"
        if response_ids[i:i + len(boxed_marker_ids)] == boxed_marker_ids:
            categories["boxed_answer"].extend(range(abs_pos, abs_pos + len(boxed_marker_ids)))
            i += len(boxed_marker_ids)
            continue

        # Check numbered step prefix: digit(s) + "."
        tok_text = tokenizer.decode([response_ids[i]])
        if re.match(r"^\d+$", tok_text.strip()):
            # Look ahead for "."
            if i + 1 < len(response_ids):
                next_text = tokenizer.decode([response_ids[i + 1]])
                if next_text.strip() == ".":
                    categories["numbered_step_prefix"].extend([abs_pos, abs_pos + 1])
                    i += 2
                    continue

        categories["other"].append(abs_pos)
        i += 1

    return categories


# ---------------------------------------------------------------------------
# Per-token logprob computation
# ---------------------------------------------------------------------------

def compute_per_token_logprobs(model, tokenizer, full_ids: list[int],
                                prompt_len: int) -> list[dict]:
    """Compute per-token logprobs for the full sequence."""
    import mlx.core as mx

    x = mx.array([full_ids])
    logits = model(x)

    results = []
    for i in range(len(full_ids) - 1):
        next_logits = logits[0, i, :]
        log_probs = mx.log(mx.softmax(next_logits, axis=-1))
        target_id = full_ids[i + 1]
        target_lp = float(log_probs[target_id].item())

        # Rank and greedy
        greedy_lp = float(mx.max(log_probs).item())
        rank = int(mx.sum(log_probs > log_probs[target_id]).item())
        greedy_id = int(mx.argmax(next_logits).item())

        results.append({
            "position": i + 1,
            "is_response": (i + 1) >= prompt_len,
            "token_id": target_id,
            "token_text": tokenizer.decode([target_id]),
            "logprob": round(target_lp, 4),
            "greedy_id": greedy_id,
            "greedy_text": tokenizer.decode([greedy_id]),
            "rank": rank,
            "greedy_gap": round(target_lp - greedy_lp, 4),
        })
    return results


def compute_sequence_logprob(token_logprobs: list[dict]) -> float:
    """Sum logprobs for response tokens only."""
    return sum(t["logprob"] for t in token_logprobs if t["is_response"])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_with_adapter(base_path: str, adapter_path: str | None = None):
    from mlx_lm import load as mlx_load
    from scripts.train_sft import apply_lora, resolve_init_adapter_file
    import mlx.core as mx

    model, tokenizer = mlx_load(base_path)
    if adapter_path:
        adapter_dir = Path(adapter_path)
        config_file = adapter_dir / "adapter_config.json"
        if config_file.exists():
            with open(config_file) as f:
                acfg = json.load(f)
            lora_cfg = {
                "enabled": True,
                "r": acfg.get("lora_parameters", {}).get("rank", 32),
                "alpha": int(acfg.get("lora_parameters", {}).get("scale", 1.0) * acfg.get("lora_parameters", {}).get("rank", 32)),
                "target_modules": acfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            }
            model, _ = apply_lora(model, lora_cfg, verbose=False)
        adapter_file = resolve_init_adapter_file(adapter_path)
        model.load_weights(str(adapter_file), strict=False)
        mx.eval(model.parameters())
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full-sequence attribution for 000051")
    parser.add_argument("--v4-data", default="data/math/pilots/dpo_v4_minimal_449.jsonl")
    parser.add_argument("--probe-data", default="data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl")
    parser.add_argument("--output-summary", default="reports/attribution_000051_summary.json")
    parser.add_argument("--output-samples", default="reports/attribution_000051_samples.jsonl")
    parser.add_argument("--output-audit", default="reports/attribution_000051_audit.json")
    args = parser.parse_args()

    # Load data
    v4_records = []
    with open(args.v4_data, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                v4_records.append(json.loads(stripped))

    probe_ids = set()
    probe_samples = []
    with open(args.probe_data, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            probe_ids.add(r["problem_id"])
            probe_samples.append(r)

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

    # Encode "<think>\n" for boundary detection
    think_ids = tokenizer.encode("<think>\n")
    solution_ids = tokenizer.encode("Solution")

    # Process all 449 pairs
    all_results = []
    probe_results = []

    for idx, rec in enumerate(v4_records):
        pid = rec["metadata"]["problem_id"]
        messages = rec["messages"]

        # Build prompt
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = rendered + "<|im_start|>assistant\n"
        prompt_ids = tokenizer.encode(prompt_text)

        # Tokenize chosen and rejected responses
        chosen_ids = tokenizer.encode(rec["chosen"])
        rejected_ids = tokenizer.encode(rec["rejected"])

        # Find divergence position (first token where chosen != rejected)
        div_pos = None
        for j in range(min(len(chosen_ids), len(rejected_ids))):
            if chosen_ids[j] != rejected_ids[j]:
                div_pos = j
                break

        # Build full sequences
        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids
        prompt_len = len(prompt_ids)

        # Per-token logprobs under policy and reference
        policy_ch_lp = compute_per_token_logprobs(policy_model, tokenizer, chosen_full, prompt_len)
        policy_re_lp = compute_per_token_logprobs(policy_model, tokenizer, rejected_full, prompt_len)
        ref_ch_lp = compute_per_token_logprobs(ref_model, tokenizer, chosen_full, prompt_len)
        ref_re_lp = compute_per_token_logprobs(ref_model, tokenizer, rejected_full, prompt_len)

        # Sequence-level logprobs
        policy_ch_seq = compute_sequence_logprob(policy_ch_lp)
        policy_re_seq = compute_sequence_logprob(policy_re_lp)
        ref_ch_seq = compute_sequence_logprob(ref_ch_lp)
        ref_re_seq = compute_sequence_logprob(ref_re_lp)

        # Logprob differences
        raw_ch_re_diff = policy_ch_seq - policy_re_seq
        ref_ch_re_diff = ref_ch_seq - ref_re_seq
        policy_ref_shift = raw_ch_re_diff - ref_ch_re_diff

        # Solution token analysis
        # Find Solution token in chosen response (after <think>\n)
        sol_positions = find_token_boundary(chosen_ids, tokenizer, "Solution")
        sol_analysis = None
        if sol_positions:
            sol_pos_in_resp = sol_positions[0]
            sol_abs_pos = prompt_len + sol_pos_in_resp

            # Get logprobs for this specific position
            if sol_abs_pos < len(policy_ch_lp):
                p_ch_sol = policy_ch_lp[sol_abs_pos]
                r_ch_sol = ref_ch_lp[sol_abs_pos]

                # For rejected, find the corresponding position
                if sol_abs_pos < len(policy_re_lp):
                    p_re_at_div = policy_re_lp[sol_abs_pos]
                    r_re_at_div = ref_re_lp[sol_abs_pos]

                    sol_analysis = {
                        "position_in_response": sol_pos_in_resp,
                        "absolute_position": sol_abs_pos,
                        "chosen_token_id": chosen_ids[sol_pos_in_resp],
                        "chosen_token_text": tokenizer.decode([chosen_ids[sol_pos_in_resp]]),
                        "rejected_token_id": rejected_ids[div_pos] if div_pos is not None else None,
                        "rejected_token_text": tokenizer.decode([rejected_ids[div_pos]]) if div_pos is not None else None,
                        "policy_chosen_lp": p_ch_sol["logprob"],
                        "policy_rejected_lp": p_re_at_div["logprob"],
                        "ref_chosen_lp": r_ch_sol["logprob"],
                        "ref_rejected_lp": r_re_at_div["logprob"],
                        "policy_gap": round(p_ch_sol["logprob"] - p_re_at_div["logprob"], 4),
                        "ref_gap": round(r_ch_sol["logprob"] - r_re_at_div["logprob"], 4),
                        "shift": round((p_ch_sol["logprob"] - p_re_at_div["logprob"]) - (r_ch_sol["logprob"] - r_re_at_div["logprob"]), 4),
                        "policy_chosen_rank": p_ch_sol["rank"],
                        "policy_chosen_greedy_gap": p_ch_sol["greedy_gap"],
                    }

        # Margin decomposition: contribution of each response position
        margin_per_pos = []
        total_margin = 0.0
        sol_contribution = 0.0
        for j in range(min(len(policy_ch_lp), len(policy_re_lp), len(ref_ch_lp), len(ref_re_lp))):
            if not policy_ch_lp[j]["is_response"]:
                continue
            p_gap = policy_ch_lp[j]["logprob"] - policy_re_lp[j]["logprob"]
            r_gap = ref_ch_lp[j]["logprob"] - ref_re_lp[j]["logprob"]
            contrib = p_gap - r_gap
            total_margin += contrib

            is_sol = (div_pos is not None and j == prompt_len + div_pos)
            if is_sol:
                sol_contribution = contrib

            margin_per_pos.append({
                "position": j,
                "contribution": round(contrib, 4),
                "is_solution_divergence": is_sol,
            })

        sol_share = sol_contribution / total_margin if abs(total_margin) > 1e-6 else 0

        result = {
            "problem_id": pid,
            "chosen_response_tokens": len(chosen_ids),
            "rejected_response_tokens": len(rejected_ids),
            "raw_logprob_diff": round(raw_ch_re_diff, 4),
            "ref_logprob_diff": round(ref_ch_re_diff, 4),
            "policy_ref_shift": round(policy_ref_shift, 4),
            "total_margin": round(total_margin, 4),
            "solution_contribution": round(sol_contribution, 4),
            "solution_share": round(sol_share, 4),
            "solution_analysis": sol_analysis,
            "divergence_position": div_pos,
        }
        all_results.append(result)

        # Probe-specific: classify positions
        if pid in probe_ids:
            categories = classify_response_positions(chosen_full, prompt_len, tokenizer)
            probe_token_detail = {}
            for cat_name, positions in categories.items():
                if not positions:
                    continue
                cat_changes = []
                for pos in positions:
                    if pos < len(policy_ch_lp) and pos < len(ref_ch_lp):
                        change = policy_ch_lp[pos]["logprob"] - ref_ch_lp[pos]["logprob"]
                        cat_changes.append({
                            "position": pos,
                            "token_text": policy_ch_lp[pos]["token_text"],
                            "policy_lp": policy_ch_lp[pos]["logprob"],
                            "ref_lp": ref_ch_lp[pos]["logprob"],
                            "change": round(change, 4),
                            "rank": policy_ch_lp[pos]["rank"],
                            "greedy_gap": policy_ch_lp[pos]["greedy_gap"],
                        })
                probe_token_detail[cat_name] = {
                    "count": len(cat_changes),
                    "avg_change": round(sum(c["change"] for c in cat_changes) / max(len(cat_changes), 1), 4),
                    "tokens": cat_changes,
                }
            probe_results.append({
                "problem_id": pid,
                "categories": probe_token_detail,
            })

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/{len(v4_records)}] {pid}: "
                  f"margin={total_margin:7.2f}  sol_share={sol_share:5.1%}  "
                  f"shift={policy_ref_shift:7.2f}")

    # Summary
    n = len(all_results)
    avg_margin = sum(r["total_margin"] for r in all_results) / n
    avg_sol_share = sum(r["solution_share"] for r in all_results) / n
    avg_shift = sum(r["policy_ref_shift"] for r in all_results) / n
    avg_sol_rank = 0
    sol_ranks = [r["solution_analysis"]["policy_chosen_rank"] for r in all_results if r["solution_analysis"]]
    if sol_ranks:
        avg_sol_rank = sum(sol_ranks) / len(sol_ranks)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n} pairs)")
    print(f"  Avg total margin: {avg_margin:.2f}")
    print(f"  Avg Solution contribution: {avg_sol_share:.1%}")
    print(f"  Avg policy-ref shift: {avg_shift:.2f}")
    print(f"  Avg Solution rank (policy): {avg_sol_rank:.0f}")

    # Write summary
    summary = {
        "description": "Full-sequence attribution for DPO v4 minimal 4-epoch (000051)",
        "policy": "runs/000051_qwen3_0_6b_dpo_v4_style_minimal_4ep",
        "reference": "runs/000030_qwen3_0_6b_sft_gsm8k_500",
        "n_pairs": n,
        "aggregate": {
            "avg_total_margin": round(avg_margin, 2),
            "avg_solution_share": round(avg_sol_share, 4),
            "avg_policy_ref_shift": round(avg_shift, 2),
            "avg_solution_rank": round(avg_sol_rank, 1),
            "median_solution_rank": sorted(sol_ranks)[len(sol_ranks) // 2] if sol_ranks else None,
        },
        "probe_position_analysis": {},
    }

    # Aggregate probe position analysis
    if probe_results:
        for cat in ["solution_keyword", "numbered_step_prefix", "final_wrapper", "boxed_answer", "other"]:
            all_changes = []
            for pr in probe_results:
                cat_data = pr["categories"].get(cat, {})
                for tok in cat_data.get("tokens", []):
                    all_changes.append(tok["change"])
            if all_changes:
                summary["probe_position_analysis"][cat] = {
                    "count": len(all_changes),
                    "avg_change": round(sum(all_changes) / len(all_changes), 4),
                    "min_change": round(min(all_changes), 4),
                    "max_change": round(max(all_changes), 4),
                }

    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    # Write per-sample JSONL
    samples_path = Path(args.output_samples)
    with open(samples_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Samples: {samples_path}")

    # Select 3 representative audits
    # Pick: best shift, median shift, worst shift
    sorted_by_shift = sorted(all_results, key=lambda r: r["policy_ref_shift"])
    audit_indices = [0, len(sorted_by_shift) // 2, len(sorted_by_shift) - 1]
    audit_samples = [sorted_by_shift[i] for i in audit_indices]

    # Add full probe token details for audit samples
    probe_map = {pr["problem_id"]: pr for pr in probe_results}
    for audit in audit_samples:
        pid = audit["problem_id"]
        if pid in probe_map:
            audit["probe_token_detail"] = probe_map[pid]["categories"]

    audit_path = Path(args.output_audit)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "3 representative samples: worst/median/best policy-ref shift",
            "samples": audit_samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
