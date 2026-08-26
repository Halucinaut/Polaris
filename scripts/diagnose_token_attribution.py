"""
Token attribution report for binary prefix DPO (run 000044).

For each of 30 unique probe prompts, decompose the DPO margin per-token:
  - Per-token: policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp
  - Per-token contribution = (policy_ch_lp - policy_re_lp) - (ref_ch_lp - ref_re_lp)
  - First divergence token contribution vs tail contribution
  - Share of total margin from first token vs tail

Usage:
    ./.venv/bin/python scripts/diagnose_token_attribution.py --output reports/token_attribution_000044.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_probe_samples(path: str, n: int = 30) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records[:n]


def load_binary_pairs(path: str) -> dict[str, dict]:
    """Load first occurrence of each problem_id from binary prefix data."""
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r["metadata"]["problem_id"]
            if pid not in seen:
                seen[pid] = r
    return seen


def build_prompt_text(tokenizer, messages: list[dict]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return rendered + "<|im_start|>assistant\n"


def compute_per_token_logprobs(model, tokenizer, prompt_ids: list[int],
                                response_ids: list[int]) -> list[dict]:
    """Compute per-token log P(response_token | prompt + previous response tokens)."""
    import mlx.core as mx

    full_ids = prompt_ids + response_ids
    x = mx.array([full_ids])
    logits = model(x)

    prompt_len = len(prompt_ids)
    results = []
    for i in range(len(response_ids)):
        logit_pos = prompt_len + i - 1
        if logit_pos < 0:
            continue
        next_logits = logits[0, logit_pos, :]
        log_probs = mx.log(mx.softmax(next_logits, axis=-1))
        target_id = response_ids[i]
        target_lp = float(log_probs[target_id].item())
        token_text = tokenizer.decode([target_id])
        results.append({
            "position": i,
            "token_id": target_id,
            "token_text": token_text,
            "logprob": round(target_lp, 4),
        })
    return results


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


def main():
    parser = argparse.ArgumentParser(description="Token attribution for binary prefix DPO")
    parser.add_argument("--probe-data", default="data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl")
    parser.add_argument("--binary-data", default="data/math/pilots/binary_prefix_dpo_control_480.jsonl")
    parser.add_argument("--output", default="reports/token_attribution_000044.json")
    args = parser.parse_args()

    samples = load_probe_samples(args.probe_data, 30)
    pairs = load_binary_pairs(args.binary_data)

    print("Loading policy model (000044)...")
    policy_model, tokenizer = load_model_with_adapter(
        "models/qwen3_0_6b/mlx",
        "runs/000044_qwen3_0_6b_dpo_binary_prefix_ctrl/checkpoints/final",
    )

    print("Loading reference model (M1)...")
    ref_model, _ = load_model_with_adapter(
        "models/qwen3_0_6b/mlx",
        "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
    )

    per_sample = []
    total_margin_sum = 0.0
    first_token_margin_sum = 0.0
    tail_margin_sum = 0.0

    for i, sample in enumerate(samples):
        pid = sample["problem_id"]
        pair = pairs.get(pid)
        if pair is None:
            print(f"  [{i+1}] {pid}: no binary pair found, skipping")
            continue

        messages = pair["messages"]
        chosen_ids = pair["metadata"]["chosen_response_token_ids"]
        rejected_ids = pair["metadata"]["rejected_response_token_ids"]

        prompt_text = build_prompt_text(tokenizer, messages)
        prompt_ids = tokenizer.encode(prompt_text)

        # Per-token logprobs under policy and reference
        policy_ch = compute_per_token_logprobs(policy_model, tokenizer, prompt_ids, chosen_ids)
        policy_re = compute_per_token_logprobs(policy_model, tokenizer, prompt_ids, rejected_ids)
        ref_ch = compute_per_token_logprobs(ref_model, tokenizer, prompt_ids, chosen_ids)
        ref_re = compute_per_token_logprobs(ref_model, tokenizer, prompt_ids, rejected_ids)

        # Compute per-token contributions
        # contribution[i] = (policy_ch_lp[i] - policy_re_lp[i]) - (ref_ch_lp[i] - ref_re_lp[i])
        # But chosen and rejected may differ at position 2 only; positions 0,1 and 3+ are shared.
        # For shared positions, chosen_ids[i] == rejected_ids[i], but logprobs differ due to context.

        # Find divergence position
        div_pos = None
        for j in range(min(len(chosen_ids), len(rejected_ids))):
            if chosen_ids[j] != rejected_ids[j]:
                div_pos = j
                break

        token_details = []
        total_margin = 0.0
        first_token_contrib = 0.0
        tail_contrib = 0.0

        for j in range(min(len(policy_ch), len(policy_re), len(ref_ch), len(ref_re))):
            p_ch = policy_ch[j]["logprob"]
            p_re = policy_re[j]["logprob"]
            r_ch = ref_ch[j]["logprob"]
            r_re = ref_re[j]["logprob"]

            policy_gap = p_ch - p_re
            ref_gap = r_ch - r_re
            contrib = policy_gap - ref_gap

            is_divergence = (j == div_pos)
            is_shared = (chosen_ids[j] == rejected_ids[j])

            token_details.append({
                "position": j,
                "chosen_id": chosen_ids[j],
                "rejected_id": rejected_ids[j],
                "chosen_text": policy_ch[j]["token_text"],
                "rejected_text": policy_re[j]["token_text"],
                "is_shared": is_shared,
                "is_divergence": is_divergence,
                "policy_chosen_lp": round(p_ch, 4),
                "policy_rejected_lp": round(p_re, 4),
                "ref_chosen_lp": round(r_ch, 4),
                "ref_rejected_lp": round(r_re, 4),
                "policy_gap": round(policy_gap, 4),
                "ref_gap": round(ref_gap, 4),
                "contribution": round(contrib, 4),
            })

            total_margin += contrib
            if is_divergence:
                first_token_contrib += contrib
            else:
                tail_contrib += contrib

        first_share = first_token_contrib / total_margin if abs(total_margin) > 1e-6 else 0
        tail_share = tail_contrib / total_margin if abs(total_margin) > 1e-6 else 0

        per_sample.append({
            "problem_id": pid,
            "divergence_position": div_pos,
            "total_margin": round(total_margin, 4),
            "first_token_contribution": round(first_token_contrib, 4),
            "tail_contribution": round(tail_contrib, 4),
            "first_token_share": round(first_share, 4),
            "tail_share": round(tail_share, 4),
            "tokens": token_details,
        })

        total_margin_sum += total_margin
        first_token_margin_sum += first_token_contrib
        tail_margin_sum += tail_contrib

        print(f"  [{i+1:2d}/30] {pid}: margin={total_margin:7.2f}  "
              f"first={first_token_contrib:7.2f} ({first_share:5.1%})  "
              f"tail={tail_contrib:7.2f} ({tail_share:5.1%})")

    n = len(per_sample)
    avg_first_share = first_token_margin_sum / total_margin_sum if abs(total_margin_sum) > 1e-6 else 0
    avg_tail_share = tail_margin_sum / total_margin_sum if abs(total_margin_sum) > 1e-6 else 0

    print(f"\n{'='*60}")
    print(f"AGGREGATE: {n} samples")
    print(f"  Total margin sum: {total_margin_sum:.2f}")
    print(f"  First token contribution: {first_token_margin_sum:.2f} ({avg_first_share:.1%})")
    print(f"  Tail contribution: {tail_margin_sum:.2f} ({avg_tail_share:.1%})")

    output = {
        "description": (
            "Per-token DPO margin attribution for binary prefix control (000044). "
            "Contribution = (policy_ch_lp - policy_re_lp) - (ref_ch_lp - ref_re_lp) at each position."
        ),
        "policy_adapter": "runs/000044_qwen3_0_6b_dpo_binary_prefix_ctrl/checkpoints/final",
        "reference_adapter": "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
        "n_samples": n,
        "aggregate": {
            "total_margin_sum": round(total_margin_sum, 2),
            "first_token_contribution_sum": round(first_token_margin_sum, 2),
            "tail_contribution_sum": round(tail_margin_sum, 2),
            "first_token_share": round(avg_first_share, 4),
            "tail_share": round(avg_tail_share, 4),
        },
        "per_sample": per_sample,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
