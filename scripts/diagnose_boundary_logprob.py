"""
Extended boundary diagnosis: quantify the DPO style-transfer gap.

For each model (M1, DPO v2, v3, v4, SFT control) on all 30 probe samples:
  1. Multi-token prefix scoring: log P("Solution:\n1." | prompt + <think>\n)
  2. First-token vocabulary rank and logprob gap vs greedy
  3. Forced-prefix generation: <think>\nSolution:\n1. → 128 tokens, style check

Style adherence uses check_style_adherence from scripts/eval_style_dpo.py.

Usage:
    ./.venv/bin/python scripts/diagnose_boundary_logprob.py --output reports/boundary_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_style_dpo import check_style_adherence


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_probe_samples(path: str, n: int = 30) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records[:n]


def build_prompt(tokenizer, messages: list[dict]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return rendered + "<|im_start|>assistant\n"


def score_prefix(model, tokenizer, prompt_ids: list[int], prefix_text: str) -> dict:
    """Score a multi-token prefix: log P(prefix | prompt), per-token logprobs.

    Also reports the first token's vocab rank and logprob gap vs greedy.
    """
    import mlx.core as mx

    prefix_ids = tokenizer.encode(prefix_text)
    if not prefix_ids:
        return {"total_logprob": float("-inf"), "tokens": [], "first_token_rank": None}

    full_ids = prompt_ids + prefix_ids
    x = mx.array([full_ids])
    logits = model(x)

    prompt_len = len(prompt_ids)
    prefix_len = len(prefix_ids)

    token_details = []
    total_logprob = 0.0
    first_token_rank = None
    first_token_gap = None

    for i in range(prefix_len):
        logit_pos = prompt_len + i - 1
        if logit_pos < 0:
            continue
        next_logits = logits[0, logit_pos, :]
        log_probs = mx.log(mx.softmax(next_logits, axis=-1))

        target_id = prefix_ids[i]
        target_lp = float(log_probs[target_id].item())
        total_logprob += target_lp

        if i == 0:
            rank = int(mx.sum(log_probs > target_lp).item())
            first_token_rank = rank
            greedy_lp = float(mx.max(log_probs).item())
            first_token_gap = target_lp - greedy_lp

        token_details.append({
            "token_id": target_id,
            "token_text": tokenizer.decode([target_id]),
            "logprob": round(target_lp, 4),
        })

    return {
        "total_logprob": round(total_logprob, 4),
        "num_tokens": prefix_len,
        "avg_logprob": round(total_logprob / max(prefix_len, 1), 4),
        "first_token_rank": first_token_rank,
        "first_token_gap": round(first_token_gap, 4) if first_token_gap is not None else None,
        "tokens": token_details,
    }


def forced_prefix_generate(model, tokenizer, prompt_ids: list[int],
                           prefix_text: str, max_tokens: int = 128) -> dict:
    """Force a prefix, then generate max_tokens, check style adherence."""
    import mlx.core as mx

    prefix_ids = tokenizer.encode(prefix_text)
    tokens = list(prompt_ids) + list(prefix_ids)

    for _ in range(max_tokens):
        x = mx.array([tokens])
        logits = model(x)
        next_logits = logits[0, -1, :]
        next_id = int(mx.argmax(next_logits).item())
        if next_id == tokenizer.eos_token_id:
            break
        tokens.append(next_id)

    generated = tokenizer.decode(tokens[len(prompt_ids):])
    adherent, reasons = check_style_adherence(generated)

    return {
        "generated_text": generated,
        "style_adherent": adherent,
        "style_errors": reasons,
        "num_tokens_generated": len(tokens) - len(prompt_ids),
    }


def load_model_with_adapter(base_path: str, adapter_path: str | None = None):
    """Load model, optionally with LoRA adapter."""
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
    parser = argparse.ArgumentParser(description="Extended boundary logprob diagnosis")
    parser.add_argument("--probe-data", default="data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl")
    parser.add_argument("--output", default="reports/boundary_diagnosis.json")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--force-gen-tokens", type=int, default=128)
    args = parser.parse_args()

    samples = load_probe_samples(args.probe_data, args.n_samples)

    models = {
        "m1_sft": ("models/qwen3_0_6b/mlx", "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final"),
        "dpo_v2": ("models/qwen3_0_6b/mlx", "runs/000036_qwen3_0_6b_dpo_v2_style/checkpoints/final"),
        "dpo_v3": ("models/qwen3_0_6b/mlx", "runs/000041_qwen3_0_6b_dpo_v3_style_lr5e6/checkpoints/final"),
        "dpo_v4": ("models/qwen3_0_6b/mlx", "runs/000043_qwen3_0_6b_dpo_v4_style_minimal/checkpoints/final"),
        "sft_ctrl": ("models/qwen3_0_6b/mlx", "runs/000039_qwen3_0_6b_sft_v2_style_control/checkpoints/final"),
    }

    scored_prefix = "Solution:\n1."
    forced_prefix = "<think>\nSolution:\n1."

    all_results = {}

    for model_name, (base, adapter) in models.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        model, tokenizer = load_model_with_adapter(base, adapter)

        model_results = []
        for i, sample in enumerate(samples):
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve the problem and put the final answer in \\boxed{}."},
                {"role": "user", "content": sample["problem"]},
            ]
            prompt = build_prompt(tokenizer, messages)
            prompt_ids = tokenizer.encode(prompt)

            think_prefix = "<think>\n"
            think_ids = tokenizer.encode(think_prefix)
            branch_prompt_ids = prompt_ids + think_ids

            prefix_score = score_prefix(model, tokenizer, branch_prompt_ids, scored_prefix)

            import mlx.core as mx
            x = mx.array([branch_prompt_ids])
            logits = model(x)
            next_logits = logits[0, -1, :]
            greedy_id = int(mx.argmax(next_logits).item())
            greedy_token = tokenizer.decode([greedy_id])
            greedy_is_solution = greedy_token.strip().startswith("Solution")

            forced = forced_prefix_generate(
                model, tokenizer, prompt_ids, forced_prefix, args.force_gen_tokens,
            )

            model_results.append({
                "problem_id": sample["problem_id"],
                "prefix_logprob": prefix_score["total_logprob"],
                "prefix_avg_logprob": prefix_score["avg_logprob"],
                "prefix_num_tokens": prefix_score["num_tokens"],
                "first_token_rank": prefix_score["first_token_rank"],
                "first_token_gap": prefix_score["first_token_gap"],
                "greedy_token": repr(greedy_token),
                "greedy_enters_chosen": greedy_is_solution,
                "forced_style_adherent": forced["style_adherent"],
                "forced_style_errors": forced["style_errors"],
                "forced_tokens_generated": forced["num_tokens_generated"],
            })

            status = "✓" if forced["style_adherent"] else "✗"
            print(f"  [{i+1:2d}/{len(samples)}] {sample['problem_id']}: "
                  f"prefix_lp={prefix_score['total_logprob']:7.1f}  "
                  f"rank={prefix_score['first_token_rank']:5d}  "
                  f"gap={prefix_score['first_token_gap']:6.1f}  "
                  f"greedy={'Sol' if greedy_is_solution else '---'}  "
                  f"forced={status}")

        all_results[model_name] = model_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':12s} {'Greedy→Sol':>10s} {'AvgPrefixLP':>11s} {'AvgRank':>8s} {'AvgGap':>8s} {'Forced✓':>8s}")
    print("-" * 60)
    for model_name, results in all_results.items():
        n = len(results)
        n_chosen = sum(1 for r in results if r["greedy_enters_chosen"])
        avg_lp = sum(r["prefix_logprob"] for r in results) / n
        avg_rank = sum(r["first_token_rank"] for r in results) / n
        avg_gap = sum(r["first_token_gap"] for r in results) / n
        n_forced = sum(1 for r in results if r["forced_style_adherent"])
        print(f"{model_name:12s} {n_chosen:4d}/{n:4d}   {avg_lp:8.1f}  {avg_rank:7.1f}  {avg_gap:7.1f}  {n_forced:4d}/{n:4d}")

    # Save
    output = {
        "description": (
            "Extended boundary diagnosis: multi-token prefix scoring, "
            "first-token vocab rank, forced-prefix generation. "
            "Style check uses eval_style_dpo.check_style_adherence."
        ),
        "scored_prefix": scored_prefix,
        "forced_prefix": forced_prefix,
        "probe_data": args.probe_data,
        "n_samples": len(samples),
        "force_gen_tokens": args.force_gen_tokens,
        "results": all_results,
        "summary": {},
    }
    for model_name, results in all_results.items():
        n = len(results)
        output["summary"][model_name] = {
            "greedy_enters_chosen": sum(1 for r in results if r["greedy_enters_chosen"]),
            "forced_style_adherent": sum(1 for r in results if r["forced_style_adherent"]),
            "avg_prefix_logprob": round(sum(r["prefix_logprob"] for r in results) / n, 2),
            "avg_first_token_rank": round(sum(r["first_token_rank"] for r in results) / n, 1),
            "avg_first_token_gap": round(sum(r["first_token_gap"] for r in results) / n, 2),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
