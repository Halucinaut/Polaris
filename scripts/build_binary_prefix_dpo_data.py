"""
Build binary prefix DPO control data for mechanism diagnosis.

For each of 30 probe prompts:
  - Divergence at prompt + "<think>\n"
  - Chosen: first token = "Solution" (a token M1 never greedily picks)
  - Rejected: first token = M1's actual greedy token at that position
  - After the first token, both sequences are identical (the rest of a
    minimal style-compliant completion)
  - Full token IDs saved, re-encoding consistency verified

30 pairs × 16 repetitions = 480 pairs total.
This data is for mechanism diagnosis only, not math evaluation.

Usage:
    ./.venv/bin/python scripts/build_binary_prefix_dpo_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem and put the final answer in \\boxed{}."
)


def load_probe_samples(path: str, n: int = 30) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records[:n]


def build_prompt_text(tokenizer, messages: list[dict]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return rendered + "<|im_start|>assistant\n"


def get_greedy_token_id(model, tokenizer, input_ids: list[int]) -> int:
    import mlx.core as mx
    x = mx.array([input_ids])
    logits = model(x)
    next_logits = logits[0, -1, :]
    return int(mx.argmax(next_logits).item())


def get_token_logprob(model, tokenizer, input_ids: list[int], target_id: int) -> float:
    import mlx.core as mx
    x = mx.array([input_ids])
    logits = model(x)
    next_logits = logits[0, -1, :]
    log_probs = mx.log(mx.softmax(next_logits, axis=-1))
    return float(log_probs[target_id].item())


def get_token_rank(model, tokenizer, input_ids: list[int], target_id: int) -> int:
    import mlx.core as mx
    x = mx.array([input_ids])
    logits = model(x)
    next_logits = logits[0, -1, :]
    log_probs = mx.log(mx.softmax(next_logits, axis=-1))
    return int(mx.sum(log_probs > log_probs[target_id]).item())


def load_m1_model(base_path: str, adapter_path: str):
    from mlx_lm import load as mlx_load
    from scripts.train_sft import apply_lora, resolve_init_adapter_file
    import mlx.core as mx

    model, tokenizer = mlx_load(base_path)
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
    parser = argparse.ArgumentParser(description="Build binary prefix DPO control data")
    parser.add_argument("--probe-data", default="data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl")
    parser.add_argument("--output", default="data/math/pilots/binary_prefix_dpo_control_480.jsonl")
    parser.add_argument("--repetitions", type=int, default=16)
    args = parser.parse_args()

    import mlx.core as mx

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    samples = load_probe_samples(args.probe_data, 30)

    print("Loading M1 model...")
    m1_model, tokenizer = load_m1_model(
        "models/qwen3_0_6b/mlx",
        "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
    )

    # Encode the chosen token "Solution" and verify it's a single token
    chosen_text = "Solution"
    chosen_ids = tokenizer.encode(chosen_text)
    assert len(chosen_ids) == 1, f"'{chosen_text}' encodes to {len(chosen_ids)} tokens: {chosen_ids}"
    chosen_token_id = chosen_ids[0]
    print(f"Chosen token: '{chosen_text}' → id={chosen_token_id}")

    # Build the common continuation after the first token
    # "Solution:\n1. " is the minimal prefix that establishes the style
    continuation_text = ":\n1. "
    continuation_ids = tokenizer.encode(continuation_text)
    print(f"Continuation: '{continuation_text}' → ids={continuation_ids}")

    # The full chosen sequence after <think>\n: "Solution:\n1. "
    chosen_full_text = chosen_text + continuation_text
    chosen_full_ids = tokenizer.encode(chosen_full_text)
    # Verify: chosen_full_ids should == [chosen_token_id] + continuation_ids
    assert chosen_full_ids == [chosen_token_id] + continuation_ids, (
        f"Re-encoding mismatch: {chosen_full_ids} != {[chosen_token_id] + continuation_ids}"
    )

    print(f"\nBuilding binary prefix pairs...")
    all_pairs = []

    for i, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["problem"]},
        ]
        prompt_text = build_prompt_text(tokenizer, messages)
        prompt_ids = tokenizer.encode(prompt_text)

        # Append <think>\n to reach the branch point
        think_text = "<think>\n"
        think_ids = tokenizer.encode(think_text)
        branch_ids = prompt_ids + think_ids

        # Get M1's greedy token at the branch point
        greedy_id = get_greedy_token_id(m1_model, tokenizer, branch_ids)
        greedy_text = tokenizer.decode([greedy_id])

        # Verify chosen != greedy
        assert greedy_id != chosen_token_id, (
            f"Sample {i}: M1 greedy is already '{chosen_text}' (id={chosen_token_id})"
        )

        # Get M1 logprobs and ranks for both tokens
        chosen_lp = get_token_logprob(m1_model, tokenizer, branch_ids, chosen_token_id)
        chosen_rank = get_token_rank(m1_model, tokenizer, branch_ids, chosen_token_id)
        greedy_lp = get_token_logprob(m1_model, tokenizer, branch_ids, greedy_id)
        greedy_rank = get_token_rank(m1_model, tokenizer, branch_ids, greedy_id)

        # Build chosen response: <think>\n + "Solution:\n1. " (then model generates)
        chosen_response = think_text + chosen_full_text
        chosen_response_ids = tokenizer.encode(chosen_response)
        # Verify re-encoding
        assert chosen_response_ids == think_ids + chosen_full_ids, (
            f"Chosen re-encoding mismatch for {sample['problem_id']}"
        )

        # Build rejected response: <think>\n + M1_greedy_token (then model generates)
        # We need a "completion" for rejected too. Use the greedy token + same continuation
        # to keep sequences identical after the first token.
        rejected_response = think_text + greedy_text + continuation_text
        rejected_response_ids = tokenizer.encode(rejected_response)
        # Verify: rejected_ids == think_ids + [greedy_id] + continuation_ids
        assert rejected_response_ids == think_ids + [greedy_id] + continuation_ids, (
            f"Rejected re-encoding mismatch for {sample['problem_id']}: "
            f"{rejected_response_ids} != {think_ids + [greedy_id] + continuation_ids}"
        )

        pair = {
            "problem_id": sample["problem_id"],
            "messages": messages,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "answer": sample["answer"],
            "pair_type": "binary_prefix_control",
            "metadata": {
                "problem_id": sample["problem_id"],
                "source": "binary_prefix_dpo_control",
                "chosen_token_id": chosen_token_id,
                "chosen_token_text": chosen_text,
                "rejected_token_id": greedy_id,
                "rejected_token_text": greedy_text,
                "chosen_token_logprob_m1": round(chosen_lp, 4),
                "chosen_token_rank_m1": chosen_rank,
                "rejected_token_logprob_m1": round(greedy_lp, 4),
                "rejected_token_rank_m1": greedy_rank,
                "logprob_gap_m1": round(chosen_lp - greedy_lp, 4),
                "chosen_response_token_ids": chosen_response_ids,
                "rejected_response_token_ids": rejected_response_ids,
                "re_encoding_consistent": True,
            },
        }
        all_pairs.append(pair)

        print(f"  [{i+1:2d}/30] {sample['problem_id']}: "
              f"chosen='{chosen_text}'(id={chosen_token_id}, lp={chosen_lp:.1f}, rank={chosen_rank})  "
              f"rejected='{greedy_text}'(id={greedy_id}, lp={greedy_lp:.1f}, rank={greedy_rank})  "
              f"gap={chosen_lp - greedy_lp:.1f}")

    # Repeat 16× to get 480 pairs
    repeated = []
    for rep in range(args.repetitions):
        for pair in all_pairs:
            p = dict(pair)
            p["metadata"] = dict(pair["metadata"])
            p["metadata"]["repetition"] = rep
            repeated.append(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in repeated:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nCreated {len(repeated)} pairs ({len(all_pairs)} × {args.repetitions}) → {output_path}")

    # Verify output
    with open(output_path, encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    assert count == len(repeated), f"Output count mismatch: {count} != {len(repeated)}"
    print(f"Verified: {count} lines in output")


if __name__ == "__main__":
    main()
