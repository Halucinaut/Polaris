"""
Build boundary-only DPO control data.

Response = single divergence token only (no continuation).
Common prefix "<think>\n" is part of prompt, not response.
Chosen token = "Solution", Rejected = M1's greedy token.

Also builds single-token SFT control: response = "Solution" only.

Usage:
    ./.venv/bin/python scripts/build_boundary_only_dpo_data.py
"""

from __future__ import annotations

import argparse
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


def get_token_logprob_and_rank(model, tokenizer, input_ids: list[int], target_id: int) -> tuple[float, int]:
    import mlx.core as mx
    x = mx.array([input_ids])
    logits = model(x)
    next_logits = logits[0, -1, :]
    log_probs = mx.log(mx.softmax(next_logits, axis=-1))
    lp = float(log_probs[target_id].item())
    rank = int(mx.sum(log_probs > log_probs[target_id]).item())
    return lp, rank


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-data", default="data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl")
    parser.add_argument("--dpo-output", default="data/math/pilots/boundary_only_dpo_480.jsonl")
    parser.add_argument("--sft-output", default="data/math/pilots/boundary_only_sft_480.jsonl")
    parser.add_argument("--repetitions", type=int, default=16)
    args = parser.parse_args()

    for p in [args.dpo_output, args.sft_output]:
        if Path(p).exists():
            raise FileExistsError(f"Output already exists: {p}")

    samples = load_probe_samples(args.probe_data, 30)

    print("Loading M1 model...")
    m1_model, tokenizer = load_m1_model(
        "models/qwen3_0_6b/mlx",
        "runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final",
    )

    chosen_text = "Solution"
    chosen_ids = tokenizer.encode(chosen_text)
    assert len(chosen_ids) == 1, f"'{chosen_text}' → {len(chosen_ids)} tokens"
    chosen_id = chosen_ids[0]

    # The response is JUST the single token. Prompt includes <think>\n.
    think_text = "<think>\n"
    think_ids = tokenizer.encode(think_text)

    print(f"Chosen token: '{chosen_text}' → id={chosen_id}")
    print(f"Think prefix: '{think_text}' → ids={think_ids} (part of prompt)")

    dpo_pairs = []
    sft_records = []

    for i, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["problem"]},
        ]
        prompt_text = build_prompt_text(tokenizer, messages)
        # Prompt includes <think>\n — the response is just the single token
        full_prompt_ids = tokenizer.encode(prompt_text) + think_ids

        greedy_id = get_greedy_token_id(m1_model, tokenizer, full_prompt_ids)
        greedy_text = tokenizer.decode([greedy_id])
        assert greedy_id != chosen_id, f"{sample['problem_id']}: greedy == chosen"

        chosen_lp, chosen_rank = get_token_logprob_and_rank(m1_model, tokenizer, full_prompt_ids, chosen_id)
        greedy_lp, greedy_rank = get_token_logprob_and_rank(m1_model, tokenizer, full_prompt_ids, greedy_id)

        # DPO pair: response = single token, prompt_suffix = "<think>\n"
        dpo_pairs.append({
            "problem_id": sample["problem_id"],
            "messages": messages,
            "chosen": chosen_text,
            "rejected": greedy_text,
            "answer": sample["answer"],
            "pair_type": "boundary_only_dpo",
            "prompt_suffix": "<think>\n",
            "no_eos": True,
            "metadata": {
                "problem_id": sample["problem_id"],
                "source": "boundary_only_dpo",
                "think_included_in_prompt": True,
                "chosen_token_id": chosen_id,
                "chosen_token_text": chosen_text,
                "rejected_token_id": greedy_id,
                "rejected_token_text": greedy_text,
                "chosen_token_logprob_m1": round(chosen_lp, 4),
                "chosen_token_rank_m1": chosen_rank,
                "rejected_token_logprob_m1": round(greedy_lp, 4),
                "rejected_token_rank_m1": greedy_rank,
                "logprob_gap_m1": round(chosen_lp - greedy_lp, 4),
                "response_token_ids": [chosen_id],
                "re_encoding_consistent": True,
            },
        })

        # SFT record: target = single token, prompt_suffix = "<think>\n"
        sft_records.append({
            "messages": messages,
            "target": chosen_text,
            "prompt_suffix": "<think>\n",
            "no_eos": True,
            "metadata": {
                "problem_id": sample["problem_id"],
                "source": "boundary_only_sft",
                "chosen_token_id": chosen_id,
                "think_included_in_prompt": True,
            },
        })

        print(f"  [{i+1:2d}/30] {sample['problem_id']}: "
              f"chosen='{chosen_text}'(lp={chosen_lp:.1f},rank={chosen_rank})  "
              f"rejected='{greedy_text}'(lp={greedy_lp:.1f},rank={greedy_rank})  "
              f"gap={chosen_lp - greedy_lp:.1f}")

    # Repeat
    dpo_repeated = []
    sft_repeated = []
    for rep in range(args.repetitions):
        for p in dpo_pairs:
            r = dict(p)
            r["metadata"] = dict(p["metadata"])
            r["metadata"]["repetition"] = rep
            dpo_repeated.append(r)
        for s in sft_records:
            r = dict(s)
            r["metadata"] = dict(s["metadata"])
            r["metadata"]["repetition"] = rep
            sft_repeated.append(r)

    # Write DPO
    Path(args.dpo_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dpo_output, "w", encoding="utf-8") as f:
        for p in dpo_repeated:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write SFT
    with open(args.sft_output, "w", encoding="utf-8") as f:
        for s in sft_repeated:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDPO: {len(dpo_repeated)} pairs → {args.dpo_output}")
    print(f"SFT: {len(sft_repeated)} records → {args.sft_output}")

    # Verify
    for path, expected in [(args.dpo_output, 480), (args.sft_output, 480)]:
        with open(path, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        assert count == expected, f"{path}: {count} != {expected}"
    print("Verified: 480 lines each")


if __name__ == "__main__":
    main()
