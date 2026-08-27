#!/usr/bin/env python3
"""
Reasoning quality DPO pilot: candidate generation.

Loads M1 adapter (SFT GSM8K 500) and generates 4 candidates per problem
with temperature sampling.  Only candidates whose extracted answer matches
gold are kept for downstream pairing.

Requires MLX.  Run under .venv:
    .venv/bin/python scripts/generate_reasoning_quality_candidates.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

# Import eval_math helpers
_scripts_dir = PROJECT_ROOT / "scripts"
_spec = importlib.util.spec_from_file_location("eval_math", _scripts_dir / "eval_math.py")
_eval_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_math)

extract_predicted_answer = _eval_math.extract_predicted_answer
answers_match = _eval_math.answers_match

# Import load_model helpers
_smoke_dir = _scripts_dir / "smoke"
_spec2 = importlib.util.spec_from_file_location("load_model", _smoke_dir / "load_model.py")
_load_model = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_load_model)

apply_chat_template_safe = _load_model.apply_chat_template_safe

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem and put the final answer in \\boxed{}."
)
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reasoning quality candidates")
    parser.add_argument("--model-path", default="models/qwen3_0_6b/mlx")
    parser.add_argument("--adapter-path", default="runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final")
    parser.add_argument("--problems", default="data/math/pilots/dpo_reasoning_quality_problems_50.jsonl")
    parser.add_argument("--output", default="data/math/pilots/dpo_reasoning_quality_candidates_50.jsonl")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--base-seed", type=int, default=20260827)
    args = parser.parse_args()

    problems = load_jsonl(Path(args.problems))
    print(f"Loaded {len(problems)} problems")

    print("Loading model + adapter...")
    tic = time.perf_counter()
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)
    print(f"Model loaded in {time.perf_counter() - tic:.1f}s")

    sampler = make_sampler(args.temperature)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_candidates = 0
    total_correct = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for pidx, prob in enumerate(problems):
            problem_id = prob["problem_id"]
            problem_text = prob["problem"]
            gold_answer = str(prob["answer"])

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem_text},
            ]
            rendered = apply_chat_template_safe(tokenizer, messages)

            for ci in range(args.num_candidates):
                seed_i = args.base_seed + pidx * 100 + ci
                rng = random.Random(seed_i)

                tic = time.perf_counter()
                raw_output = generate(
                    model, tokenizer,
                    prompt=rendered,
                    max_tokens=args.max_new_tokens,
                    sampler=sampler,
                )
                gen_time = time.perf_counter() - tic

                predicted, method = extract_predicted_answer(raw_output)
                is_correct = answers_match(predicted, gold_answer)

                total_candidates += 1
                if is_correct:
                    total_correct += 1

                record = {
                    "problem_id": problem_id,
                    "candidate_idx": ci,
                    "seed": seed_i,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "raw_output": raw_output,
                    "predicted_answer": predicted,
                    "extraction_method": method,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct,
                    "generation_time_sec": round(gen_time, 2),
                    "problem": problem_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                status = "OK" if is_correct else "WRONG"
                print(f"  [{pidx+1}/{len(problems)}] {problem_id} cand={ci} {status} ({gen_time:.1f}s)")

    print(f"\nDone: {total_candidates} candidates, {total_correct} correct ({total_correct/total_candidates:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
