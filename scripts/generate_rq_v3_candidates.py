#!/usr/bin/env python3
"""
Reasoning quality DPO pilot v3: candidate generation.

v3 improvements over v2:
- problem_hash() as canonical identity (cross-file dedup)
- mx.random.seed(seed_i) set before EVERY generation (full determinism)
- Zero-overlap assertion against Probe-30 / Stress-50

Requires MLX.  Run under .venv:
    .venv/bin/python scripts/generate_rq_v3_candidates.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from polaris.problem_hash import (
    assert_no_overlap,
    build_exclude_hashes,
    problem_hash,
)

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


def extract_gsm8k_answer(raw_answer: str) -> str:
    m = re.search(r"####\s*(.+)", raw_answer)
    return m.group(1).strip() if m else raw_answer.strip()


def load_gsm8k_train(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append({
                "problem_index": i,
                "problem": r["question"],
                "answer": extract_gsm8k_answer(r["answer"]),
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate rq v3 candidates")
    parser.add_argument("--model-path", default="models/qwen3_0_6b/mlx")
    parser.add_argument("--adapter-path", default="runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final")
    parser.add_argument("--gsm8k-train", default="data/math/gsm8k/train.jsonl")
    parser.add_argument("--output-dir", default="data/math/pilots/dpo_rq_v3")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--calibration", action="store_true", help="Run determinism calibration on 3 problems")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build exclude hashes
    exclude_hashes = build_exclude_hashes([
        PROJECT_ROOT / "data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl",
        PROJECT_ROOT / "data/math/splits/dpo_v2_style_stress_eval_50.jsonl",
    ])
    print(f"Exclude hashes: {len(exclude_hashes)}")

    # Also exclude v2 pilot problem hashes
    v2_hash_set: set[str] = set()
    for v2_path in [
        PROJECT_ROOT / "data/math/pilots/dpo_rq_v2_problems_50.jsonl",
        PROJECT_ROOT / "data/math/pilots/dpo_rq_v2b_problems_100.jsonl",
        PROJECT_ROOT / "data/math/pilots/dpo_rq_v2c_problems_100.jsonl",
    ]:
        if v2_path.exists():
            with v2_path.open() as f:
                for line in f:
                    r = json.loads(line)
                    v2_hash_set.add(problem_hash(r["problem"]))

    all_exclude = exclude_hashes | v2_hash_set

    # Load and filter problems
    all_problems = load_gsm8k_train(Path(args.gsm8k_train))

    # Filter: problem_hash not in exclude
    eligible = []
    for p in all_problems:
        h = problem_hash(p["problem"])
        if h not in all_exclude:
            p["phash"] = h
            eligible.append(p)

    print(f"Total: {len(all_problems)}, excluded: {len(all_problems) - len(eligible)}, eligible: {len(eligible)}")

    # Assert zero overlap
    eligible_hashes = {p["phash"] for p in eligible}
    assert_no_overlap(eligible_hashes, exclude_hashes, "eligible")

    # Sample
    rng = random.Random(args.base_seed)
    if args.calibration:
        sampled = eligible[:3]
    else:
        sampled = rng.sample(eligible, args.num_problems)
    sampled.sort(key=lambda x: x["phash"])
    print(f"Sampled: {len(sampled)} problems")

    # Save problems
    problems_path = out_dir / "problems.jsonl"
    with problems_path.open("w", encoding="utf-8") as f:
        for p in sampled:
            f.write(json.dumps({
                "phash": p["phash"],
                "problem": p["problem"],
                "answer": p["answer"],
                "problem_index": p["problem_index"],
            }, ensure_ascii=False) + "\n")

    # Load model
    print("Loading model + adapter...")
    tic = time.perf_counter()
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)
    print(f"Model loaded in {time.perf_counter() - tic:.1f}s")

    sampler = make_sampler(args.temperature)
    candidates_path = out_dir / "candidates.jsonl"

    total_candidates = 0
    total_correct = 0
    candidate_hashes: set[str] = set()

    with candidates_path.open("w", encoding="utf-8") as fout:
        for pidx, prob in enumerate(sampled):
            phash = prob["phash"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            rendered = apply_chat_template_safe(tokenizer, messages)

            for ci in range(args.num_candidates):
                seed_i = args.base_seed + pidx * 1000 + ci

                # KEY: set MLX seed before each generation
                mx.random.seed(seed_i)

                tic = time.perf_counter()
                raw_output = generate(
                    model, tokenizer,
                    prompt=rendered,
                    max_tokens=args.max_new_tokens,
                    sampler=sampler,
                )
                gen_time = time.perf_counter() - tic

                predicted, method = extract_predicted_answer(raw_output)
                is_correct = answers_match(predicted, prob["answer"])

                total_candidates += 1
                if is_correct:
                    total_correct += 1

                candidate_hashes.add(phash)

                record = {
                    "phash": phash,
                    "candidate_idx": ci,
                    "seed": seed_i,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "raw_output": raw_output,
                    "output_hash": problem_hash(raw_output),
                    "predicted_answer": predicted,
                    "extraction_method": method,
                    "gold_answer": prob["answer"],
                    "is_correct": is_correct,
                    "generation_time_sec": round(gen_time, 2),
                    "problem": prob["problem"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                status = "OK" if is_correct else "WRONG"
                print(f"  [{pidx+1}/{len(sampled)}] {phash[:8]}.. cand={ci} {status} ({gen_time:.1f}s)")

    # Assert no overlap with exclude set
    assert_no_overlap(candidate_hashes, exclude_hashes, "generated candidates")

    print(f"\nDone: {total_candidates} candidates, {total_correct} correct ({total_correct/total_candidates:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
