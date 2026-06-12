#!/usr/bin/env python3
"""Build a deterministic converted GSM8K train slice."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def extract_final_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)$", answer_text.strip(), re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build converted GSM8K train slice")
    p.add_argument("--input", type=Path, default=Path("data/math/gsm8k/train.jsonl"))
    p.add_argument("--output", type=Path,
                   default=Path("data/math/gsm8k/split/train_converted_d5_500.jsonl"))
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    selected = samples[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx, raw in enumerate(selected, start=1):
            converted = {
                "problem_id": f"gsm8k_train_d5_{idx:04d}",
                "problem": raw["question"],
                "answer": extract_final_answer(raw["answer"]),
                "solution": raw["answer"],
                "source": "gsm8k",
                "domain": "grade_school_math",
                "split": "train",
            }
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
