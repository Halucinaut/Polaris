#!/usr/bin/env python3
"""
Reasoning quality DPO pilot: pair formation and Claude blind review preparation.

Reads candidates, forms C(n,2) pairs from correct-answer candidates,
anonymizes as A/B, and prepares review batches for Claude scoring.

No external API calls. No training. No modification of existing data.
"""

from __future__ import annotations

import json
import random
import hashlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def form_pairs(candidates_path: Path, seed: int = 20260827) -> list[dict]:
    """Form all C(n,2) pairs from correct candidates per problem."""
    # Group by problem_id, keep only correct candidates
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for rec in load_jsonl(candidates_path):
        if rec["is_correct"]:
            by_problem[rec["problem_id"]].append(rec)

    rng = random.Random(seed)
    pairs: list[dict] = []

    for problem_id, cands in sorted(by_problem.items()):
        if len(cands) < 2:
            continue
        for c1, c2 in combinations(cands, 2):
            # Random A/B assignment
            if rng.random() < 0.5:
                a_cand, b_cand = c1, c2
            else:
                a_cand, b_cand = c2, c1

            pair_id = hashlib.sha256(
                f"{problem_id}:{c1['candidate_idx']}:{c2['candidate_idx']}".encode()
            ).hexdigest()[:12]

            pairs.append({
                "pair_id": f"rq_{pair_id}",
                "problem_id": problem_id,
                "problem": a_cand["problem"],
                "gold_answer": a_cand["gold_answer"],
                "candidate_a": {
                    "candidate_idx": a_cand["candidate_idx"],
                    "seed": a_cand["seed"],
                    "raw_output": a_cand["raw_output"],
                    "predicted_answer": a_cand["predicted_answer"],
                    "extraction_method": a_cand["extraction_method"],
                },
                "candidate_b": {
                    "candidate_idx": b_cand["candidate_idx"],
                    "seed": b_cand["seed"],
                    "raw_output": b_cand["raw_output"],
                    "predicted_answer": b_cand["predicted_answer"],
                    "extraction_method": b_cand["extraction_method"],
                },
                # True mapping (for later use, not shown to reviewer)
                "_mapping": {
                    "a_is_c1": a_cand is c1,
                    "c1_idx": c1["candidate_idx"],
                    "c2_idx": c2["candidate_idx"],
                },
            })

    rng.shuffle(pairs)
    return pairs


def format_review_prompt(pair: dict) -> str:
    """Format a pair for Claude review."""
    # Extract think content and post-think text for readability
    def extract_parts(raw: str) -> tuple[str, str]:
        import re
        m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        think = m.group(1).strip() if m else ""
        post = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return think, post

    a_think, a_post = extract_parts(pair["candidate_a"]["raw_output"])
    b_think, b_post = extract_parts(pair["candidate_b"]["raw_output"])

    return f"""## Problem
{pair['problem']}

**Gold answer:** {pair['gold_answer']}

---

## Candidate A

**Reasoning:**
{a_think}

**Final answer section:**
{a_post}

---

## Candidate B

**Reasoning:**
{b_think}

**Final answer section:**
{b_post}

---

## Scoring Criteria (0-2 each)

1. **Completeness**: Does it cover all problem conditions and intermediate quantities?
2. **Logic**: Any skipped steps, contradictions, or unjustified conclusions?
3. **Efficiency**: Any无效重复、冗余计算?
4. **Clarity**: Is the step order clear and logical?

## Your Task

1. Score A and B on each criterion (0-2)
2. Compute total for A and B
3. Choose the better candidate (A or B), or "tie" if genuinely equal
4. Explain your reasoning in 1-2 sentences

Respond in this exact JSON format:
```json
{{
  "a_scores": [completeness, logic, efficiency, clarity],
  "b_scores": [completeness, logic, efficiency, clarity],
  "a_total": <sum>,
  "b_total": <sum>,
  "choice": "A" | "B" | "tie",
  "reasoning": "<1-2 sentences>"
}}
```"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/math/pilots/dpo_reasoning_quality_candidates_50.jsonl")
    parser.add_argument("--output-pairs", default="data/math/pilots/dpo_reasoning_quality_all_pairs.jsonl")
    parser.add_argument("--output-review", default="data/math/pilots/dpo_reasoning_quality_review_batch.jsonl")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    pairs = form_pairs(Path(args.candidates))
    print(f"Formed {len(pairs)} pairs from correct candidates")

    # Save all pairs
    out_pairs = Path(args.output_pairs)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved all pairs to {out_pairs}")

    # Prepare first review batch
    batch = pairs[:args.batch_size]
    out_review = Path(args.output_review)
    with out_review.open("w", encoding="utf-8") as f:
        for p in batch:
            review_rec = {
                "pair_id": p["pair_id"],
                "problem_id": p["problem_id"],
                "review_prompt": format_review_prompt(p),
            }
            f.write(json.dumps(review_rec, ensure_ascii=False) + "\n")
    print(f"Saved first {len(batch)} review prompts to {out_review}")


if __name__ == "__main__":
    main()
