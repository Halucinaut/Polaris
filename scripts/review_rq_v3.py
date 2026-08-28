#!/usr/bin/env python3
"""
Reasoning quality DPO pilot v3: pair formation and classified review.

Review labels (per candidate):
1. condition_omission — 遗漏题目条件或中间量
2. logic_error — 跳步、矛盾、无依据结论
3. unit_error — 单位转换错误、量纲不匹配
4. redundancy — 重复计算、措辞不当、拼写错误

Main training set: pairs where chosen优于rejected in categories 1-3
Auxiliary list: pairs with only category 4 differences
"""

from __future__ import annotations

import json
import random
import hashlib
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

# Review categories — order matters
REVIEW_CATEGORIES = [
    "condition_omission",  # 1: 条件遗漏
    "logic_error",         # 2: 逻辑错误
    "unit_error",          # 3: 单位/量纲错误
    "redundancy",          # 4: 冗余/表达
]

# Categories that qualify for main training set
MAIN_CATEGORIES = {"condition_omission", "logic_error", "unit_error"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def form_best_pair_per_problem(
    candidates_path: Path,
    seed: int = 20260827,
) -> list[dict]:
    """Group by phash, keep correct candidates, select most diverse pair per problem."""
    by_phash: dict[str, list[dict]] = defaultdict(list)
    for rec in load_jsonl(candidates_path):
        if rec["is_correct"]:
            by_phash[rec["phash"]].append(rec)

    rng = random.Random(seed)
    pairs = []

    for phash, cands in sorted(by_phash.items()):
        if len(cands) < 2:
            continue

        # Select pair with lowest similarity (most different reasoning)
        best_pair = None
        best_score = -1
        for c1, c2 in combinations(cands, 2):
            sim = SequenceMatcher(None, c1["raw_output"], c2["raw_output"]).ratio()
            score = 1.0 - sim  # prefer low similarity
            if score > best_score:
                best_score = score
                best_pair = (c1, c2, sim)

        c1, c2, sim = best_pair
        if rng.random() < 0.5:
            a, b = c1, c2
        else:
            a, b = c2, c1

        pair_id = hashlib.sha256(f"{phash}:best".encode()).hexdigest()[:12]
        pairs.append({
            "pair_id": f"rq3_{pair_id}",
            "phash": phash,
            "problem": a["problem"],
            "gold_answer": a["gold_answer"],
            "candidate_a": {
                "candidate_idx": a["candidate_idx"],
                "seed": a["seed"],
                "raw_output": a["raw_output"],
            },
            "candidate_b": {
                "candidate_idx": b["candidate_idx"],
                "seed": b["seed"],
                "raw_output": b["raw_output"],
            },
            "similarity": round(sim, 3),
        })

    rng.shuffle(pairs)
    return pairs


def format_review_prompt(pair: dict) -> str:
    """Format pair for Claude review with 4-category scoring."""
    import re

    def extract_parts(raw: str) -> tuple[str, str]:
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

**Final answer:**
{a_post}

---

## Candidate B

**Reasoning:**
{b_think}

**Final answer:**
{b_post}

---

## Scoring — 4 categories (0=none, 1=minor, 2=major)

For EACH candidate (A and B), score independently:

1. **条件遗漏** (condition_omission): 遗漏题目条件或中间量?
2. **逻辑错误** (logic_error): 跳步、矛盾、无依据结论?
3. **单位/量纲错误** (unit_error): 单位转换错误、量纲不匹配?
4. **冗余/表达** (redundancy): 重复计算、措辞不当、拼写错误?

## Output format (JSON)

```json
{{
  "a_issues": {{"condition_omission": 0, "logic_error": 0, "unit_error": 0, "redundancy": 0}},
  "b_issues": {{"condition_omission": 0, "logic_error": 0, "unit_error": 0, "redundancy": 0}},
  "choice": "A" | "B" | "tie",
  "reasoning": "1-2 sentences explaining the key difference"
}}
```

Rules:
- "choice" = which candidate has FEWER issues in categories 1-3 (condition/logic/unit)
- If categories 1-3 are tied, consider category 4 (redundancy)
- If truly tied, mark "tie"
- Score each candidate INDEPENDENTLY — don't just mirror"""


def classify_pair(review: dict) -> str:
    """Classify a review result as 'main', 'auxiliary', or 'tie'."""
    choice = review.get("choice", "tie")
    if choice == "tie":
        return "tie"

    a_issues = review.get("a_issues", {})
    b_issues = review.get("b_issues", {})

    # Count main-category issues (1-3) for each candidate
    a_main = sum(a_issues.get(c, 0) for c in MAIN_CATEGORIES)
    b_main = sum(b_issues.get(c, 0) for c in MAIN_CATEGORIES)

    # Count all issues
    a_all = sum(a_issues.values())
    b_all = sum(b_issues.values())

    if choice == "A":
        chosen_main, rejected_main = a_main, b_main
        chosen_all, rejected_all = a_all, b_all
    else:
        chosen_main, rejected_main = b_main, a_main
        chosen_all, rejected_all = b_all, a_all

    # Main: chosen has fewer main-category issues
    if chosen_main < rejected_main:
        return "main"
    # Auxiliary: only difference is in redundancy (category 4)
    elif chosen_main == rejected_main and chosen_all < rejected_all:
        return "auxiliary"
    else:
        return "tie"


def apply_mechanical_filters(pair: dict, review: dict) -> tuple[bool, str]:
    """Apply mechanical filters. Returns (pass, reason)."""
    choice = review.get("choice", "tie")
    if choice == "tie":
        return False, "tie"

    # Get chosen/rejected raw output
    if choice == "A":
        chosen_raw = pair["candidate_a"]["raw_output"]
        rejected_raw = pair["candidate_b"]["raw_output"]
    else:
        chosen_raw = pair["candidate_b"]["raw_output"]
        rejected_raw = pair["candidate_a"]["raw_output"]

    # Length ratio
    chosen_len = len(chosen_raw)
    rejected_len = len(rejected_raw)
    if rejected_len == 0:
        return False, "zero_rejected_length"
    ratio = chosen_len / rejected_len
    if ratio < 0.75 or ratio > 1.33:
        return False, f"length_ratio_{ratio:.2f}"

    # Similarity
    sim = SequenceMatcher(None, chosen_raw, rejected_raw).ratio()
    if sim > 0.95:
        return False, f"too_similar_{sim:.2f}"
    if sim < 0.1:
        return False, f"too_different_{sim:.2f}"

    return True, "passed"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/math/pilots/dpo_rq_v3/candidates.jsonl")
    parser.add_argument("--output-dir", default="data/math/pilots/dpo_rq_v3")
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Form pairs
    pairs = form_best_pair_per_problem(Path(args.candidates))
    print(f"Formed {len(pairs)} pairs (1 per problem)")

    # Save all pairs
    with (out_dir / "all_pairs.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Split into review batches
    batch_size = args.batch_size
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_num = i // batch_size + 1
        with (out_dir / f"review_batch_{batch_num}.jsonl").open("w") as f:
            for p in batch:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Review batches: {(len(pairs) + batch_size - 1) // batch_size}")


if __name__ == "__main__":
    main()
