#!/usr/bin/env python3
"""
Batch review of reasoning quality pairs.
Processes pairs from stdin, outputs scored reviews to stdout.
"""

import json
import re
import sys


def extract_parts(raw: str) -> tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    think = m.group(1).strip() if m else ""
    post = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return think, post


def format_pair_for_review(pair: dict) -> str:
    a_think, a_post = extract_parts(pair["candidate_a"]["raw_output"])
    b_think, b_post = extract_parts(pair["candidate_b"]["raw_output"])
    return f"""Problem: {pair['problem']}
Gold answer: {pair['gold_answer']}

--- Candidate A ---
{a_think}
[Final section]: {a_post}

--- Candidate B ---
{b_think}
[Final section]: {b_post}"""


def main():
    pairs = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))

    for pair in pairs:
        formatted = format_pair_for_review(pair)
        print(json.dumps({
            "pair_id": pair["pair_id"],
            "problem_id": pair["problem_id"],
            "formatted": formatted,
            "gold_answer": pair["gold_answer"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
