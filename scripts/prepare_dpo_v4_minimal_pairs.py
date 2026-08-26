"""
Prepare DPO v4 minimal contrast pairs from DPO v2 training data.

Chosen: kept as-is (structured template with Solution:/numbered steps/Final:)
Rejected: derived from chosen by stripping ONLY the style wrapper:
  - Remove "Solution:\n" prefix
  - Remove "N. " step number prefixes
  - Replace "Final: The answer is \\boxed{X}." with "\\boxed{X}"
  - Reasoning text and boxed answer are byte-identical

Input:  data/math/splits/dpo_v2_style_train_449.jsonl
Output: data/math/pilots/dpo_v4_minimal_pilot_30.jsonl (first 30 for review)
        data/math/pilots/dpo_v4_minimal_449.jsonl (full set, after review)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _extract_boxed(text: str) -> str | None:
    """Extract content from last \\boxed{...} using brace-depth parsing."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start:pos - 1]


def strip_style_wrapper(chosen: str) -> str:
    """Derive rejected from chosen by removing style wrapper only.

    Keeps reasoning content and boxed answer byte-identical.
    """
    text = chosen

    # 1. Extract the boxed answer
    boxed_content = _extract_boxed(text)
    if boxed_content is None:
        raise ValueError(f"Cannot extract \\boxed{{}} from: {text[:100]}")

    # 2. Extract thinking block content (between <think> and </think>)
    think_match = re.search(r"<think>\s*\n(.*?)\s*</think>", text, re.DOTALL)
    if not think_match:
        raise ValueError(f"Cannot find <think> block in: {text[:100]}")
    think_body = think_match.group(1)

    # 3. Remove "Solution:\n" prefix if present
    think_body = re.sub(r"^Solution:\s*\n", "", think_body)

    # 4. Remove step number prefixes (e.g., "1. ", "2. ", "12. ")
    lines = think_body.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"^\d+\.\s*", "", line)
        cleaned_lines.append(cleaned)
    reasoning = "\n".join(cleaned_lines).strip()

    # 5. Reconstruct rejected
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{boxed_content}}}"


def convert_record(record: dict) -> dict:
    """Convert one DPO v2 record to v4 minimal pair."""
    meta = record.get("metadata", {}) or {}
    problem_id = meta.get("problem_id") or record.get("problem_id", "")
    if not problem_id or not str(problem_id).strip():
        raise ValueError("problem_id missing")

    chosen = record.get("chosen", "")
    if not chosen or not isinstance(chosen, str):
        raise ValueError(f"chosen missing or invalid for {problem_id}")

    messages = record.get("messages", [])
    answer = record.get("answer", "")

    rejected = strip_style_wrapper(chosen)

    # Verify content identity
    chosen_boxed = _extract_boxed(chosen)
    rejected_boxed = _extract_boxed(rejected)
    if chosen_boxed != rejected_boxed:
        raise ValueError(
            f"Boxed mismatch for {problem_id}: chosen={chosen_boxed!r} vs rejected={rejected_boxed!r}"
        )

    return {
        "problem_id": problem_id,
        "messages": messages,
        "chosen": chosen,
        "rejected": rejected,
        "answer": answer,
        "pair_type": "dpo_v4_minimal",
        "metadata": {
            **meta,
            "source": "dpo_v4_minimal",
            "chosen_style": "structured",
            "rejected_style": "free_form",
            "content_identical": True,
            "style_only_diff": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO v4 minimal contrast pairs")
    parser.add_argument("--input", default="data/math/splits/dpo_v2_style_train_449.jsonl")
    parser.add_argument("--pilot-output", default="data/math/pilots/dpo_v4_minimal_pilot_30.jsonl")
    parser.add_argument("--full-output", default="data/math/pilots/dpo_v4_minimal_449.jsonl")
    parser.add_argument("--pilot-count", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)
    pilot_path = Path(args.pilot_output)
    full_path = Path(args.full_output)

    if pilot_path.exists():
        raise FileExistsError(f"Pilot output already exists: {pilot_path}")
    if full_path.exists():
        raise FileExistsError(f"Full output already exists: {full_path}")

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    pairs = [convert_record(r) for r in records]

    # Write pilot (first 30)
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pilot_path, "w", encoding="utf-8") as f:
        for p in pairs[:args.pilot_count]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write full set
    with open(full_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Pilot ({args.pilot_count} pairs) → {pilot_path}")
    print(f"Full ({len(pairs)} pairs) → {full_path}")

    # Show first pair for review
    p0 = pairs[0]
    print(f"\n=== Sample: {p0['problem_id']} ===")
    print(f"Chosen:\n{p0['chosen']}")
    print(f"\nRejected:\n{p0['rejected']}")
    print(f"\nBoxed match: {_extract_boxed(p0['chosen'])} == {_extract_boxed(p0['rejected'])}")


if __name__ == "__main__":
    main()
