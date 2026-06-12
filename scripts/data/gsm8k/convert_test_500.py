#!/usr/bin/env python3
"""Convert first 500 samples from full GSM8K test set into D3-compatible schema."""
import json
from pathlib import Path

SPLIT_DIR = Path("data/math/gsm8k/split")
SOURCE_PATH = Path("data/math/gsm8k/test.jsonl")
OUT_PATH = SPLIT_DIR / "test_converted_500.jsonl"

def extract_final_answer(answer_text: str) -> str:
    import re
    match = re.search(r"####\s*(.+)$", answer_text.strip(), re.MULTILINE)
    return match.group(1).strip() if match else ""

def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    selected = records[:500]
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for idx, raw in enumerate(selected, start=1):
            converted = {
                "problem_id": f"gsm8k_test_{idx:04d}",
                "problem": raw["question"],
                "answer": extract_final_answer(raw["answer"]),
                "solution": raw["answer"],
                "source": "gsm8k",
                "domain": "grade_school_math",
                "split": "test",
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} records to {OUT_PATH}")

if __name__ == "__main__":
    main()
