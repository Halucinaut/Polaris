#!/usr/bin/env python3
"""
Convert D2-constrained GSM8K splits into M1 SFT data.

Input:  data/math/gsm8k/split/*_converted.jsonl  (D2 schema)
Output: data/math/splits/sft_v1.jsonl
        data/math/reports/sft_v1_data_report.json
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem and put the final answer in \\boxed{}."
)

# D2 source files to read (all splits)
INPUT_FILENAMES = [
    "train_converted.jsonl",
    "val_converted.jsonl",
    "test_converted.jsonl",
    "review_converted.jsonl",
]


# ---------------------------------------------------------------------------
# Solution cleaning
# ---------------------------------------------------------------------------

def clean_solution(solution: str) -> str:
    """Remove GSM8K calculation markers and the #### final-answer line."""
    cleaned = re.sub(r"<<[^>]*>>", "", solution)
    cleaned = re.sub(r"####\s*.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    return cleaned


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_target(solution_cleaned: str, answer: str) -> str:
    return f"<think>\n{solution_cleaned}\n</think>\n\n\\boxed{{{answer}}}"


# ---------------------------------------------------------------------------
# Answer extraction (mirrors eval_math.py logic)
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str):
    m = re.search(r"\\boxed\{(.*?)\}", text)
    return m.group(1) if m else None


def extract_hash_answer(text: str):
    m = re.search(r"####\s*(.+)", text)
    return m.group(1).strip() if m else None


def get_post_think_text(text: str) -> str:
    m = re.search(r"</think>(.*)", text, re.DOTALL)
    return m.group(1) if m else text


def normalize_answer(answer: str) -> str:
    return answer.strip().lower().replace(" ", "")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_target(target: str, expected_answer: str) -> dict:
    has_think = bool(re.search(r"<think>.*?</think>", target, re.DOTALL))
    post_think = get_post_think_text(target)
    extracted = extract_boxed_answer(post_think)
    format_valid = has_think and extracted is not None

    answer_match = False
    if extracted is not None:
        answer_match = normalize_answer(extracted) == normalize_answer(expected_answer)

    return {
        "format_valid": format_valid,
        "answer_extracted": extracted is not None,
        "answer_match": answer_match,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_samples(input_dir: Path, input_filenames: list[str] | None = None) -> list[dict]:
    samples = []
    for fname in (input_filenames or INPUT_FILENAMES):
        fpath = input_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fpath} not found")
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def convert(samples: list[dict]) -> list[dict]:
    records = []
    for s in samples:
        solution_cleaned = clean_solution(s["solution"])
        target = build_target(solution_cleaned, s["answer"])
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["problem"]},
            ],
            "target": target,
            "metadata": {
                "problem_id": s["problem_id"],
                "source": s["source"],
                "domain": s["domain"],
                "split": s["split"],
                "answer": s["answer"],
                "format": "think_tags_boxed_answer",
            },
        })
    return records


def build_report(records: list[dict]) -> dict:
    n = len(records)
    format_valid_count = 0
    answer_extract_ok = 0
    invalid_samples = []

    problem_chars = []
    solution_chars = []
    target_chars = []
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for r in records:
        meta = r["metadata"]
        target = r["target"]
        problem = r["messages"][1]["content"]

        # stats
        problem_chars.append(len(problem))
        target_chars.append(len(target))
        split_counts[meta["split"]] = split_counts.get(meta["split"], 0) + 1
        source_counts[meta["source"]] = source_counts.get(meta["source"], 0) + 1

        # we need the cleaned solution length; reconstruct from target
        think_match = re.search(r"<think>\n(.*?)\n</think>", target, re.DOTALL)
        if think_match:
            solution_chars.append(len(think_match.group(1)))

        # validation
        v = validate_target(target, meta["answer"])
        if v["format_valid"]:
            format_valid_count += 1
        if v["answer_extracted"]:
            answer_extract_ok += 1
        if not v["format_valid"] or not v["answer_match"]:
            invalid_samples.append({
                "problem_id": meta["problem_id"],
                "format_valid": v["format_valid"],
                "answer_extracted": v["answer_extracted"],
                "answer_match": v["answer_match"],
                "expected_answer": meta["answer"],
            })

    return {
        "num_samples": n,
        "format_valid_rate": round(format_valid_count / n, 6) if n else 0,
        "answer_extraction_success_rate": round(answer_extract_ok / n, 6) if n else 0,
        "avg_problem_chars": round(statistics.mean(problem_chars), 1) if problem_chars else 0,
        "avg_solution_chars": round(statistics.mean(solution_chars), 1) if solution_chars else 0,
        "avg_target_chars": round(statistics.mean(target_chars), 1) if target_chars else 0,
        "max_target_chars": max(target_chars) if target_chars else 0,
        "split_distribution": split_counts,
        "source_distribution": source_counts,
        "invalid_samples_count": len(invalid_samples),
        "examples_of_invalid_samples": invalid_samples[:10],
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert D2 GSM8K splits to M1 SFT data")
    p.add_argument("--input-dir", type=Path,
                   default=Path("data/math/gsm8k/split"))
    p.add_argument("--output", type=Path,
                   default=Path("data/math/splits/sft_v1.jsonl"))
    p.add_argument("--report", type=Path,
                   default=Path("data/math/reports/sft_v1_data_report.json"))
    p.add_argument("--input-files", nargs="*", default=None,
                   help="Input files under --input-dir. Defaults to all D2 converted splits.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading samples from {args.input_dir} ...")
    samples = load_samples(args.input_dir, args.input_files)
    print(f"  loaded {len(samples)} samples")

    if not samples:
        print("ERROR: no samples loaded. Check input directory.", file=sys.stderr)
        return 1

    print("Converting to SFT format ...")
    records = convert(samples)

    print(f"Writing {len(records)} records to {args.output} ...")
    write_jsonl(records, args.output)

    print("Building data report ...")
    report = build_report(records)
    write_json(report, args.report)

    print()
    print("=== SFT Data Report ===")
    for k, v in report.items():
        if k == "examples_of_invalid_samples":
            print(f"  {k}: ({len(v)} shown)")
        else:
            print(f"  {k}: {v}")

    ok = (
        report["format_valid_rate"] >= 0.95
        and report["answer_extraction_success_rate"] >= 0.95
    )
    print()
    if ok:
        print("PASS: format_valid_rate and answer_extraction_success_rate >= 95%")
    else:
        print("FAIL: quality thresholds not met")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
