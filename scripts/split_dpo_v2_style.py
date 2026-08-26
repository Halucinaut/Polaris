#!/usr/bin/env python3
"""
Split DPO v2 style-controlled dataset into train / stress / quarantine.

Validation protocol (strict):
  1. chosen/rejected boxed answer must match gold answer
  2. chosen must follow: <think>\\nSolution:\\n + contiguous numbered steps + </think>\\n\\nFinal: The answer is \\boxed{...}.
  3. length ratio (chosen/rejected tokens) in [0.55, 1.60]
  4. character similarity (SequenceMatcher) <= 0.97

Split classification (exact set matching):
  - train:      errors == {} (empty — all checks pass)
  - stress:     errors == {"chosen_length_ratio_out_of_range"} (exactly)
  - quarantine: all other error combinations

Outputs (default paths):
  data/math/splits/dpo_v2_style_train_449.jsonl
  data/math/splits/dpo_v2_style_stress_50.jsonl
  data/math/quarantine/dpo_v2_style_invalid_1.jsonl
  data/math/reports/dpo_v2_style_split_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Validation helpers (ported from generate_style_dpo.py)
# ---------------------------------------------------------------------------

def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower().replace(",", ""))


def parse_numeric(value: str) -> Fraction | None:
    value = normalize_answer(value)
    value = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    try:
        if value.count("/") == 1:
            numerator, denominator = value.split("/")
            return Fraction(int(numerator), int(denominator))
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(candidate: str, gold: str) -> bool:
    if normalize_answer(candidate) == normalize_answer(gold):
        return True
    candidate_number = parse_numeric(candidate)
    gold_number = parse_numeric(gold)
    return candidate_number is not None and gold_number is not None and candidate_number == gold_number


def extract_boxed_answer(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    position = start + len(marker)
    depth = 1
    chars: list[str] = []
    while position < len(text):
        char = text[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        position += 1
    return None


def estimated_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


# ---------------------------------------------------------------------------
# Validation protocol
# ---------------------------------------------------------------------------

def validate_record(
    record: dict[str, Any],
    min_length_ratio: float = 0.55,
    max_length_ratio: float = 1.60,
    max_similarity: float = 0.97,
) -> dict[str, Any]:
    """Apply strict validation protocol. Returns dict with errors list and diagnostics."""
    errors: list[str] = []

    chosen: str = record["chosen"]
    rejected: str = record["rejected"]
    gold_answer: str = record["answer"]

    # 1. Boxed answer checks
    chosen_answer = extract_boxed_answer(chosen)
    if chosen_answer is None or not answers_match(chosen_answer, gold_answer):
        errors.append("chosen_boxed_answer_mismatch")

    rejected_answer = extract_boxed_answer(rejected)
    if rejected_answer is None or not answers_match(rejected_answer, gold_answer):
        errors.append("rejected_boxed_answer_mismatch")

    # 2. Template checks (chosen only)
    normalized = chosen.strip()
    expected_prefix = "<think>\nSolution:\n"
    if not normalized.startswith(expected_prefix):
        errors.append("chosen_missing_solution_prefix")
    if normalized.count("<think>") != 1 or normalized.count("</think>") != 1:
        errors.append("chosen_invalid_think_tags")

    closing = normalized.find("</think>")
    thought = normalized[len("<think>"):closing] if closing >= 0 else ""
    step_numbers = [int(number) for number in re.findall(r"(?m)^\s*(\d+)\.\s+", thought)]
    if not step_numbers or step_numbers != list(range(1, len(step_numbers) + 1)):
        errors.append("chosen_steps_not_contiguous")

    if not re.search(r"(?s)</think>\s*Final: The answer is \\boxed\{.+\}\.\s*$", normalized):
        errors.append("chosen_invalid_final_template")

    # 3. Length ratio
    chosen_length = estimated_tokens(chosen)
    rejected_length = estimated_tokens(rejected)
    length_ratio = chosen_length / rejected_length if rejected_length > 0 else 0.0
    if not min_length_ratio <= length_ratio <= max_length_ratio:
        errors.append("chosen_length_ratio_out_of_range")

    # 4. Character similarity
    similarity = SequenceMatcher(a=rejected, b=chosen).ratio()
    if similarity > max_similarity:
        errors.append("chosen_too_similar_to_rejected")

    return {
        "ok": not errors,
        "errors": errors,
        "problem_id": record.get("metadata", {}).get("problem_id", ""),
        "chosen_answer": chosen_answer,
        "rejected_answer": rejected_answer,
        "chosen_estimated_token_length": chosen_length,
        "rejected_estimated_token_length": rejected_length,
        "length_ratio": round(length_ratio, 4),
        "character_similarity": round(similarity, 4),
        "step_count": len(step_numbers),
    }


# ---------------------------------------------------------------------------
# Split classification (exact set matching)
# ---------------------------------------------------------------------------

# The only error set that maps to "stress"
_STRESS_ERRORS = frozenset({"chosen_length_ratio_out_of_range"})


def classify_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify a record into train/stress/quarantine using exact set matching.

    Rules:
      - train:      errors is empty (all checks pass)
      - stress:     errors is exactly {"chosen_length_ratio_out_of_range"}
      - quarantine: all other error combinations (similarity-only, mixed, etc.)

    Returns (split_name, validation_result).
    """
    result = validate_record(record)
    errors = frozenset(result["errors"])

    if not errors:
        return "train", result

    if errors == _STRESS_ERRORS:
        return "stress", result

    return "quarantine", result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DPO v2 style-controlled dataset")
    parser.add_argument(
        "--input", type=Path,
        default=Path("transfer/style_dpo_v2_returned/dpo_v2_style_controlled.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/math/splits/dpo_v2_style_train_449.jsonl"))
    parser.add_argument("--stress-path", type=Path, default=Path("data/math/splits/dpo_v2_style_stress_50.jsonl"))
    parser.add_argument("--quarantine-path", type=Path, default=Path("data/math/quarantine/dpo_v2_style_invalid_1.jsonl"))
    parser.add_argument("--report-path", type=Path, default=Path("data/math/reports/dpo_v2_style_split_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path: Path = args.input
    train_path: Path = args.train_path
    stress_path: Path = args.stress_path
    quarantine_path: Path = args.quarantine_path
    report_path: Path = args.report_path

    # Check for existing output files (all paths, regardless of default vs CLI)
    for path in [train_path, stress_path, quarantine_path, report_path]:
        if path.exists():
            print(f"ERROR: {path} already exists. Aborting to avoid overwrite.")
            return 1

    # Load input
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    print(f"Loading {input_path} ...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records")

    # Validate input count
    if len(records) != 500:
        print(f"ERROR: expected 500 input records, got {len(records)}")
        return 1

    # Classify
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "stress": [], "quarantine": []}
    results: dict[str, list[dict[str, Any]]] = {"train": [], "stress": [], "quarantine": []}
    all_problem_ids: set[str] = set()
    problem_ids: dict[str, set[str]] = {"train": set(), "stress": set(), "quarantine": set()}

    for record in records:
        split_name, result = classify_record(record)
        pid = result["problem_id"]

        # Check for duplicate problem_id across ALL splits
        if pid in all_problem_ids:
            print(f"ERROR: duplicate problem_id {pid} detected. Aborting.")
            return 1
        all_problem_ids.add(pid)

        splits[split_name].append(record)
        results[split_name].append(result)
        problem_ids[split_name].add(pid)

    # Verify expected counts before writing
    expected = {"train": 449, "stress": 50, "quarantine": 1}
    for split_name, expected_count in expected.items():
        actual = len(splits[split_name])
        if actual != expected_count:
            print(f"ERROR: {split_name} expected {expected_count}, got {actual}. No files written.")
            return 1

    # Verify no overlap (belt-and-suspenders)
    all_ids = set()
    for split_name, ids in problem_ids.items():
        overlap = all_ids & ids
        if overlap:
            print(f"ERROR: overlapping problem_ids in {split_name}: {overlap}. No files written.")
            return 1
        all_ids |= ids

    # All validations passed — write outputs
    write_jsonl(train_path, splits["train"])
    write_jsonl(stress_path, splits["stress"])
    write_jsonl(quarantine_path, splits["quarantine"])

    # Compute length ratio stats
    def ratio_stats(split_results: list[dict[str, Any]]) -> dict[str, float]:
        ratios = [r["length_ratio"] for r in split_results if r["length_ratio"] > 0]
        if not ratios:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        ratios.sort()
        return {
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "mean": round(sum(ratios) / len(ratios), 4),
            "median": round(ratios[len(ratios) // 2], 4),
        }

    # Build failure reasons
    quarantine_reasons: dict[str, list[str]] = {}
    for result in results["quarantine"]:
        pid = result["problem_id"]
        quarantine_reasons[pid] = result["errors"]

    stress_reasons: dict[str, list[str]] = {}
    for result in results["stress"]:
        pid = result["problem_id"]
        stress_reasons[pid] = result["errors"]

    # Generate report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "source_sha256": file_sha256(input_path),
        "total_input_records": len(records),
        "splits": {
            "train": {
                "count": len(splits["train"]),
                "file": str(train_path),
                "problem_ids": sorted(problem_ids["train"]),
                "length_ratio_stats": ratio_stats(results["train"]),
            },
            "stress": {
                "count": len(splits["stress"]),
                "file": str(stress_path),
                "problem_ids": sorted(problem_ids["stress"]),
                "length_ratio_stats": ratio_stats(results["stress"]),
                "failure_reasons": stress_reasons,
            },
            "quarantine": {
                "count": len(splits["quarantine"]),
                "file": str(quarantine_path),
                "problem_ids": sorted(problem_ids["quarantine"]),
                "failure_reasons": quarantine_reasons,
            },
        },
        "validation_protocol": {
            "min_length_ratio": 0.55,
            "max_length_ratio": 1.60,
            "max_similarity": 0.97,
            "split_rules": {
                "train": "errors == {} (empty)",
                "stress": "errors == {chosen_length_ratio_out_of_range} (exact set)",
                "quarantine": "all other error combinations",
            },
            "checks": [
                "chosen_boxed_answer_matches_gold",
                "rejected_boxed_answer_matches_gold",
                "chosen_solution_prefix",
                "chosen_think_tags",
                "chosen_contiguous_steps",
                "chosen_final_template",
                "length_ratio_in_range",
                "similarity_below_threshold",
            ],
        },
    }

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Print summary
    print(f"\n=== DPO v2 Style Split ===")
    print(f"Input:            {len(records)} records")
    print(f"Train:            {len(splits['train'])} records")
    print(f"Stress:           {len(splits['stress'])} records")
    print(f"Quarantine:       {len(splits['quarantine'])} records")
    print(f"Total:            {len(splits['train']) + len(splits['stress']) + len(splits['quarantine'])} records")
    print(f"\nOutputs:")
    print(f"  {train_path}")
    print(f"  {stress_path}")
    print(f"  {quarantine_path}")
    print(f"  {report_path}")
    print(f"\nAll counts match expected values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
