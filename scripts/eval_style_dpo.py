#!/usr/bin/env python3
"""
DPO v2 style evaluation script.

Reads predictions and references from JSONL files, evaluates answer correctness
and style adherence against the DPO v2 "Solution + numbered steps + Final" template.

Expected input formats:
  --predictions: {"problem_id": str, "prediction": str, ...}
  --references:  {"problem_id": str, "answer": str, ...}
                 or {"metadata": {"problem_id": str}, "answer": str, ...}

Outputs (written to a NEW --output-dir that must not pre-exist):
  summary.json     — aggregate metrics
  per_sample.jsonl — per-sample diagnostics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Re-exported helpers (avoid modifying eval_math.py)
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
    """Extract the last \\boxed{...} value using brace-depth tracking."""
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
    """Rough token estimate: regex-based count of word and punctuation tokens."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


# ---------------------------------------------------------------------------
# Style adherence
# ---------------------------------------------------------------------------

def _parse_single_boxed(payload: str) -> str | None:
    """Parse *payload* as exactly one ``\\boxed{<content>}``.

    Returns the inner content (stripped) on success, or None on failure.
    The payload must:
      - start with ``\\boxed{``
      - have balanced braces
      - close the outer brace exactly at the end of *payload*
      - have non-empty inner content
    Supports nested braces (e.g. ``\\boxed{\\frac{3}{4}}``).
    """
    marker = r"\boxed{"
    if not payload.startswith(marker):
        return None

    position = len(marker)
    depth = 1
    chars: list[str] = []
    while position < len(payload):
        char = payload[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                # Must be exactly at the end of payload
                if position != len(payload) - 1:
                    return None
                content = "".join(chars).strip()
                return content if content else None
        chars.append(char)
        position += 1

    # Never reached depth 0 → unbalanced
    return None


def check_style_adherence(text: str) -> tuple[bool, list[str]]:
    """Check whether *text* strictly follows the DPO v2 style template.

    Template requirements:
      1. Starts with ``<think>\\nSolution:\\n``
      2. Exactly one ``<think>`` and one ``</think>``, in order
      3. Steps inside the think block are contiguous numbered 1, 2, 3, …
      4. Ends with ``Final: The answer is \\boxed{...}.``
         - The payload between ``Final: The answer is `` and the trailing ``.``
           must be exactly one complete ``\\boxed{<non-empty-content>}``
         - Parsed via brace-depth; nested braces supported
         - No residual content after the outer closing brace

    Returns (adherent: bool, failure_reasons: list[str]).
    """
    reasons: list[str] = []
    normalized = text.strip()

    # 1. Prefix
    if not normalized.startswith("<think>\nSolution:\n"):
        reasons.append("missing_solution_prefix")

    # 2. Think tags
    if normalized.count("<think>") != 1 or normalized.count("</think>") != 1:
        reasons.append("invalid_think_tags")

    # 3. Contiguous numbered steps
    closing = normalized.find("</think>")
    thought = normalized[len("<think>"):closing] if closing >= 0 else ""
    step_numbers = [int(n) for n in re.findall(r"(?m)^\s*(\d+)\.\s+", thought)]
    if not step_numbers or step_numbers != list(range(1, len(step_numbers) + 1)):
        reasons.append("steps_not_contiguous")

    # 4. Final template — strict brace-depth validation
    final_match = re.search(r"(?s)</think>\s*Final: The answer is (.+)$", normalized)
    if not final_match:
        reasons.append("invalid_final_template")
    else:
        tail = final_match.group(1)
        # Must end with exactly one '.'
        if not tail.endswith(".") or tail.endswith(".."):
            reasons.append("invalid_final_template")
        else:
            payload = tail[:-1]  # strip trailing '.'
            if _parse_single_boxed(payload) is None:
                reasons.append("invalid_final_template")

    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    prediction: str,
    reference_answer: str,
) -> dict[str, Any]:
    """Evaluate a single prediction against its reference answer.

    Returns a dict with all per-sample metrics.
    """
    # Answer extraction
    predicted_answer = extract_boxed_answer(prediction)
    answer_extractable = predicted_answer is not None
    answer_correct = answer_extractable and answers_match(predicted_answer, reference_answer)

    # Style adherence
    style_adherent, style_failures = check_style_adherence(prediction)

    # Step count and character count
    normalized = prediction.strip()
    closing = normalized.find("</think>")
    thought = normalized[len("<think>"):closing] if closing >= 0 else ""
    step_count = len(re.findall(r"(?m)^\s*\d+\.\s+", thought))
    char_count = len(prediction)
    token_estimate = estimated_tokens(prediction)

    return {
        "answer_extractable": answer_extractable,
        "answer_correct": answer_correct,
        "style_adherent": style_adherent,
        "predicted_answer": predicted_answer,
        "step_count": step_count,
        "char_count": char_count,
        "token_estimate": token_estimate,
        "style_failures": style_failures,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_problem_id(record: dict[str, Any]) -> str:
    """Extract problem_id from top-level, falling back to metadata.problem_id.

    Returns empty string if neither yields a non-whitespace value.
    """
    pid = record.get("problem_id")
    if pid is not None:
        pid_str = str(pid).strip()
        if pid_str:
            return pid_str
    meta_pid = record.get("metadata", {}).get("problem_id")
    if meta_pid is not None:
        meta_str = str(meta_pid).strip()
        if meta_str:
            return meta_str
    return ""


def resolve_answer(record: dict[str, Any]) -> str:
    """Extract the gold answer, preferring 'answer' field.

    GSM8K compatibility: if the answer contains '####', take the last
    non-empty text after the last '####' as the canonical answer.
    """
    answer = record.get("answer")
    if answer is None:
        answer = record.get("metadata", {}).get("answer", "")
    answer = str(answer)

    # GSM8K #### format
    if "####" in answer:
        parts = answer.split("####")
        last_part = parts[-1].strip()
        if last_part:
            return last_part

    return answer


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    predictions: list[dict[str, Any]],
    references_raw: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Validate prediction and reference records before evaluation.

    Returns (ok, error_message).
    """
    # Validate predictions
    pred_ids: dict[str, int] = {}
    for i, pred in enumerate(predictions):
        pid = str(pred.get("problem_id") or "").strip()
        if not pid:
            return False, f"prediction record {i} has empty problem_id"
        if pid in pred_ids:
            return False, f"duplicate prediction problem_id: {pid}"
        pred_ids[pid] = i

    # Validate references and build lookup
    ref_ids: dict[str, int] = {}
    for i, ref in enumerate(references_raw):
        pid = resolve_problem_id(ref)
        if not pid:
            return False, f"reference record {i} has empty problem_id"
        answer = resolve_answer(ref)
        if not answer:
            return False, f"reference record {i} (problem_id={pid}) has empty answer"
        if pid in ref_ids:
            return False, f"duplicate reference problem_id: {pid}"
        ref_ids[pid] = i

    # Both empty is allowed (0-sample report)
    if not pred_ids and not ref_ids:
        return True, ""

    # Predictions empty + references non-empty → fail
    if not pred_ids and ref_ids:
        return False, f"predictions file is empty but references has {len(ref_ids)} records"

    # Set comparison
    pred_set = set(pred_ids)
    ref_set = set(ref_ids)

    missing_in_refs = pred_set - ref_set
    extra_in_refs = ref_set - pred_set

    if missing_in_refs:
        sample = sorted(missing_in_refs)[:3]
        return False, f"{len(missing_in_refs)} prediction(s) have no matching reference (e.g. {sample})"
    if extra_in_refs:
        sample = sorted(extra_in_refs)[:3]
        return False, f"{len(extra_in_refs)} reference(s) have no matching prediction (e.g. {sample})"

    return True, ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPO v2 style adherence and answer correctness")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL")
    parser.add_argument("--references", type=Path, required=True, help="Path to references JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for output (must not exist)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Guard: output dir must not already exist
    if args.output_dir.exists():
        print(f"ERROR: output directory already exists: {args.output_dir}")
        return 1

    # Load inputs
    if not args.predictions.exists():
        print(f"ERROR: predictions file not found: {args.predictions}")
        return 1
    if not args.references.exists():
        print(f"ERROR: references file not found: {args.references}")
        return 1

    predictions = load_jsonl(args.predictions)
    references_raw = load_jsonl(args.references)

    # Validate inputs before creating output dir
    ok, error_msg = _validate_inputs(predictions, references_raw)
    if not ok:
        print(f"ERROR: {error_msg}")
        return 1

    # Build reference lookup
    references: dict[str, str] = {}
    for ref in references_raw:
        pid = resolve_problem_id(ref)
        answer = resolve_answer(ref)
        if pid and answer:
            references[pid] = answer

    # Evaluate each prediction
    per_sample: list[dict[str, Any]] = []
    for pred in predictions:
        pid = str(pred.get("problem_id", "")).strip()
        prediction_text = str(pred.get("prediction", ""))
        ref_answer = references.get(pid, "")

        result = evaluate_sample(prediction_text, ref_answer)
        result["problem_id"] = pid
        per_sample.append(result)

    # Compute summary
    n = len(per_sample)
    n_correct = sum(1 for r in per_sample if r["answer_correct"])
    n_extractable = sum(1 for r in per_sample if r["answer_extractable"])
    n_adherent = sum(1 for r in per_sample if r["style_adherent"])
    n_both = sum(1 for r in per_sample if r["answer_correct"] and r["style_adherent"])
    avg_steps = sum(r["step_count"] for r in per_sample) / n if n else 0.0
    avg_chars = sum(r["char_count"] for r in per_sample) / n if n else 0.0
    avg_tokens = sum(r["token_estimate"] for r in per_sample) / n if n else 0.0

    summary = {
        "total_samples": n,
        "answer_correct_count": n_correct,
        "answer_correct_rate": round(n_correct / n, 4) if n else 0.0,
        "answer_extractable_count": n_extractable,
        "answer_extractable_rate": round(n_extractable / n, 4) if n else 0.0,
        "style_adherent_count": n_adherent,
        "style_adherent_rate": round(n_adherent / n, 4) if n else 0.0,
        "correct_and_adherent_count": n_both,
        "correct_and_adherent_rate": round(n_both / n, 4) if n else 0.0,
        "avg_step_count": round(avg_steps, 2),
        "avg_char_count": round(avg_chars, 1),
        "avg_token_estimate": round(avg_tokens, 1),
        "token_estimate_method": r"regex word+punctuation count (\w+|[^\w\s])",
    }

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=False)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_sample_path = args.output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for record in per_sample:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    print(f"=== DPO v2 Style Evaluation ===")
    print(f"Samples:          {n}")
    print(f"Answer correct:   {n_correct}/{n} ({summary['answer_correct_rate']:.2%})")
    print(f"Answer extractable: {n_extractable}/{n} ({summary['answer_extractable_rate']:.2%})")
    print(f"Style adherent:   {n_adherent}/{n} ({summary['style_adherent_rate']:.2%})")
    print(f"Correct+adherent: {n_both}/{n} ({summary['correct_and_adherent_rate']:.2%})")
    print(f"Avg steps:        {summary['avg_step_count']}")
    print(f"Avg chars:        {summary['avg_char_count']}")
    print(f"\nOutputs:")
    print(f"  {summary_path}")
    print(f"  {per_sample_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
