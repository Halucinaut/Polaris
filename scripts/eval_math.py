#!/usr/bin/env python3
"""
Math evaluation script.

Reads predictions and references from JSONL files, extracts answers from model
predictions, and computes accuracy metrics.

Expected input formats:
- predictions.jsonl: {"problem_id": str, "prediction": str}
- references.jsonl:  {"problem_id": str, "problem": str, "answer": str, ...}
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReferenceRecord:
    """A single reference (ground-truth) record."""
    problem_id: str
    problem: str
    answer: str
    level: int
    source: str
    domain: str


@dataclass
class PredictionRecord:
    """A single model prediction record."""
    problem_id: str
    prediction: str          # Raw model output (may contain <think> blocks)


@dataclass
class EvalResult:
    """Result of evaluating one prediction against its reference."""
    problem_id: str
    reference_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    extraction_method: str   # e.g. "boxed", "answer_tag", "numeric_fallback", "none"
    extraction_success: int  # 1 if any answer was extracted, else 0
    format_adherence: int    # 1 if <think>...</think> + final answer format, else 0
    pass_: int               # 1 if extraction_success and is_correct, else 0
    completion_length: int   # len(prediction)


# ---------------------------------------------------------------------------
# Format & extraction helpers
# ---------------------------------------------------------------------------

def has_think_block(text: str) -> bool:
    """Check whether text contains a well-formed <think>...</think> block."""
    return bool(re.search(r"<think>.*?</think>", text, re.DOTALL))


def get_post_think_text(text: str) -> str:
    """Return the portion of text that comes after </think>."""
    match = re.search(r"</think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def extract_think_content(text: str) -> Optional[str]:
    """Extract content between <think> and </think>."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}."""
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return None

    idx = start + len(marker)
    depth = 1
    chars = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        idx += 1

    return None


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract content after #### marker."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_numeric_fallback(text: str) -> Optional[str]:
    """
    Fallback: extract the last standalone numeric token in the text.
    Handles integers, decimals, and fractions like 3/4.
    """
    # Look for numbers: integers, decimals, or fractions (e.g. 3/4, 2.5)
    candidates = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?:/\d+)?(?!\w)", text)
    if candidates:
        return candidates[-1]
    return None


def extract_predicted_answer(prediction: str) -> tuple[Optional[str], str]:
    """
    Extract the final answer from a raw model prediction.
    Only looks *after* </think> for the final answer.

    Returns:
        (answer_string, extraction_method_name)
    """
    post_think = get_post_think_text(prediction)

    # Try boxed first
    ans = extract_boxed_answer(post_think)
    if ans is not None:
        return ans, "boxed"

    # Try <answer> tag
    ans = extract_answer_tag(post_think)
    if ans is not None:
        return ans, "answer_tag"

    # Try #### marker
    ans = extract_hash_answer(post_think)
    if ans is not None:
        return ans, "hash_answer"

    # Numeric fallback
    ans = extract_numeric_fallback(post_think)
    if ans is not None:
        return ans, "numeric_fallback"

    return None, "none"


def has_m1_format_adherence(prediction: str) -> bool:
    """Check M1 assistant continuation format: think block plus boxed answer."""
    if "<think>" not in prediction or "</think>" not in prediction:
        return False
    if not has_think_block(prediction):
        return False
    post_think = get_post_think_text(prediction)
    return extract_boxed_answer(post_think) is not None


# ---------------------------------------------------------------------------
# Answer normalization & comparison
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    normalized = answer.strip().lower().replace(",", "").replace(" ", "")
    return re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", normalized)


def parse_numeric(answer: str) -> Optional[Fraction]:
    """
    Try to parse an answer as a numeric value (int, float, or fraction).
    Returns a Fraction for exact comparison, or None if not parseable.
    """
    normalized = normalize_answer(answer)
    if not normalized:
        return None

    # Try fraction first: e.g. "3/4", "1/2"
    if "/" in normalized and normalized.count("/") == 1:
        try:
            num, den = normalized.split("/")
            return Fraction(int(num), int(den))
        except (ValueError, ZeroDivisionError):
            pass

    # Try integer
    try:
        return Fraction(int(normalized))
    except ValueError:
        pass

    # Try float
    try:
        return Fraction(normalized)
    except ValueError:
        pass

    return None


def answers_match(pred: Optional[str], ref: str) -> bool:
    """Check whether predicted answer matches reference answer."""
    if pred is None:
        return False

    # First try exact string match
    if normalize_answer(pred) == normalize_answer(ref):
        return True

    # Then try numeric equivalence
    pred_num = parse_numeric(pred)
    ref_num = parse_numeric(ref)
    if pred_num is not None and ref_num is not None:
        return pred_num == ref_num

    return False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_references(path: Path) -> dict[str, ReferenceRecord]:
    """Load references.jsonl into a dict keyed by problem_id."""
    records: dict[str, ReferenceRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = ReferenceRecord(
                problem_id=obj["problem_id"],
                problem=obj["problem"],
                answer=obj["answer"],
                level=obj["level"],
                source=obj["source"],
                domain=obj["domain"],
            )
            records[rec.problem_id] = rec
    return records


def load_predictions(path: Path) -> list[PredictionRecord]:
    """Load predictions.jsonl into a list of PredictionRecord."""
    records: list[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = PredictionRecord(
                problem_id=obj["problem_id"],
                prediction=obj["prediction"],
            )
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

def evaluate(
    predictions: list[PredictionRecord],
    references: dict[str, ReferenceRecord],
) -> list[EvalResult]:
    """Run extraction + comparison for every prediction."""
    results: list[EvalResult] = []
    for pred in predictions:
        ref = references.get(pred.problem_id)
        if ref is None:
            continue

        predicted_answer, method = extract_predicted_answer(pred.prediction)
        extraction_success = 1 if predicted_answer is not None else 0
        is_correct = answers_match(predicted_answer, ref.answer)

        format_adherence = 1 if has_m1_format_adherence(pred.prediction) else 0

        pass_ = 1 if (extraction_success == 1 and is_correct) else 0

        results.append(EvalResult(
            problem_id=pred.problem_id,
            reference_answer=ref.answer,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            extraction_method=method,
            extraction_success=extraction_success,
            format_adherence=format_adherence,
            pass_=pass_,
            completion_length=len(pred.prediction),
        ))
    return results


def compute_metrics(results: list[EvalResult]) -> dict:
    """Aggregate accuracy and per-extraction-method statistics."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total > 0 else 0.0
    pass_count = sum(r.pass_ for r in results)
    extraction_count = sum(r.extraction_success for r in results)
    format_count = sum(r.format_adherence for r in results)

    method_counts: dict[str, int] = {}
    method_correct: dict[str, int] = {}
    for r in results:
        method_counts[r.extraction_method] = method_counts.get(r.extraction_method, 0) + 1
        if r.is_correct:
            method_correct[r.extraction_method] = method_correct.get(r.extraction_method, 0) + 1

    method_accuracy = {
        method: method_correct.get(method, 0) / count
        for method, count in method_counts.items()
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "pass_at_1": pass_count / total if total > 0 else 0.0,
        "answer_extraction_success": extraction_count / total if total > 0 else 0.0,
        "format_adherence": format_count / total if total > 0 else 0.0,
        "method_breakdown": {
            method: {
                "count": count,
                "correct": method_correct.get(method, 0),
                "accuracy": method_accuracy[method],
            }
            for method, count in method_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: dict, results: list[EvalResult]) -> None:
    """Print human-readable evaluation report to stdout."""
    print(f"Total: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print()
    for method, stats in metrics["method_breakdown"].items():
        print(f"  Method '{method}': {stats['correct']}/{stats['count']} = {stats['accuracy']:.2%}")


def write_results_jsonl(results: list[EvalResult], out_path: Path) -> None:
    """Write detailed results to a JSONL file."""
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            obj = {
                "problem_id": r.problem_id,
                "reference_answer": r.reference_answer,
                "predicted_answer": r.predicted_answer,
                "is_correct": r.is_correct,
                "extraction_method": r.extraction_method,
                "extraction_success": r.extraction_success,
                "format_adherence": r.format_adherence,
                "pass": r.pass_,
                "completion_length": r.completion_length,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate math predictions against references."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Path to references.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write detailed results JSONL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    references = load_references(args.references)
    predictions = load_predictions(args.predictions)

    results = evaluate(predictions, references)
    metrics = compute_metrics(results)

    print_report(metrics, results)

    if args.output:
        write_results_jsonl(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
