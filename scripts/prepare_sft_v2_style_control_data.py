"""
Convert DPO v2 style train split to SFT control format.

Input:  data/math/splits/dpo_v2_style_train_449.jsonl
Output: data/math/splits/sft_v2_style_control_train_449.jsonl

Each output record:
  - messages: copied from input (system + user)
  - target: chosen response (exact bytes from input)
  - metadata: problem_id, answer, source_pair_id, source="dpo_v2_style_chosen"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_EXPECTED_INPUT_COUNT = 449


def _check_non_empty_str(value, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _require_non_empty_str(value, label: str) -> str:
    _check_non_empty_str(value, label)
    return value.strip()


def _resolve_problem_id(record: dict) -> str:
    """Resolve problem_id: top-level metadata first, then top-level field."""
    top_meta = record.get("metadata", {}) or {}
    pid = top_meta.get("problem_id")
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    pid = record.get("problem_id")
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    raise ValueError("problem_id not found in metadata or top-level")


def convert_record(record: dict) -> dict:
    """Convert one DPO v2 record to SFT control format."""
    problem_id = _resolve_problem_id(record)
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 1:
        raise ValueError("messages must be a non-empty list")
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) != 1:
        raise ValueError(f"Expected exactly 1 user message, got {len(user_msgs)}")
    _check_non_empty_str(user_msgs[0].get("content", ""), "User message content")

    chosen = record.get("chosen")
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError(f"chosen must be a non-empty string, got {type(chosen).__name__}")

    answer = record.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        # Fallback: try metadata
        top_meta = record.get("metadata", {}) or {}
        answer = top_meta.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer not found")

    pair_id = record.get("id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        pair_id = ""

    return {
        "messages": messages,
        "target": chosen,
        "metadata": {
            "problem_id": problem_id,
            "answer": answer.strip(),
            "source_pair_id": pair_id,
            "source": "dpo_v2_style_chosen",
        },
    }


def load_input(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DPO v2 train split to SFT control format"
    )
    parser.add_argument(
        "--input",
        default="data/math/splits/dpo_v2_style_train_449.jsonl",
        help="Input DPO v2 train split",
    )
    parser.add_argument(
        "--output",
        default="data/math/splits/sft_v2_style_control_train_449.jsonl",
        help="Output SFT JSONL",
    )
    parser.add_argument(
        "--report",
        default="data/math/reports/sft_v2_style_control_report.json",
        help="Output report JSON",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=_EXPECTED_INPUT_COUNT,
        help=f"Expected input record count (default: {_EXPECTED_INPUT_COUNT})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    records = load_input(input_path)
    expected = args.expected_count
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} input records, got {len(records)}"
        )

    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()

    # Validate IDs
    seen_ids: set[str] = set()
    for r in records:
        pid = _resolve_problem_id(r)
        if pid in seen_ids:
            raise ValueError(f"Duplicate problem_id: {pid}")
        seen_ids.add(pid)

    # Convert
    converted = []
    target_lengths = []
    for r in records:
        c = convert_record(r)
        converted.append(c)
        target_lengths.append(len(c["target"]))

    if len(converted) != expected:
        raise ValueError(f"Converted count mismatch: {len(converted)}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()

    # Report
    report = {
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "record_count": len(converted),
        "target_length_stats": {
            "min": min(target_lengths),
            "max": max(target_lengths),
            "mean": round(sum(target_lengths) / len(target_lengths), 1),
            "median": sorted(target_lengths)[len(target_lengths) // 2],
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} records")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Target length: min={report['target_length_stats']['min']}, "
          f"mean={report['target_length_stats']['mean']}, "
          f"max={report['target_length_stats']['max']}")


if __name__ == "__main__":
    main()
