#!/usr/bin/env python3
"""
Convert DPO v2 style stress-50 split into eval_gsm8k-compatible format.

Input:  data/math/splits/dpo_v2_style_stress_50.jsonl (DPO pair records)
Output: data/math/splits/dpo_v2_style_stress_eval_50.jsonl (problem_id, problem, answer, source)

Each output line:
  {"problem_id": str, "problem": str, "answer": str, "source": "dpo_v2_style_stress"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _check_non_empty_str(value: Any, label: str) -> None:
    """Validate that *value* is a non-empty, non-whitespace string.

    Raises ValueError if value is not a str, or strip() yields empty.
    Does not return the value — use the original for output.
    """
    if not isinstance(value, str):
        raise ValueError(f"{label} must be str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{label} is empty or whitespace-only")


def _require_non_empty_str(value: Any, label: str) -> str:
    """Validate and return stripped value (for problem_id)."""
    _check_non_empty_str(value, label)
    return value.strip()  # type: ignore[union-attr]


def extract_user_content(messages: list[dict[str, str]]) -> str:
    """Extract the original content of the single user message (unstripped)."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) != 1:
        raise ValueError(f"expected exactly 1 user message, got {len(user_msgs)}")
    content = user_msgs[0].get("content")
    _check_non_empty_str(content, "user message content")
    return content  # type: ignore[return-value]


def convert_record(record: dict[str, Any]) -> dict[str, str]:
    """Convert a DPO pair record to eval format."""
    pid = _require_non_empty_str(
        record.get("metadata", {}).get("problem_id"), "metadata.problem_id"
    )
    problem = extract_user_content(record.get("messages", []))
    answer_raw = record.get("answer")
    _check_non_empty_str(answer_raw, "answer")
    return {
        "problem_id": pid,
        "problem": problem,
        "answer": answer_raw,  # preserve original text
        "source": "dpo_v2_style_stress",
    }


def validate_inputs(records: list[dict[str, Any]]) -> None:
    """Validate input records before conversion. Raises ValueError on failure."""
    if len(records) != 50:
        raise ValueError(f"expected 50 records, got {len(records)}")

    seen_ids: set[str] = set()
    for i, rec in enumerate(records):
        pid = _require_non_empty_str(
            rec.get("metadata", {}).get("problem_id"), f"record {i} metadata.problem_id"
        )
        if pid in seen_ids:
            raise ValueError(f"duplicate problem_id at record {i}: {pid}")
        seen_ids.add(pid)

        messages = rec.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if len(user_msgs) != 1:
            raise ValueError(f"record {i} (problem_id={pid}): expected 1 user message, got {len(user_msgs)}")
        _check_non_empty_str(
            user_msgs[0].get("content"), f"record {i} (problem_id={pid}) user message content"
        )

        _check_non_empty_str(
            rec.get("answer"), f"record {i} (problem_id={pid}) answer"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert stress-50 DPO pairs to eval format")
    parser.add_argument("--input", type=Path, required=True, help="Input stress-50 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output eval-format JSONL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output.exists():
        print(f"ERROR: output file already exists: {args.output}")
        return 1

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}")
        return 1

    records = load_jsonl(args.input)

    try:
        validate_inputs(records)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    converted = [convert_record(r) for r in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted),
        encoding="utf-8",
    )

    print(f"Converted {len(converted)} records: {args.input} → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
