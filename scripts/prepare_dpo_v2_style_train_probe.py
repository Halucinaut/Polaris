"""
Prepare DPO v2 style train probe: 30 samples from training set split by length ratio.

Three bins (short/balanced/long) x 10 each, deterministic selection via sha256 sort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_BIN_SPECS = [
    ("short", 0.0, 0.9),
    ("balanced", 0.9, 1.2),
    ("long", 1.2, float("inf")),
]

_SELECTION_PREFIX = "dpo_v2_style_probe_v1:"
_PER_BIN = 10
_EXPECTED_TOTAL = _PER_BIN * len(_BIN_SPECS)
_EXPECTED_INPUT_COUNT = 449


def _sha256_hex(problem_id: str) -> str:
    payload = f"{_SELECTION_PREFIX}{problem_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_non_empty_str(value, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _require_non_empty_str(value, label: str) -> str:
    _check_non_empty_str(value, label)
    return value.strip()


def _extract_user_content(messages) -> str:
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) != 1:
        raise ValueError(
            f"Expected exactly 1 user message, got {len(user_msgs)}"
        )
    content = user_msgs[0].get("content", "")
    _check_non_empty_str(content, "User message content")
    return content


def _extract_answer(record: dict) -> str:
    """Extract answer: top-level answer field, or from messages."""
    # Top-level answer field (DPO v2 format)
    ans = record.get("answer")
    if isinstance(ans, str) and ans.strip():
        return ans
    # Fallback: assistant message content
    messages = record.get("messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if assistant_msgs:
        content = assistant_msgs[0].get("content", "")
        if isinstance(content, str) and content.strip():
            return content
    raise ValueError("answer not found in record")


def _bin_for_ratio(ratio: float) -> str:
    for name, lo, hi in _BIN_SPECS:
        if lo <= ratio < hi:
            return name
    return _BIN_SPECS[-1][0]


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
    problem_id = _resolve_problem_id(record)
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 1:
        raise ValueError("messages must be a non-empty list")
    # Ratio from top-level metadata (DPO v2 format)
    top_meta = record.get("metadata", {}) or {}
    ratio = top_meta.get("token_length_ratio")
    # Fallback: chosen assistant metadata
    if not isinstance(ratio, (int, float)):
        chosen = [m for m in messages if m.get("role") == "assistant"]
        if chosen:
            chosen_meta = chosen[0].get("metadata", {}) or {}
            ratio = chosen_meta.get("chosen_length_ratio")
    if not isinstance(ratio, (int, float)):
        raise ValueError(f"token_length_ratio missing or invalid: {ratio!r}")
    content = _extract_user_content(messages)
    answer = _extract_answer(record)
    return {
        "problem_id": problem_id,
        "problem": content,
        "answer": answer,
        "source": "dpo_v2_style_train_probe",
        "probe_bin": _bin_for_ratio(float(ratio)),
        "source_length_ratio": float(ratio),
    }


def _load_input(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def select_probe(records: list[dict]) -> list[dict]:
    """Select exactly 30 probe records: 10 per bin, deterministic by sha256."""
    if len(records) != _EXPECTED_INPUT_COUNT:
        raise ValueError(
            f"Expected {_EXPECTED_INPUT_COUNT} records, got {len(records)}"
        )
    # Validate IDs
    seen_ids: set[str] = set()
    for r in records:
        pid = _resolve_problem_id(r)
        if pid in seen_ids:
            raise ValueError(f"Duplicate problem_id: {pid}")
        seen_ids.add(pid)
    # Bin candidates
    bins: dict[str, list[tuple[str, dict]]] = {name: [] for name, _, _ in _BIN_SPECS}
    for r in records:
        converted = convert_record(r)
        bins[converted["probe_bin"]].append((converted["problem_id"], converted))
    # Select top-10 per bin by sha256
    selected: list[dict] = []
    report_bins = {}
    for name, lo, hi in _BIN_SPECS:
        candidates = bins[name]
        sorted_candidates = sorted(candidates, key=lambda t: _sha256_hex(t[0]))
        if len(sorted_candidates) < _PER_BIN:
            raise ValueError(
                f"Bin '{name}' has {len(sorted_candidates)} candidates, need {_PER_BIN}"
            )
        chosen = sorted_candidates[:_PER_BIN]
        selected.extend(c for _, c in chosen)
        report_bins[name] = {
            "candidates": len(candidates),
            "selected_ids": [pid for pid, _ in chosen],
        }
    if len(selected) != _EXPECTED_TOTAL:
        raise ValueError(f"Expected {_EXPECTED_TOTAL} selected, got {len(selected)}")
    selected_ids = [r["problem_id"] for r in selected]
    if len(set(selected_ids)) != _EXPECTED_TOTAL:
        raise ValueError("Duplicate problem_id in selected probe")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DPO v2 style train probe (30 samples)"
    )
    parser.add_argument(
        "--input",
        default="data/math/splits/dpo_v2_style_train_449.jsonl",
        help="Input DPO v2 train split",
    )
    parser.add_argument(
        "--output",
        default="data/math/probes/dpo_v2_style_train_probe_30.jsonl",
        help="Output probe JSONL",
    )
    parser.add_argument(
        "--report",
        default="data/math/probes/dpo_v2_style_train_probe_30_report.json",
        help="Output report JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    records = _load_input(input_path)
    input_sha = hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()

    probe = select_probe(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in probe:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()

    # Recompute bins for report
    bins: dict[str, list[str]] = {name: [] for name, _, _ in _BIN_SPECS}
    for rec in probe:
        bins[rec["probe_bin"]].append(rec["problem_id"])

    report = {
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "per_bin": {
            name: {"count": len(bins[name]), "selected_ids": bins[name]}
            for name, _, _ in _BIN_SPECS
        },
        "total": len(probe),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Selected {len(probe)} probe records:")
    for name, _, _ in _BIN_SPECS:
        print(f"  {name}: {len(bins[name])}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
