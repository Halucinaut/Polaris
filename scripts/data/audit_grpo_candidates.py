#!/usr/bin/env python3
"""
Deterministic audit of local JSONL datasets for GRPO Math candidate selection.

Read-only: does not modify any data files. Reports:
1. All local JSONL sources with record counts and schema summary.
2. Contamination exclusion overlap (canonical_question_hash based).
3. GRPO input schema compatibility check (all records, not just first).

Usage:
    python scripts/data/audit_grpo_candidates.py [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from polaris.json_records import load_json_record_stream


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SchemaAudit:
    """Schema compatibility results for a file."""
    valid_count: int = 0
    invalid_count: int = 0
    invalid_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class FileAudit:
    path: str
    record_count: int
    top_level_keys_sample: list[str]
    has_messages: bool
    has_metadata_answer: bool
    has_problem_id: bool
    problem_id_source: str  # "top-level", "metadata", "phash+index", "none"
    has_answer: bool
    answer_source: str  # "top-level", "metadata", "none"
    has_source_tag: bool
    has_level: bool
    grpo_compatible: bool  # True only if ALL records pass schema check
    grpo_schema: SchemaAudit = field(default_factory=SchemaAudit)
    convertible_to_grpo: bool = False  # True if deterministic conversion exists
    conversion_note: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class OverlapReport:
    candidate_name: str
    excluded_name: str
    canonical_overlap_count: int  # per-question deduplicated
    id_match_count: int           # records where problem_id matched
    text_hash_match_count: int    # records where text hash matched
    overlap_examples: list[str]   # sample canonical hashes


@dataclass
class CandidateAudit:
    """Audit of a candidate source against all excluded sets."""
    candidate_name: str
    total_records: int
    unique_hashes: int
    excluded_by_training: int
    excluded_by_eval: int
    excluded_by_pilots: int
    excluded_by_union: int
    clean_remaining: int
    per_excluded_set: list[OverlapReport]


@dataclass
class AuditReport:
    files: list[FileAudit]
    candidate_audits: list[CandidateAudit]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonical_question_hash(text: str) -> str:
    """Canonical hash of normalized question text. One hash per unique question."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# Keep old name as alias for backward compat in tests
_text_hash = canonical_question_hash


def _extract_problem_id(record: dict) -> str | None:
    """Extract problem_id from various schema locations."""
    pid = record.get("problem_id")
    if pid:
        return str(pid)
    meta = record.get("metadata", {})
    if isinstance(meta, dict):
        pid = meta.get("problem_id")
        if pid:
            return str(pid)
    # v3 style: phash + problem_index
    phash = record.get("phash")
    pidx = record.get("problem_index")
    if phash is not None and pidx is not None:
        return f"phash:{phash}:idx:{pidx}"
    return None


def _extract_problem_text(record: dict) -> str | None:
    """Extract problem text from various schema locations."""
    # messages format
    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    # raw format
    return record.get("problem") or record.get("question")


def _extract_answer(record: dict) -> str | None:
    """Extract answer from various schema locations."""
    meta = record.get("metadata", {})
    if isinstance(meta, dict) and "answer" in meta:
        return meta["answer"]
    return record.get("answer")


def _check_grpo_compatibility(record: dict) -> list[str]:
    """Check if a record meets train_grpo.py input requirements.

    Mirrors validate_data_schema() in train_grpo.py:
    - messages must be a list (of dicts)
    - metadata.answer must exist as a string
    """
    issues = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        issues.append("missing or invalid 'messages' field")
    meta = record.get("metadata", {})
    if not isinstance(meta, dict) or "answer" not in meta:
        issues.append("missing metadata.answer")
    return issues


# ---------------------------------------------------------------------------
# File audit
# ---------------------------------------------------------------------------

def audit_file(path: Path) -> FileAudit:
    """Audit a single JSONL file. Checks ALL records for GRPO compatibility."""
    try:
        records = load_json_record_stream(path)
    except Exception as exc:
        return FileAudit(
            path=str(path),
            record_count=0,
            top_level_keys_sample=[],
            has_messages=False,
            has_metadata_answer=False,
            has_problem_id=False,
            problem_id_source="none",
            has_answer=False,
            answer_source="none",
            has_source_tag=False,
            has_level=False,
            grpo_compatible=False,
            notes=[f"LOAD ERROR: {exc}"],
        )

    if not records:
        return FileAudit(
            path=str(path),
            record_count=0,
            top_level_keys_sample=[],
            has_messages=False,
            has_metadata_answer=False,
            has_problem_id=False,
            problem_id_source="none",
            has_answer=False,
            answer_source="none",
            has_source_tag=False,
            has_level=False,
            grpo_compatible=False,
            notes=["empty file"],
        )

    # Sample-based schema detection (first record)
    sample = records[0]
    top_keys = sorted(sample.keys())

    has_messages = isinstance(sample.get("messages"), list)
    meta = sample.get("metadata", {})
    has_metadata_answer = isinstance(meta, dict) and "answer" in meta
    has_top_answer = "answer" in sample
    has_answer = has_metadata_answer or has_top_answer
    answer_source = "metadata" if has_metadata_answer else ("top-level" if has_top_answer else "none")

    pid = _extract_problem_id(sample)
    has_pid = pid is not None
    if sample.get("problem_id"):
        pid_source = "top-level"
    elif isinstance(meta, dict) and meta.get("problem_id"):
        pid_source = "metadata"
    elif sample.get("phash") and sample.get("problem_index") is not None:
        pid_source = "phash+index"
    else:
        pid_source = "none"

    has_source = "source" in sample or (isinstance(meta, dict) and "source" in meta)
    has_level = "level" in sample or (isinstance(meta, dict) and "level" in meta)

    # Full-record GRPO schema audit
    schema = SchemaAudit()
    reason_counter: Counter[str] = Counter()
    for r in records:
        issues = _check_grpo_compatibility(r)
        if issues:
            schema.invalid_count += 1
            for issue in issues:
                reason_counter[issue] += 1
        else:
            schema.valid_count += 1
    schema.invalid_reasons = dict(reason_counter)

    grpo_ok = schema.invalid_count == 0 and schema.valid_count > 0

    return FileAudit(
        path=str(path),
        record_count=len(records),
        top_level_keys_sample=top_keys,
        has_messages=has_messages,
        has_metadata_answer=has_metadata_answer,
        has_problem_id=has_pid,
        problem_id_source=pid_source,
        has_answer=has_answer,
        answer_source=answer_source,
        has_source_tag=has_source,
        has_level=has_level,
        grpo_compatible=grpo_ok,
        grpo_schema=schema,
    )


def load_canonical_hashes(path: Path) -> set[str]:
    """Load canonical question hashes from a JSONL file."""
    try:
        records = load_json_record_stream(path)
    except Exception:
        return set()

    hashes: set[str] = set()
    for r in records:
        text = _extract_problem_text(r)
        if text:
            hashes.add(canonical_question_hash(text))
    return hashes


def load_ids_and_hashes(path: Path) -> tuple[set[str], set[str]]:
    """Load problem_ids and canonical text hashes from a JSONL file."""
    try:
        records = load_json_record_stream(path)
    except Exception:
        return set(), set()

    ids: set[str] = set()
    hashes: set[str] = set()
    for r in records:
        pid = _extract_problem_id(r)
        if pid:
            ids.add(pid)
        text = _extract_problem_text(r)
        if text:
            hashes.add(canonical_question_hash(text))
    return ids, hashes


# ---------------------------------------------------------------------------
# Overlap detection (canonical_hash based, one count per question)
# ---------------------------------------------------------------------------

def compute_overlap(
    candidate_hashes: set[str],
    candidate_ids: set[str],
    excluded_hashes: set[str],
    excluded_ids: set[str],
) -> tuple[int, int, int]:
    """Compute overlap between one candidate and one excluded set.

    Returns (canonical_overlap_count, id_match_count, text_hash_match_count).
    canonical_overlap_count counts each question at most once.
    id_match_count and text_hash_match_count are diagnostic breakdowns
    (their union >= canonical_overlap_count due to questions matching both ways).
    """
    hash_matches = candidate_hashes & excluded_hashes
    id_matches = candidate_ids & excluded_ids

    # Canonical overlap: union of both match methods, deduplicated by hash.
    # A question that matches by both ID and hash is counted once.
    canonical = set(hash_matches)
    # For ID matches, we need to find their corresponding hashes to avoid double-counting.
    # ID matches whose hash is already in hash_matches are already counted.
    # ID matches whose hash is NOT in hash_matches represent questions that matched
    # by ID but have different text (shouldn't happen with consistent data, but handle it).
    # We can't directly map IDs to hashes without the full record, so we use the
    # hash set as the canonical representation and add ID-only matches as a separate count.
    # However, the correct approach is: canonical_overlap = |hash_matches ∪ id_only_hashes|.
    # Since we don't have id->hash mapping, we report hash_matches as the canonical count
    # and note id_matches separately.

    # For the canonical count, we use hash_matches as the primary (it's the per-question
    # deduplication key). ID matches that don't have a corresponding hash match are
    # additional overlaps from questions that appear with different text (edge case).
    canonical_count = len(hash_matches)

    # Check if any ID matches are NOT represented in hash matches.
    # This happens when the same problem appears with different text in different schemas.
    # We count these as additional canonical overlaps.
    # Since we can't map IDs to hashes directly, we note this as a limitation.
    # The id_match_count is provided as a diagnostic.

    return canonical_count, len(id_matches), len(hash_matches)


def audit_candidate_source(
    candidate_name: str,
    candidate_hashes: set[str],
    candidate_ids: set[str],
    excluded_by_group: dict[str, set[str]],
    excluded_all_hashes: set[str],
    excluded_all_ids: set[str],
    per_excluded_detail: dict[str, tuple[set[str], set[str]]],
) -> CandidateAudit:
    """Audit one candidate source against all excluded sets."""
    total = len(candidate_hashes)  # unique questions

    # Per-group exclusion
    ex_training = len(candidate_hashes & excluded_by_group.get("training", set()))
    ex_eval = len(candidate_hashes & excluded_by_group.get("eval", set()))
    ex_pilots = len(candidate_hashes & excluded_by_group.get("pilots", set()))
    ex_union = len(candidate_hashes & excluded_all_hashes)
    clean = total - ex_union

    # Per-excluded-set detail
    per_set: list[OverlapReport] = []
    for name, (ex_h, ex_ids) in per_excluded_detail.items():
        canon, id_cnt, hash_cnt = compute_overlap(
            candidate_hashes, candidate_ids, ex_h, ex_ids,
        )
        overlap_examples = sorted(candidate_hashes & ex_h)[:5]
        per_set.append(OverlapReport(
            candidate_name=candidate_name,
            excluded_name=name,
            canonical_overlap_count=canon,
            id_match_count=id_cnt,
            text_hash_match_count=hash_cnt,
            overlap_examples=overlap_examples,
        ))

    return CandidateAudit(
        candidate_name=candidate_name,
        total_records=total,
        unique_hashes=len(candidate_hashes),
        excluded_by_training=ex_training,
        excluded_by_eval=ex_eval,
        excluded_by_pilots=ex_pilots,
        excluded_by_union=ex_union,
        clean_remaining=clean,
        per_excluded_set=per_set,
    )


# ---------------------------------------------------------------------------
# Main audit pipeline
# ---------------------------------------------------------------------------

# Candidate sources: files that could provide GRPO training data
CANDIDATE_SOURCES = [
    "data/math/gsm8k/train.jsonl",  # raw GSM8K train (convertible)
    "data/math/gsm8k/split/train_converted_d5_500.jsonl",
    "data/math/splits/sft_v1.jsonl",
    "data/math/splits/sft_d5_500.jsonl",
    "data/math/splits/sft_v2_style_control_train_449.jsonl",
    "data/math/gsm8k/split/test_converted_500.jsonl",
]

# Conversion notes for non-GRPO-compatible candidates
CONVERSION_NOTES = {
    "data/math/gsm8k/train.jsonl": "Requires conversion: wrap question in messages, extract answer from #### format into metadata.answer",
    "data/math/gsm8k/split/train_converted_d5_500.jsonl": "Requires conversion: wrap problem in messages, move answer into metadata.answer",
    "data/math/gsm8k/split/test_converted_500.jsonl": "Requires conversion: wrap problem in messages, move answer into metadata.answer",
}

EXCLUDED_TRAINING = {
    "m1_sft_train": "data/math/splits/sft_d5_500.jsonl",
    "dpo_v1_train": "data/math/splits/dpo_v1.jsonl",
    "dpo_v2_style_train": "data/math/splits/dpo_v2_style_train_449.jsonl",
    "dpo_v4_minimal_train": "data/math/pilots/dpo_v4_minimal_449.jsonl",
    "dpo_v4_minimal_pilot": "data/math/pilots/dpo_v4_minimal_pilot_30.jsonl",
    "boundary_only_dpo": "data/math/pilots/boundary_only_dpo_480.jsonl",
    "binary_prefix_dpo_ctrl": "data/math/pilots/binary_prefix_dpo_control_480.jsonl",
    "boundary_only_sft": "data/math/pilots/boundary_only_sft_480.jsonl",
}

EXCLUDED_EVAL = {
    "probe_30": "data/math/probes/dpo_v2_style_train_probe_30.jsonl",
    "probe_30_eval": "data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl",
    "stress_50": "data/math/splits/dpo_v2_style_stress_50.jsonl",
    "stress_eval_50": "data/math/splits/dpo_v2_style_stress_eval_50.jsonl",
}

EXCLUDED_PILOTS = {
    "rq_v2_problems_50": "data/math/pilots/dpo_rq_v2_problems_50.jsonl",
    "rq_v2b_problems_100": "data/math/pilots/dpo_rq_v2b_problems_100.jsonl",
    "rq_v2c_problems_100": "data/math/pilots/dpo_rq_v2c_problems_100.jsonl",
    "rq_v3_problems": "data/math/pilots/dpo_rq_v3/problems.jsonl",
    "rq_v3b_problems": "data/math/pilots/dpo_rq_v3_b/problems.jsonl",
    "rq_problems_50": "data/math/pilots/dpo_reasoning_quality_problems_50.jsonl",
}

ALL_JSONL_FILES = [
    # GSM8K raw
    "data/math/gsm8k/train.jsonl",
    "data/math/gsm8k/test.jsonl",
    # GSM8K split raw
    "data/math/gsm8k/split/train.jsonl",
    "data/math/gsm8k/split/test.jsonl",
    "data/math/gsm8k/split/val.jsonl",
    "data/math/gsm8k/split/review.jsonl",
    # GSM8K split converted
    "data/math/gsm8k/split/train_converted.jsonl",
    "data/math/gsm8k/split/test_converted.jsonl",
    "data/math/gsm8k/split/val_converted.jsonl",
    "data/math/gsm8k/split/review_converted.jsonl",
    "data/math/gsm8k/split/train_converted_d5_500.jsonl",
    "data/math/gsm8k/split/test_converted_500.jsonl",
    # Splits
    "data/math/splits/sft_v1.jsonl",
    "data/math/splits/sft_d5_500.jsonl",
    "data/math/splits/sft_v2_style_control_train_449.jsonl",
    "data/math/splits/dpo_v1.jsonl",
    "data/math/splits/dpo_v2_style_train_449.jsonl",
    "data/math/splits/dpo_v2_style_stress_50.jsonl",
    "data/math/splits/dpo_v2_style_stress_eval_50.jsonl",
    "data/math/splits/dpo_candidates.jsonl",
    # Probes
    "data/math/probes/dpo_v2_style_train_probe_30.jsonl",
    "data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl",
    # Pilots (selected)
    "data/math/pilots/dpo_reasoning_quality_problems_50.jsonl",
    "data/math/pilots/dpo_rq_v2_problems_50.jsonl",
    "data/math/pilots/dpo_rq_v2b_problems_100.jsonl",
    "data/math/pilots/dpo_rq_v2c_problems_100.jsonl",
    "data/math/pilots/dpo_rq_v3/problems.jsonl",
    "data/math/pilots/dpo_rq_v3_b/problems.jsonl",
    "data/math/pilots/boundary_only_sft_480.jsonl",
    "data/math/pilots/boundary_only_dpo_480.jsonl",
    "data/math/pilots/binary_prefix_dpo_control_480.jsonl",
    "data/math/pilots/dpo_v4_minimal_449.jsonl",
    "data/math/pilots/dpo_v4_minimal_pilot_30.jsonl",
]


def run_audit(project_root: Path) -> AuditReport:
    """Run the full audit. Returns structured report."""
    files: list[FileAudit] = []
    for rel in ALL_JSONL_FILES:
        path = project_root / rel
        if path.exists():
            fa = audit_file(path)
            # Mark convertible candidates
            if rel in CONVERSION_NOTES:
                fa.convertible_to_grpo = True
                fa.conversion_note = CONVERSION_NOTES[rel]
            files.append(fa)
        else:
            files.append(FileAudit(
                path=rel,
                record_count=0,
                top_level_keys_sample=[],
                has_messages=False,
                has_metadata_answer=False,
                has_problem_id=False,
                problem_id_source="none",
                has_answer=False,
                answer_source="none",
                has_source_tag=False,
                has_level=False,
                grpo_compatible=False,
                notes=["file not found"],
            ))

    # Build excluded sets grouped by category
    excluded_by_group: dict[str, set[str]] = {
        "training": set(),
        "eval": set(),
        "pilots": set(),
    }
    per_excluded_detail: dict[str, tuple[set[str], set[str]]] = {}

    for group_name, group in [("training", EXCLUDED_TRAINING), ("eval", EXCLUDED_EVAL), ("pilots", EXCLUDED_PILOTS)]:
        for name, rel in group.items():
            path = project_root / rel
            if path.exists():
                ids, hashes = load_ids_and_hashes(path)
                per_excluded_detail[name] = (hashes, ids)
                excluded_by_group[group_name] |= hashes

    # GSM8K-50 eval (first 50 of test_converted_500)
    eval_path = project_root / "data/math/gsm8k/split/test_converted_500.jsonl"
    if eval_path.exists():
        try:
            recs = load_json_record_stream(eval_path)[:50]
            ids50: set[str] = set()
            hashes50: set[str] = set()
            for r in recs:
                pid = _extract_problem_id(r)
                if pid:
                    ids50.add(pid)
                text = _extract_problem_text(r)
                if text:
                    hashes50.add(canonical_question_hash(text))
            per_excluded_detail["gsm8k_50_eval_first50"] = (hashes50, ids50)
            excluded_by_group["eval"] |= hashes50
        except Exception:
            pass

    # Union of all excluded hashes
    excluded_all_hashes: set[str] = set()
    excluded_all_ids: set[str] = set()
    for hashes, ids in per_excluded_detail.values():
        excluded_all_hashes |= hashes
        excluded_all_ids |= ids

    # Audit each candidate source
    candidate_audits: list[CandidateAudit] = []
    for rel in CANDIDATE_SOURCES:
        path = project_root / rel
        if not path.exists():
            continue
        c_ids, c_hashes = load_ids_and_hashes(path)
        ca = audit_candidate_source(
            candidate_name=rel,
            candidate_hashes=c_hashes,
            candidate_ids=c_ids,
            excluded_by_group=excluded_by_group,
            excluded_all_hashes=excluded_all_hashes,
            excluded_all_ids=excluded_all_ids,
            per_excluded_detail=per_excluded_detail,
        )
        candidate_audits.append(ca)

    # Summary
    grpo_ready = [f for f in files if f.grpo_compatible and f.record_count > 0]
    convertible = [f for f in files if f.convertible_to_grpo and f.record_count > 0]

    summary = {
        "total_files_audited": len(files),
        "grpo_direct_compatible_files": len(grpo_ready),
        "grpo_direct_compatible_records": sum(f.record_count for f in grpo_ready),
        "grpo_convertible_files": len(convertible),
        "hendrycks_math_status": "NOT_DOWNLOADED",
        "openr1_math_status": "NOT_DOWNLOADED",
        "math_level_3_5_file_exists": (project_root / "data/math/splits/math_level_3_5.jsonl").exists(),
    }

    return AuditReport(files=files, candidate_audits=candidate_audits, summary=summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_report(report: AuditReport) -> None:
    """Print human-readable audit report."""
    print("=" * 70)
    print("M2.5 GRPO Data Source Audit")
    print("=" * 70)

    # Section 1: File audit
    print("\n## 1. Local JSONL Data Sources\n")
    print(f"{'File':<60} {'Recs':>5} {'GRPO':>5} {'Valid':>5} {'Invalid':>7}")
    print("-" * 85)
    for f in report.files:
        short = f.path.replace("data/math/", "")
        grpo = "YES" if f.grpo_compatible else ("CVT" if f.convertible_to_grpo else "NO")
        v = f.grpo_schema.valid_count
        inv = f.grpo_schema.invalid_count
        print(f"{short:<60} {f.record_count:>5} {grpo:>5} {v:>5} {inv:>7}")
        if f.grpo_schema.invalid_reasons:
            for reason, count in f.grpo_schema.invalid_reasons.items():
                print(f"  {'':60} [{count}] {reason}")
        if f.conversion_note:
            print(f"  {'':60} NOTE: {f.conversion_note}")

    # Section 2: Candidate overlap
    print("\n## 2. Candidate Source Overlap (canonical question hash)\n")
    for ca in report.candidate_audits:
        short = ca.candidate_name.replace("data/math/", "")
        print(f"### {short}")
        print(f"  Total questions:     {ca.total_records}")
        print(f"  Excluded (training): {ca.excluded_by_training}")
        print(f"  Excluded (eval):     {ca.excluded_by_eval}")
        print(f"  Excluded (pilots):   {ca.excluded_by_pilots}")
        print(f"  Excluded (union):    {ca.excluded_by_union}")
        print(f"  Clean remaining:     {ca.clean_remaining}")
        print()
        print(f"  {'Excluded Set':<35} {'Canon':>6} {'ID':>6} {'Hash':>6}")
        print(f"  {'-'*55}")
        for ov in ca.per_excluded_set:
            if ov.canonical_overlap_count > 0 or ov.id_match_count > 0:
                print(f"  {ov.excluded_name:<35} {ov.canonical_overlap_count:>6} {ov.id_match_count:>6} {ov.text_hash_match_count:>6}")
        print()

    # Section 3: Schema compatibility
    print("\n## 3. GRPO Schema Compatibility\n")
    grpo_ready = [f for f in report.files if f.grpo_compatible and f.record_count > 0]
    print(f"  Directly GRPO-compatible (all records valid): {len(grpo_ready)}")
    for f in grpo_ready:
        short = f.path.replace("data/math/", "")
        print(f"    {short} ({f.record_count} records)")

    convertible = [f for f in report.files if f.convertible_to_grpo and f.record_count > 0]
    print(f"\n  Convertible to GRPO (deterministic conversion): {len(convertible)}")
    for f in convertible:
        short = f.path.replace("data/math/", "")
        print(f"    {short} ({f.record_count} records): {f.conversion_note}")

    # Section 4: math_level_3_5
    print("\n## 4. math_level_3_5 Status\n")
    print(f"  File exists: {report.summary['math_level_3_5_file_exists']}")
    print(f"  Hendrycks MATH: {report.summary['hendrycks_math_status']}")
    print(f"  OpenR1-Math-220k: {report.summary['openr1_math_status']}")

    # Section 5: Summary
    print("\n## 5. Summary\n")
    print(f"  Total files audited: {report.summary['total_files_audited']}")
    print(f"  Directly GRPO-compatible: {report.summary['grpo_direct_compatible_files']} files / {report.summary['grpo_direct_compatible_records']} records")
    print(f"  GRPO-convertible: {report.summary['grpo_convertible_files']} files")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="M2.5 GRPO Data Source Audit")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parent.parent.parent)
    args = parser.parse_args()

    report = run_audit(args.project_root)

    if args.json:
        output = {
            "files": [asdict(f) for f in report.files],
            "candidate_audits": [asdict(c) for c in report.candidate_audits],
            "summary": report.summary,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
