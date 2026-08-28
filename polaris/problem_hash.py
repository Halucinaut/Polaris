"""
Canonical problem hashing for cross-file identity.

Every problem in the Polaris pipeline gets a deterministic hash derived
from its *normalized* text.  Two problems with identical whitespace-
collapsed, lowercased text always produce the same hash regardless of
source file, index, or problem_id scheme.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def normalize_problem(text: str) -> str:
    """Normalize problem text for hashing.

    - Strip leading/trailing whitespace
    - Collapse internal whitespace runs to single space
    - Lowercase
    """
    return " ".join(text.lower().split())


def problem_hash(text: str) -> str:
    """SHA256 of normalized text, truncated to 16 hex chars."""
    return hashlib.sha256(normalize_problem(text).encode()).hexdigest()[:16]


def build_exclude_hashes(paths: list[Path]) -> set[str]:
    """Build a set of problem hashes from Probe-30 / Stress-50 eval files.

    Each file is JSONL with a ``problem`` field.
    """
    hashes: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("problem", "")
                if text:
                    hashes.add(problem_hash(text))
    return hashes


def assert_no_overlap(
    candidate_hashes: set[str],
    exclude_hashes: set[str],
    label: str = "candidates",
) -> None:
    """Assert zero overlap between candidate and exclude sets."""
    overlap = candidate_hashes & exclude_hashes
    if overlap:
        raise AssertionError(
            f"{label} overlap with exclude set: {len(overlap)} hashes: "
            f"{sorted(overlap)[:5]}..."
        )
