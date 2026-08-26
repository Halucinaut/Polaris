"""Compatibility loader for newline-delimited and pretty-printed JSON records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_record_stream(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, a JSON array, or whitespace-separated JSON objects.

    Dataset generation normally writes compact JSONL. Historical Polaris
    artifacts also contain pretty-printed JSON objects separated by whitespace.
    Parsing with ``JSONDecoder.raw_decode`` supports all three forms while
    preserving the on-disk artifact exactly as recorded.
    """
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    position = 0

    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break

        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON record in dataset {path} at character {position}: {exc.msg}"
            ) from exc

        if isinstance(value, list):
            records.extend(value)
        else:
            records.append(value)

    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Dataset must contain JSON objects: {path}")
    return records
