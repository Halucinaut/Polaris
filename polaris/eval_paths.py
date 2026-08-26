"""Filesystem helpers shared by evaluation entry points."""

from pathlib import Path


def prepare_output_path(value: str | None) -> Path | None:
    """Create an evaluation output directory before per-split files are written."""
    if value is None:
        return None
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
