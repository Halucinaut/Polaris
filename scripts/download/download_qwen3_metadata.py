#!/usr/bin/env python3
"""
Download Qwen3-0.6B metadata files from HuggingFace without downloading model weights.

Usage:
    python scripts/download/download_qwen3_metadata.py
    HF_ENDPOINT=https://huggingface.co python scripts/download/download_qwen3_metadata.py  # use official source
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen3-0.6B"
TARGET_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen3_0_6b" / "metadata"

# Files to download (metadata/config only, NO weights)
# Must match the official Qwen/Qwen3-0.6B repo (excluding .safetensors)
METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "README.md",
    ".gitattributes",
]


def download_metadata():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Repository: {REPO_ID}")
    print(f"Endpoint: {endpoint}")
    print("-" * 50)

    success = []
    failed = []

    for filename in METADATA_FILES:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=TARGET_DIR,
                endpoint=endpoint,
            )
            print(f"[OK] {filename}")
            success.append(filename)
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")
            failed.append(filename)

    print("-" * 50)
    print(f"Downloaded: {len(success)}/{len(METADATA_FILES)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    return len(failed) == 0


if __name__ == "__main__":
    ok = download_metadata()
    sys.exit(0 if ok else 1)
