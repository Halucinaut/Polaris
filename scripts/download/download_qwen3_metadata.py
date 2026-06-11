#!/usr/bin/env python3
"""
Download Qwen3-0.6B metadata files from HuggingFace without downloading model weights.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Qwen/Qwen3-0.6B"
TARGET_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen3_0_6b" / "metadata"

# Files to download (metadata/config only, NO weights)
METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "README.md",
]


def download_metadata():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {TARGET_DIR}")
    print(f"Repository: {REPO_ID}")
    print("-" * 50)

    # List all files in repo to show what we're skipping
    all_files = list(list_repo_files(REPO_ID))
    weight_files = [f for f in all_files if any(
        f.endswith(ext) for ext in [".safetensors", ".bin", ".pt", ".ckpt", ".msgpack"]
    )]

    print(f"Total files in repo: {len(all_files)}")
    print(f"Weight files detected (SKIPPED): {len(weight_files)}")
    for wf in weight_files:
        print(f"  - {wf}")
    print("-" * 50)

    success = []
    failed = []

    for filename in METADATA_FILES:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=TARGET_DIR,
                local_dir_use_symlinks=False,
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
