#!/usr/bin/env python3
"""
SFT sanity 空跑脚本。

模拟 SFT 训练的最小日志闭环，不下载模型，不接 MLX。

用法：
    python scripts/sanity/sanity_sft.py \
        --base configs/base.yaml \
        --override configs/qwen3_0_6b/sft_math.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sanity.common import run_fake_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Sanity 空跑")
    parser.add_argument("--base", required=True, help="Base 配置文件路径")
    parser.add_argument("--override", default=None, help="Experiment 覆盖配置文件路径")
    args = parser.parse_args()

    train_rows = [
        {"loss": 2.50},
        {"loss": 2.30},
        {"loss": 2.10},
    ]
    eval_rows = [
        {"format_adherence": 0.40, "answer_extraction_success": 0.35},
        {"format_adherence": 0.55, "answer_extraction_success": 0.48},
    ]
    sample_rows = [
        {
            "sample_id": "math_000001",
            "prompt": "Solve: What is 2 + 3?",
            "before": "The answer is 5",
            "after": "<think>2 + 3 = 5</think>\n\nThe answer is 5",
            "reference_answer": "5",
            "before_score": 0.0,
            "after_score": 1.0,
            "tags": ["format_improved", "answer_extractable"],
            "notes": "",
        }
    ]

    run_dir = run_fake_pipeline(
        args.base,
        args.override,
        method="sft",
        train_rows=train_rows,
        eval_rows=eval_rows,
        sample_rows=sample_rows,
    )
    print(f"[sanity] SFT fake run completed: {run_dir}")


if __name__ == "__main__":
    main()
