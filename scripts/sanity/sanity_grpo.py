#!/usr/bin/env python3
"""
GRPO sanity 空跑脚本。

模拟 GRPO 训练的最小日志闭环，不下载模型，不接 MLX。

用法：
    python scripts/sanity/sanity_grpo.py \
        --base configs/base.yaml \
        --override configs/qwen3_0_6b/grpo_math.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sanity.common import run_fake_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO Sanity 空跑")
    parser.add_argument("--base", required=True, help="Base 配置文件路径")
    parser.add_argument("--override", default=None, help="Experiment 覆盖配置文件路径")
    args = parser.parse_args()

    train_rows = [
        {"reward_mean": 0.20, "reward_std": 0.10, "kl": 0.01, "entropy": 3.8},
        {"reward_mean": 0.28, "reward_std": 0.12, "kl": 0.03, "entropy": 3.5},
    ]
    eval_rows = [
        {"pass_at_1": 0.20, "invalid_output_rate": 0.18, "avg_completion_length": 128},
        {"pass_at_1": 0.24, "invalid_output_rate": 0.15, "avg_completion_length": 136},
    ]
    sample_rows = [
        {
            "sample_id": "grpo_000001",
            "prompt": "Solve: What is 12 / 4?",
            "before": "3",
            "after": "<think>12 / 4 = 3</think>\n\n3",
            "reference_answer": "3",
            "before_score": 0.0,
            "after_score": 1.0,
            "tags": ["reward_gain", "kl_within_range"],
            "notes": "",
        }
    ]

    run_dir = run_fake_pipeline(
        args.base,
        args.override,
        method="grpo",
        train_rows=train_rows,
        eval_rows=eval_rows,
        sample_rows=sample_rows,
    )
    print(f"[sanity] GRPO fake run completed: {run_dir}")


if __name__ == "__main__":
    main()
