#!/usr/bin/env python3
"""
DPO sanity 空跑脚本。

模拟 DPO 训练的最小日志闭环，不下载模型，不接 MLX。

用法：
    python scripts/sanity/sanity_dpo.py \
        --base configs/base.yaml \
        --override configs/qwen3_0_6b/dpo_math.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sanity.common import run_fake_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO Sanity 空跑")
    parser.add_argument("--base", required=True, help="Base 配置文件路径")
    parser.add_argument("--override", default=None, help="Experiment 覆盖配置文件路径")
    args = parser.parse_args()

    train_rows = [
        {"chosen_logprob": -1.20, "rejected_logprob": -1.45, "logprob_margin": 0.25},
        {"chosen_logprob": -1.10, "rejected_logprob": -1.50, "logprob_margin": 0.40},
    ]
    eval_rows = [
        {"preference_accuracy": 0.52, "response_length_shift": 12.0},
        {"preference_accuracy": 0.58, "response_length_shift": 18.0},
    ]
    sample_rows = [
        {
            "sample_id": "dpo_000001",
            "prompt": "Solve: What is 7 * 8?",
            "before": "56",
            "after": "<think>7 * 8 = 56</think>\n\n56",
            "reference_answer": "56",
            "before_score": 0.5,
            "after_score": 1.0,
            "tags": ["preference_margin_improved", "length_shift_watch"],
            "notes": "",
        }
    ]

    run_dir = run_fake_pipeline(
        args.base,
        args.override,
        method="dpo",
        train_rows=train_rows,
        eval_rows=eval_rows,
        sample_rows=sample_rows,
    )
    print(f"[sanity] DPO fake run completed: {run_dir}")


if __name__ == "__main__":
    main()
