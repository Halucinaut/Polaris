#!/usr/bin/env python3
"""
PPO sanity 空跑脚本。

模拟 PPO 训练的最小日志闭环，覆盖 M2.5 关心的链路字段。
不下载模型，不接 MLX。

用法：
    python scripts/sanity/sanity_ppo.py \
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
    parser = argparse.ArgumentParser(description="PPO Sanity 空跑")
    parser.add_argument("--base", required=True, help="Base 配置文件路径")
    parser.add_argument("--override", default=None, help="Experiment 覆盖配置文件路径")
    args = parser.parse_args()

    train_rows = [
        {
            "rollout_count": 20,
            "group_size": 2,
            "reward_mean": 0.25,
            "policy_logprob_mean": -1.8,
            "ref_logprob_mean": -1.9,
            "kl": 0.04,
            "loss": 0.32,
            "resume_ok": True,
            "checkpoint_written": True,
        },
    ]
    eval_rows = [
        {"pass_at_1": 0.22, "invalid_output_rate": 0.16, "avg_completion_length": 132},
    ]
    sample_rows = [
        {
            "sample_id": "ppo_000001",
            "prompt": "Solve: What is 9 + 6?",
            "before": "15",
            "after": "<think>9 + 6 = 15</think>\n\n15",
            "reference_answer": "15",
            "before_score": 0.0,
            "after_score": 1.0,
            "tags": ["rollout_ok", "reward_ok", "kl_logged", "checkpoint_ok", "resume_ok"],
            "notes": "",
        }
    ]

    run_dir = run_fake_pipeline(
        args.base,
        args.override,
        method="ppo",
        train_rows=train_rows,
        eval_rows=eval_rows,
        sample_rows=sample_rows,
    )
    print(f"[sanity] PPO fake run completed: {run_dir}")


if __name__ == "__main__":
    main()
