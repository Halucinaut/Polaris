#!/usr/bin/env python3
"""
创建一个新的 Polaris run。

用法：
    python scripts/create_run.py \
        --base configs/base.yaml \
        --override configs/qwen3_0_6b/sft_math.yaml

输出：
    runs/000001_sft_math_0_6b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polaris.registry import create_run


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 Polaris run")
    parser.add_argument("--base", required=True, help="Base 配置文件路径")
    parser.add_argument("--override", default=None, help="Experiment 覆盖配置文件路径")
    parser.add_argument("--runs-dir", default="runs", help="Runs 根目录")
    args = parser.parse_args()

    run_dir = create_run(args.base, args.override, args.runs_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
