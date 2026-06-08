#!/usr/bin/env python3
"""
Sanity 脚本公共逻辑。

提供 fake run 的标准流程：
1. 创建 run（registry）
2. 更新状态为 running
3. 写入 hardware snapshot
4. 写入 fake train/eval metrics
5. 写入 sample diff
6. 生成文档骨架
7. 更新状态为 completed（或 failed）
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from polaris.registry import create_run, update_run_status
from polaris.monitoring.metrics import append_metric, append_sample_diff
from polaris.monitoring.hardware import append_hardware_log
from scripts.generate_experiment_card import generate_docs


def create_fake_run(
    base_path: str,
    override_path: str | None,
    method: str,
) -> Path:
    """
    创建一个新的 fake run，返回 run 目录路径。
    """
    run_dir = create_run(base_path, override_path, runs_dir="runs")
    run_id = run_dir.name
    update_run_status(run_id, "running", runs_dir="runs")
    print(f"[fake] Run created: {run_dir}")
    return run_dir


def write_fake_metrics(
    run_dir: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> None:
    """
    写入 fake train metrics、eval metrics 和 sample diffs。
    """
    for i, row in enumerate(train_rows, start=1):
        append_metric(str(run_dir), "train", step=i, **row)
        print(f"[fake] Train step {i}: {row}")

    for i, row in enumerate(eval_rows, start=1):
        append_metric(str(run_dir), "eval", step=i, **row)
        print(f"[fake] Eval step {i}: {row}")

    for row in sample_rows:
        append_sample_diff(
            str(run_dir),
            sample_id=row["sample_id"],
            prompt=row["prompt"],
            before=row["before"],
            after=row["after"],
            reference_answer=row.get("reference_answer", ""),
            before_score=row.get("before_score"),
            after_score=row.get("after_score"),
            tags=row.get("tags", []),
            notes=row.get("notes", ""),
        )
        print(f"[fake] Sample diff: {row['sample_id']}")

    append_hardware_log(str(run_dir))
    print("[fake] Hardware snapshot appended")


def finalize_fake_run(
    run_dir: Path,
    status: str = "completed",
) -> None:
    """
    生成文档骨架并更新 run 状态。
    """
    generate_docs(str(run_dir))
    update_run_status(run_dir.name, status, runs_dir="runs")
    print(f"[fake] Run finalized with status: {status}")


def run_fake_pipeline(
    base_path: str,
    override_path: str | None,
    method: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> Path:
    """
    完整的 fake run 流水线。
    如果中间报错，状态改为 failed，并把错误信息写入 logs/error.txt。
    """
    run_dir = create_fake_run(base_path, override_path, method)
    try:
        write_fake_metrics(run_dir, train_rows, eval_rows, sample_rows)
        finalize_fake_run(run_dir, "completed")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / "error.txt", "w", encoding="utf-8") as f:
            f.write(error_msg)
        update_run_status(run_dir.name, "failed", runs_dir="runs")
        print(f"[fake] Run failed: {run_dir}")
        raise
    return run_dir
