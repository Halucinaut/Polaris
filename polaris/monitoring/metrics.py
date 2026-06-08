"""
Polaris Metrics 日志协议。

提供能力：
1. 追加 JSONL 指标记录（train / eval / hardware）
2. 追加 sample diff 记录
3. 读取 JSONL 文件

用法：
    from polaris.monitoring.metrics import append_metric, append_sample_diff, read_jsonl

    append_metric("runs/000001", "train", step=1, loss=2.31, learning_rate=0.0001)
    append_sample_diff("runs/000001", sample_id="math_000001", prompt="...", before="...", after="...")
    records = read_jsonl("runs/000001/metrics/train_metrics.jsonl")

命令行：
    python -m polaris.monitoring.metrics \
        --run-dir runs/000001_sft_math_0_6b \
        --kind train \
        --step 1 \
        --metric loss=2.31 \
        --metric learning_rate=0.0001
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _samples_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_metric(
    run_dir: str | Path,
    kind: str,
    step: int,
    **metrics: Any,
) -> Path:
    """
    追加一条指标记录到对应 JSONL 文件。

    kind 支持：train, eval, hardware
    文件路径：runs/{run_id}/metrics/{kind}_metrics.jsonl
    """
    if kind not in {"train", "eval", "hardware"}:
        raise ValueError(f"无效 kind：{kind}。支持：train, eval, hardware")

    metrics_path = _metrics_dir(run_dir) / f"{kind}_metrics.jsonl"
    record = {
        "step": step,
        "split": kind,
        "metrics": metrics,
        "timestamp": _now_iso(),
    }
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics_path


def append_sample_diff(
    run_dir: str | Path,
    sample_id: str,
    prompt: str,
    before: str,
    after: str,
    reference_answer: str = "",
    before_score: float | None = None,
    after_score: float | None = None,
    tags: list[str] | None = None,
    notes: str = "",
) -> Path:
    """
    追加一条 sample diff 记录到 samples/sample_diff.jsonl。
    """
    samples_path = _samples_dir(run_dir) / "sample_diff.jsonl"
    record = {
        "sample_id": sample_id,
        "prompt": prompt,
        "before": before,
        "after": after,
        "reference_answer": reference_answer,
        "before_score": before_score,
        "after_score": after_score,
        "tags": tags or [],
        "notes": notes,
        "timestamp": _now_iso(),
    }
    with open(samples_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return samples_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    读取 JSONL 文件，返回记录列表。文件不存在返回空列表。
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_metric_arg(arg: str) -> tuple[str, Any]:
    """解析 --metric key=value 参数。"""
    if "=" not in arg:
        raise ValueError(f"metric 参数格式错误：{arg}，应为 key=value")
    key, value = arg.split("=", 1)
    # 尝试解析为数字
    try:
        if "." in value:
            parsed = float(value)
        else:
            parsed = int(value)
    except ValueError:
        parsed = value
    return key, parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Polaris Metrics 日志工具")
    parser.add_argument("--run-dir", required=True, help="Run 目录路径")
    parser.add_argument("--kind", required=True, choices=["train", "eval", "hardware"], help="指标类型")
    parser.add_argument("--step", type=int, required=True, help="Step 编号")
    parser.add_argument("--metric", action="append", default=[], help="指标键值对，格式：key=value")
    args = parser.parse_args()

    metrics = {}
    for m in args.metric:
        key, value = _parse_metric_arg(m)
        metrics[key] = value

    path = append_metric(args.run_dir, args.kind, args.step, **metrics)
    print(f"Appended to: {path}")


if __name__ == "__main__":
    main()
