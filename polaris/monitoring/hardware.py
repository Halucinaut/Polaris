"""
Polaris Hardware 日志协议。

提供能力：
1. 采集当前硬件/环境快照
2. 追加到 run 的 hardware_log.jsonl

用法：
    from polaris.monitoring.hardware import snapshot_hardware

    info = snapshot_hardware()
    # info 为字典，可直接写入日志

命令行：
    python -m polaris.monitoring.hardware --run-dir runs/000001_sft_math_0_6b
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_import(module_name: str) -> Any:
    """安全导入模块，失败返回 None。"""
    try:
        return __import__(module_name)
    except ImportError:
        return None


def snapshot_hardware() -> dict[str, Any]:
    """
    采集当前环境快照。
    不依赖 MLX/PyTorch，如果未安装则返回 false。
    """
    mlx = _safe_import("mlx")
    torch = _safe_import("torch")

    mlx_available = mlx is not None
    torch_available = torch is not None
    mps_available = False

    if torch_available:
        try:
            mps_available = torch.backends.mps.is_available()
        except Exception:
            mps_available = False

    # 获取内存信息（优先使用 psutil，否则跳过）
    memory_total_gb = None
    psutil = _safe_import("psutil")
    if psutil is not None:
        try:
            memory_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass

    return {
        "timestamp": _now_iso(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "memory_total_gb": memory_total_gb,
        "mlx_available": mlx_available,
        "torch_available": torch_available,
        "mps_available": mps_available,
    }


def append_hardware_log(run_dir: str | Path, snapshot: dict[str, Any] | None = None) -> Path:
    """
    追加一条硬件快照到 runs/{run_id}/metrics/hardware_log.jsonl。
    """
    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = metrics_dir / "hardware_log.jsonl"

    if snapshot is None:
        snapshot = snapshot_hardware()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Polaris Hardware 日志工具")
    parser.add_argument("--run-dir", required=True, help="Run 目录路径")
    args = parser.parse_args()

    snapshot = snapshot_hardware()
    path = append_hardware_log(args.run_dir, snapshot)
    print(f"Hardware snapshot appended to: {path}")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
