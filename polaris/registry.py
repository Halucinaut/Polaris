"""
Polaris Run Registry 模块。

提供能力：
1. 生成递增 run_id（格式：000001_<run_name>）
2. 创建 run 目录结构
3. 写入 run_meta.yaml
4. 更新 run 状态

用法：
    python -m polaris.registry --runs-dir runs --list
    python -m polaris.registry --runs-dir runs --run-id 000001_sft_math_0_6b --status running
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from polaris.config import build_config, freeze_config

VALID_STATUSES = {"created", "running", "completed", "failed", "aborted"}

RUN_SUBDIRS = ["metrics", "samples", "checkpoints", "logs"]


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _extract_tags(config: dict[str, Any]) -> dict[str, Any]:
    """从配置中提取标签信息。"""
    tags = {}
    training = config.get("training", {})
    if training.get("method"):
        tags["method"] = training["method"]
    data = config.get("data", {})
    if data.get("name"):
        tags["domain"] = data["name"].split("_")[0] if "_" in data.get("name", "") else data.get("name")
    model = config.get("model", {})
    model_name = model.get("name", "")
    if model_name:
        size_match = re.search(r"(\d+\.?\d*)[Bb]", model_name)
        if size_match:
            tags["model_size"] = size_match.group(1).lower() + "b"
    return tags


def _find_next_run_id(runs_dir: Path, run_name: str) -> str:
    """
    扫描 runs_dir 下已有的 run 目录，找到最大编号，返回下一个递增编号。
    格式：000001_<run_name>
    """
    max_num = 0
    pattern = re.compile(r"^(\d{6})_.*$")
    if runs_dir.exists():
        for entry in runs_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
    next_num = max_num + 1
    return f"{next_num:06d}_{run_name}"


def create_run(
    base_path: str | Path,
    override_path: str | Path | None = None,
    runs_dir: str | Path = "runs",
) -> Path:
    """
    创建一个新的 run。

    流程：
    1. 合并 base + override 得到完整 config
    2. 从 config.run.name 生成 run_name，计算下一个 run_id
    3. 创建 run 目录及子目录（metrics/ samples/ checkpoints/ logs/）
    4. 冻结 config.yaml
    5. 写入 run_meta.yaml

    返回新 run 目录的 Path。
    """
    config = build_config(base_path, override_path)
    run_name = config.get("run", {}).get("name")
    if not run_name:
        raise ValueError("Config 中缺少 run.name 字段，无法生成 run_id")

    runs_dir_path = Path(runs_dir)
    run_id = _find_next_run_id(runs_dir_path, run_name)
    run_dir = runs_dir_path / run_id

    if run_dir.exists():
        raise FileExistsError(f"Run 目录已存在：{run_dir}")

    run_dir.mkdir(parents=True)
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir()
        (run_dir / subdir / ".gitkeep").touch()

    freeze_config(config, run_dir)

    meta = {
        "run_id": run_id,
        "run_name": run_name,
        "status": "created",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "config_path": str(run_dir / "config.yaml"),
        "tags": _extract_tags(config),
        "notes": "",
    }
    meta_path = run_dir / "run_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return run_dir


def list_runs(runs_dir: str | Path = "runs") -> list[dict[str, Any]]:
    """
    列出 runs_dir 下所有 run 的摘要信息。
    返回按 run_id 排序的字典列表。
    """
    runs_dir_path = Path(runs_dir)
    if not runs_dir_path.exists():
        return []

    runs = []
    for entry in sorted(runs_dir_path.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "run_meta.yaml"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        runs.append({
            "run_id": meta.get("run_id", entry.name),
            "status": meta.get("status", "unknown"),
            "created_at": meta.get("created_at", ""),
        })
    return runs


def update_run_status(
    run_id: str,
    status: str,
    runs_dir: str | Path = "runs",
) -> Path:
    """
    更新指定 run 的状态和 updated_at 时间戳。
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"无效状态：{status}。有效值：{VALID_STATUSES}")

    runs_dir_path = Path(runs_dir)
    run_dir = runs_dir_path / run_id
    meta_path = run_dir / "run_meta.yaml"

    if not meta_path.exists():
        raise FileNotFoundError(f"Run meta 文件不存在：{meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    meta["status"] = status
    meta["updated_at"] = _now_iso()

    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return meta_path


def _print_run_table(runs: list[dict[str, Any]]) -> None:
    """打印 run 列表为对齐的表格。"""
    if not runs:
        print("No runs found.")
        return
    id_width = max(len(r["run_id"]) for r in runs)
    status_width = max(len(r["status"]) for r in runs)
    header = f"{'run_id':<{id_width}}  {'status':<{status_width}}  created_at"
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['run_id']:<{id_width}}  {r['status']:<{status_width}}  {r['created_at']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polaris Run Registry 工具")
    parser.add_argument("--runs-dir", default="runs", help="Runs 根目录")
    parser.add_argument("--list", action="store_true", help="列出所有 run")
    parser.add_argument("--run-id", default=None, help="指定 run_id")
    parser.add_argument("--status", default=None, help="更新 run 状态")
    args = parser.parse_args()

    if args.list:
        runs = list_runs(args.runs_dir)
        _print_run_table(runs)
        return

    if args.run_id and args.status:
        meta_path = update_run_status(args.run_id, args.status, args.runs_dir)
        print(f"Updated status to '{args.status}' for run {args.run_id}")
        print(f"Meta file: {meta_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
