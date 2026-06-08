#!/usr/bin/env python3
"""
为指定 run 生成实验文档骨架。

读取 run_meta.yaml 和 config.yaml，用 docs/templates/ 下的模板生成：
- experiment_card.md
- metric_report.md
- sample_diff.md
- failure_note.md
- run_report.md

用法：
    python scripts/generate_experiment_card.py --run-dir runs/000001_sft_math_0_6b
    python scripts/generate_experiment_card.py --run-dir runs/000001_sft_math_0_6b --overwrite
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


TEMPLATES_DIR = Path(__file__).parent.parent / "docs" / "templates"
OUTPUT_FILES = [
    "experiment_card.md",
    "metric_report.md",
    "sample_diff.md",
    "failure_note.md",
    "run_report.md",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _extract_model_size(model_name: str) -> str:
    match = re.search(r"(\d+\.?\d*)[Bb]", model_name)
    return match.group(1).lower() + "b" if match else "unknown"


def _build_context(run_meta: dict[str, Any], config: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """从 run_meta 和 config 中提取模板替换所需的上下文。"""
    ctx: dict[str, str] = {}

    # run_meta 字段
    ctx["run_id"] = str(run_meta.get("run_id", ""))
    ctx["run_name"] = str(run_meta.get("run_name", ""))
    ctx["status"] = str(run_meta.get("status", ""))
    ctx["created_at"] = str(run_meta.get("created_at", ""))
    ctx["updated_at"] = str(run_meta.get("updated_at", ""))
    ctx["config_path"] = str(run_meta.get("config_path", ""))

    # tags
    tags = run_meta.get("tags", {})
    ctx["method"] = str(tags.get("method", ""))
    ctx["domain"] = str(tags.get("domain", ""))
    ctx["model_size"] = str(tags.get("model_size", ""))

    # config 字段
    run_cfg = config.get("run", {})
    ctx["run_name"] = ctx.get("run_name") or str(run_cfg.get("name", ""))

    model_cfg = config.get("model", {})
    ctx["model_name"] = str(model_cfg.get("name", ""))
    if not ctx["model_size"]:
        ctx["model_size"] = _extract_model_size(ctx["model_name"])

    data_cfg = config.get("data", {})
    ctx["dataset_name"] = str(data_cfg.get("name", ""))
    ctx["dataset_split"] = str(data_cfg.get("split", ""))

    training_cfg = config.get("training", {})
    ctx["max_seq_length"] = str(training_cfg.get("max_seq_length", ""))
    ctx["batch_size"] = str(training_cfg.get("batch_size", ""))

    lora_cfg = config.get("lora", {})
    ctx["lora_r"] = str(lora_cfg.get("r", ""))
    ctx["lora_alpha"] = str(lora_cfg.get("alpha", ""))
    ctx["lora_dropout"] = str(lora_cfg.get("dropout", ""))

    ctx["dtype"] = str(model_cfg.get("dtype", ""))

    hardware_cfg = config.get("hardware", {})
    ctx["hardware_device"] = str(hardware_cfg.get("device", ""))

    ctx["run_dir"] = str(run_dir)

    return ctx


def _render_template(template_path: Path, context: dict[str, str]) -> str:
    """用 {{ key }} 占位符做简单字符串替换。找不到的保留原样。"""
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _replacer(match: re.Match) -> str:
        key = match.group(1).strip()
        return context.get(key, match.group(0))

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", _replacer, content)


def generate_docs(run_dir: str | Path, overwrite: bool = False) -> list[Path]:
    """
    为指定 run 生成所有文档骨架。
    返回已生成（或跳过）的文件路径列表。
    """
    run_path = Path(run_dir)
    meta_path = run_path / "run_meta.yaml"
    config_path = run_path / "config.yaml"

    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.yaml 不存在：{meta_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml 不存在：{config_path}")

    run_meta = _load_yaml(meta_path)
    config = _load_yaml(config_path)
    context = _build_context(run_meta, config, run_path)

    generated: list[Path] = []
    for filename in OUTPUT_FILES:
        template_path = TEMPLATES_DIR / filename
        if not template_path.exists():
            print(f"[warn] Template not found: {template_path}")
            continue

        output_path = run_path / filename
        if output_path.exists() and not overwrite:
            print(f"[skip] {output_path}")
            generated.append(output_path)
            continue

        rendered = _render_template(template_path, context)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"[gen] {output_path}")
        generated.append(output_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="生成实验文档骨架")
    parser.add_argument("--run-dir", required=True, help="Run 目录路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文档")
    args = parser.parse_args()

    generate_docs(args.run_dir, args.overwrite)


if __name__ == "__main__":
    main()
