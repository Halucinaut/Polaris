"""
Polaris 配置管理模块。

提供能力：
1. 读取 YAML 配置文件
2. 合并 base config 与 experiment config（experiment 覆盖 base）
3. 保存冻结后的完整 config 到 run 目录

用法：
    python -m polaris.config --base configs/base.yaml --override configs/qwen3_0_6b/sft_math.yaml --print
    python -m polaris.config --base configs/base.yaml --override configs/qwen3_0_6b/sft_math.yaml --freeze runs/test_config_freeze
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件并返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    递归合并两个字典。override 中的值会覆盖 base 中的值。
    如果两边都是 dict，则递归合并；否则 override 优先。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def freeze_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """
    将合并后的完整 config 保存到指定目录的 config.yaml 中。
    不会修改原始配置文件。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_file = output_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return config_file


def build_config(base_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    """
    读取 base config，可选地合并 override config，返回完整配置字典。
    """
    base = load_yaml(base_path)
    if override_path is None:
        return base
    override = load_yaml(override_path)
    return deep_merge(base, override)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polaris 配置管理工具")
    parser.add_argument("--base", required=True, help="Base 配置文件路径")
    parser.add_argument("--override", default=None, help="Experiment 覆盖配置文件路径")
    parser.add_argument("--print", action="store_true", help="打印合并后的配置")
    parser.add_argument("--freeze", default=None, help="保存合并后的配置到指定目录")
    args = parser.parse_args()

    config = build_config(args.base, args.override)

    if args.print:
        print(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))

    if args.freeze:
        frozen_path = freeze_config(config, args.freeze)
        print(f"Config frozen to: {frozen_path}")


if __name__ == "__main__":
    main()
