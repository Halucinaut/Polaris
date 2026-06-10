#!/usr/bin/env python3
"""
D3 KR2: 模型加载 smoke test 脚本。

使用 MLX backend 加载本地 Qwen3-0.6B MLX bf16 模型，并生成一段短文本。
该脚本只用于 smoke test，不做 baseline eval，不创建 registry run，不训练。
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: load a local MLX model and generate text."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/qwen3_0_6b_mlx",
        help="Path to the local MLX model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is 2 + 3?",
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save the JSON summary.",
    )
    return parser.parse_args()


def validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")

    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for fname in required_files:
        if not (model_path / fname).is_file():
            raise FileNotFoundError(
                f"Missing required file in model path: {fname}"
            )

    safetensors_found = any(model_path.glob("*.safetensors"))
    if not safetensors_found:
        raise FileNotFoundError(
            "No safetensors weight files found in model path."
        )


def apply_chat_template_safe(tokenizer, messages: list[dict]) -> str:
    sig = inspect.signature(tokenizer.apply_chat_template)
    kwargs: dict = {"add_generation_prompt": True}

    if "enable_thinking" in sig.parameters:
        kwargs["enable_thinking"] = False
    else:
        try:
            tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
            kwargs["enable_thinking"] = False
        except TypeError:
            pass

    return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def run_smoke(args: argparse.Namespace) -> dict:
    model_path = Path(args.model_path)
    validate_model_path(model_path)

    total_tic = time.perf_counter()

    load_tic = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_time_sec = time.perf_counter() - load_tic

    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": args.prompt},
    ]

    rendered_prompt = apply_chat_template_safe(tokenizer, messages)

    gen_tic = time.perf_counter()
    sampler = make_sampler(args.temperature)
    output = generate(
        model,
        tokenizer,
        prompt=rendered_prompt,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
    )
    generation_time_sec = time.perf_counter() - gen_tic

    total_time_sec = time.perf_counter() - total_tic

    result = {
        "status": "success",
        "backend": "mlx",
        "model_path": str(model_path),
        "prompt": args.prompt,
        "rendered_prompt_preview": rendered_prompt[:500],
        "output": output,
        "load_time_sec": round(load_time_sec, 4),
        "generation_time_sec": round(generation_time_sec, 4),
        "total_time_sec": round(total_time_sec, 4),
    }
    return result


def print_summary(result: dict) -> None:
    print(f"status: {result['status']}")
    print(f"backend: {result['backend']}")
    print(f"model_path: {result['model_path']}")
    print(f"prompt: {result['prompt']}")
    if result["status"] == "success":
        print(f"rendered_prompt_preview: {result['rendered_prompt_preview']}")
        print(f"output: {result['output']}")
        print(f"load_time_sec: {result['load_time_sec']}")
        print(f"generation_time_sec: {result['generation_time_sec']}")
        print(f"total_time_sec: {result['total_time_sec']}")
    else:
        print(f"error_type: {result.get('error_type', 'Unknown')}")
        print(f"error_message: {result.get('error_message', '')}")


def main() -> int:
    args = parse_args()

    try:
        result = run_smoke(args)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        traceback_str = traceback.format_exc()

        result = {
            "status": "failed",
            "backend": "mlx",
            "model_path": args.model_path,
            "prompt": args.prompt,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback_str,
        }

        print_summary(result)

        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return 1

    print_summary(result)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Summary saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
