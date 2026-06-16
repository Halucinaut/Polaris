#!/usr/bin/env python3
"""
GSM8K smoke evaluation script.

加载本地 MLX Qwen3 模型，在 GSM8K test/review 数据集上生成答案，
使用 eval_math.py 的提取逻辑评估准确率。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 动态导入项目内的模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
scripts_dir = PROJECT_ROOT / "scripts"
smoke_dir = scripts_dir / "smoke"

eval_math = _load_module("eval_math", scripts_dir / "eval_math.py")
load_model = _load_module("load_model", smoke_dir / "load_model.py")

extract_predicted_answer = eval_math.extract_predicted_answer
answers_match = eval_math.answers_match
validate_model_path = load_model.validate_model_path
apply_chat_template_safe = load_model.apply_chat_template_safe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K smoke eval using local MLX model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/qwen3_0_6b/mlx",
        help="Path to the local MLX model directory.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (for SFT/DPO models).",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/math/gsm8k/split/test_converted.jsonl",
        help="Path to GSM8K test JSONL.",
    )
    parser.add_argument(
        "--review-data",
        type=str,
        default="data/math/gsm8k/split/review_converted.jsonl",
        help="Path to GSM8K review JSONL.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per problem.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per dataset (for quick smoke).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSON path to save detailed results.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, problem: str, max_tokens: int, temperature: float) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem and put the final answer in \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    rendered = apply_chat_template_safe(tokenizer, messages)
    sampler = make_sampler(temperature)
    return generate(
        model,
        tokenizer,
        prompt=rendered,
        max_tokens=max_tokens,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_dataset(
    model,
    tokenizer,
    records: list[dict],
    max_tokens: int,
    temperature: float,
    dataset_name: str,
) -> list[dict]:
    """Run inference only; return raw outputs for downstream eval_math.evaluate."""
    results: list[dict] = []
    total = len(records)

    for idx, rec in enumerate(records, start=1):
        problem_id = rec["problem_id"]
        problem = rec["problem"]
        expected = str(rec["answer"])

        print(f"[{dataset_name}] {idx}/{total} {problem_id} ...", flush=True)
        tic = time.perf_counter()

        try:
            raw_output = generate_answer(model, tokenizer, problem, max_tokens, temperature)
        except Exception as exc:
            print(f"  Generation failed: {exc}")
            results.append({
                "problem_id": problem_id,
                "problem": problem,
                "expected": expected,
                "raw_output": None,
                "generation_time_sec": round(time.perf_counter() - tic, 2),
            })
            continue

        gen_time = time.perf_counter() - tic
        results.append({
            "problem_id": problem_id,
            "problem": problem,
            "expected": expected,
            "raw_output": raw_output,
            "generation_time_sec": round(gen_time, 2),
        })

    return results


def run_eval_math(inference_results: list[dict]) -> tuple[list[dict], dict]:
    """Use eval_math.evaluate + compute_metrics on inference outputs."""
    predictions = [
        eval_math.PredictionRecord(
            problem_id=r["problem_id"],
            prediction=r["raw_output"] or "",
        )
        for r in inference_results
    ]
    references = {
        r["problem_id"]: eval_math.ReferenceRecord(
            problem_id=r["problem_id"],
            problem=r["problem"],
            answer=r["expected"],
            level=0,
            source="gsm8k",
            domain="grade_school_math",
        )
        for r in inference_results
    }

    eval_results = eval_math.evaluate(predictions, references)
    metrics = eval_math.compute_metrics(eval_results)

    # Convert EvalResult back to legacy dict format for backward compatibility
    er_map = {er.problem_id: er for er in eval_results}
    legacy_results: list[dict] = []
    for r in inference_results:
        er = er_map.get(r["problem_id"])
        if er is not None:
            legacy_results.append({
                "problem_id": r["problem_id"],
                "problem": r["problem"],
                "expected": r["expected"],
                "raw_output": r["raw_output"],
                "predicted_answer": er.predicted_answer,
                "is_correct": er.is_correct,
                "extraction_method": er.extraction_method,
                "extraction_success": er.extraction_success,
                "format_adherence": er.format_adherence,
                "pass": er.pass_,
                "generation_time_sec": r["generation_time_sec"],
            })
        else:
            legacy_results.append({
                "problem_id": r["problem_id"],
                "problem": r["problem"],
                "expected": r["expected"],
                "raw_output": r["raw_output"],
                "predicted_answer": None,
                "is_correct": False,
                "extraction_method": "generation_failed",
                "extraction_success": 0,
                "format_adherence": 0,
                "pass": 0,
                "generation_time_sec": r["generation_time_sec"],
            })

    return legacy_results, metrics


def print_summary(dataset_name: str, metrics: dict) -> None:
    total = metrics["total"]
    correct = metrics["correct"]
    accuracy = metrics["accuracy"]
    print(f"\n=== {dataset_name} ===")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"pass_at_1: {metrics['pass_at_1']:.2%}")
    print(f"answer_extraction_success: {metrics['answer_extraction_success']:.2%}")
    print(f"format_adherence: {metrics['format_adherence']:.2%}")
    print("Method breakdown:")
    for method, info in sorted(metrics["method_breakdown"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {method}: {info['count']} (correct: {info['correct']}, acc: {info['accuracy']:.2%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)

    try:
        validate_model_path(model_path)
    except Exception as exc:
        print(f"Model path validation failed: {exc}")
        return 1

    print("Loading model...")
    load_tic = time.perf_counter()
    adapter_path = args.adapter_path
    if adapter_path:
        print(f"  LoRA adapter: {adapter_path}")
    model, tokenizer = load(str(model_path), adapter_path=adapter_path)
    print(f"Model loaded in {time.perf_counter() - load_tic:.2f}s\n")

    all_results: dict[str, list[dict]] = {}

    for label, data_path_str in [("test", args.test_data), ("review", args.review_data)]:
        data_path = Path(data_path_str)
        if not data_path.exists():
            print(f"Warning: {data_path} not found, skipping.")
            continue

        records = load_jsonl(data_path)
        if args.limit is not None:
            records = records[:args.limit]

        inference_results = eval_dataset(
            model,
            tokenizer,
            records,
            args.max_new_tokens,
            args.temperature,
            label,
        )

        # Delegate evaluation to eval_math.py
        legacy_results, metrics = run_eval_math(inference_results)
        all_results[label] = legacy_results
        print_summary(label, metrics)

        # Optionally save predictions.jsonl alongside the main output
        if args.output_path:
            out_dir = Path(args.output_path).parent
            pred_path = out_dir / f"{label}_predictions.jsonl"
            with pred_path.open("w", encoding="utf-8") as f:
                for r in inference_results:
                    f.write(
                        json.dumps(
                            {"problem_id": r["problem_id"], "prediction": r["raw_output"] or ""},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"  Predictions saved to: {pred_path}")

    # Overall summary
    total_all = sum(len(v) for v in all_results.values())
    correct_all = sum(sum(1 for r in v if r["is_correct"]) for v in all_results.values())
    if total_all > 0:
        print(f"\n=== Overall ===")
        print(f"Total: {total_all}")
        print(f"Correct: {correct_all}")
        print(f"Accuracy: {correct_all / total_all:.2%}")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
