#!/usr/bin/env python3
"""
M1 SFT training script (MLX / mlx-lm).

Usage:
    uv run python scripts/train_sft.py
    uv run python scripts/train_sft.py --max-steps 10 --debug-dump-batch

Exits 0 on success, 1 on handled failure (run marked failed via registry).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from polaris.config import build_config, freeze_config
from polaris.monitoring.hardware import snapshot_hardware, append_hardware_log
from polaris.monitoring.metrics import append_metric, append_sample_diff
from polaris.registry import create_run, update_run_status

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem and put the final answer in \\boxed{}."
)

SMOKE_PROMPT = "What is 2 + 3?"

EVAL_MAX_NEW_TOKENS = 128
EVAL_TEMPERATURE = 0.0
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polaris M1 SFT Trainer")
    p.add_argument("--config", type=Path,
                   default=Path("configs/qwen3_0_6b/sft_math.yaml"),
                   help="Experiment config YAML (merged with configs/base.yaml)")
    p.add_argument("--max-steps", type=int, default=10,
                   help="Override max training steps")
    p.add_argument("--base-config", type=Path,
                   default=Path("configs/base.yaml"),
                   help="Base config YAML")
    p.add_argument("--runs-dir", type=str, default="runs",
                   help="Runs root directory")
    p.add_argument("--debug-dump-batch", action="store_true",
                   help="Dump first training batch to debug file")
    p.add_argument("--debug-dump-path", type=Path, default=None,
                   help="Debug dump output path (default: runs/{run}/logs/debug_batch.json)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_sft_dataset(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _strip_leading_think(token_ids: list[int]) -> list[int]:
    """Strip leading <think>\\n from token sequence (avoid duplication with prompt)."""
    THINK_OPEN = 151667
    NEWLINE = 198
    if len(token_ids) >= 2 and token_ids[0] == THINK_OPEN and token_ids[1] == NEWLINE:
        return token_ids[2:]
    return token_ids


def tokenize_sample(
    tokenizer,
    messages: list[dict],
    target: str,
    max_seq_length: int,
) -> dict[str, list[int]] | None:
    """Tokenize one sample. Returns None if the result is empty.

    Uses enable_thinking=True so prompt ends with <think> (no empty <think></think>).
    Strips leading <think>\\n from target to avoid duplication.
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True,
    )
    target_ids = tokenizer.encode(target)
    target_ids = _strip_leading_think(target_ids)
    eos_id = tokenizer.eos_token_id

    full_ids = prompt_ids + target_ids + ([eos_id] if eos_id is not None else [])
    prompt_len = len(prompt_ids)
    target_len = len(target_ids) + (1 if eos_id is not None else 0)

    if len(full_ids) > max_seq_length:
        keep = max_seq_length
        full_ids = full_ids[:keep]
        if prompt_len > keep:
            prompt_len = keep
            target_len = 0
        else:
            target_len = keep - prompt_len

    if not full_ids:
        return None

    return {
        "input_ids": full_ids,
        "prompt_len": prompt_len,
        "target_len": target_len,
    }


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------

def build_debug_dump(
    tokenizer,
    messages: list[dict],
    target: str,
    prompt_ids: list[int],
    full_ids: list[int],
    prompt_len: int,
    target_len: int,
    problem_id: str,
) -> dict:
    """Build a debug dump for one sample with full token-level visibility."""
    prompt_text = tokenizer.decode(prompt_ids)
    full_text = tokenizer.decode(full_ids)

    labels_preview = []
    shift_check = []
    prompt_label_count = 0
    target_label_count = 0
    ignored_count = 0
    supervised_count = 0

    for i, tid in enumerate(full_ids):
        tok_text = tokenizer.decode([tid])
        is_prompt = i < prompt_len
        label = -100 if is_prompt else tid
        loss_enabled = not is_prompt
        labels_preview.append({
            "token_id": tid,
            "token_text": tok_text,
            "label": label,
            "loss_enabled": loss_enabled,
        })
        if is_prompt:
            prompt_label_count += 1
            ignored_count += 1
        else:
            target_label_count += 1
            supervised_count += 1

    for i in range(max(0, len(full_ids) - 1)):
        input_id = full_ids[i]
        label_pos = i + 1
        label_id = full_ids[label_pos]
        loss_enabled = label_pos >= prompt_len
        shift_check.append({
            "input_position": i,
            "label_position": label_pos,
            "input_token_id": input_id,
            "label_token_id": label_id if loss_enabled else -100,
            "input_token_text": tokenizer.decode([input_id]),
            "label_token_text": tokenizer.decode([label_id]) if loss_enabled else "",
            "loss_enabled": loss_enabled,
        })

    return {
        "problem_id": problem_id,
        "messages": messages,
        "target": target,
        "prompt_text": prompt_text,
        "full_text": full_text,
        "prompt_token_count": prompt_len,
        "target_token_count": target_len,
        "full_token_count": len(full_ids),
        "labels_preview": labels_preview,
        "shift_check": shift_check[:max(80, min(len(shift_check), 160))],
        "shift_check_summary": {
            "checked_pairs": len(shift_check),
            "shown_pairs": min(len(shift_check), max(80, min(len(shift_check), 160))),
            "uses_next_token_labels": True,
            "self_token_prediction": False,
        },
        "loss_mask_summary": {
            "prompt_label_count": prompt_label_count,
            "target_label_count": target_label_count,
            "ignored_label_count": ignored_count,
            "supervised_label_count": supervised_count,
        },
    }


def validate_debug_dump(dump: dict) -> list[str]:
    """Run self-checks on the debug dump. Returns list of warnings."""
    warnings = []
    prompt_text = dump["prompt_text"]
    target = dump["target"]

    if "<|im_start|>assistant" not in prompt_text:
        warnings.append("prompt_text missing <|im_start|>assistant")
    # Only warn if \boxed{} appears with content (answer leaking), not just the instruction
    if re.search(r"\\boxed\{[^}]+\}", prompt_text):
        warnings.append("prompt_text contains \\boxed{answer} — answer may be leaking into prompt")
    if "<think>" not in target:
        warnings.append("target missing <think>")
    if "</think>" not in target:
        warnings.append("target missing </think>")
    if "\\boxed{" not in target:
        warnings.append("target missing \\boxed{}")

    summary = dump["loss_mask_summary"]
    if summary["supervised_label_count"] == 0:
        warnings.append("CRITICAL: supervised_label_count = 0 — target tokens not participating in loss!")

    shift_summary = dump.get("shift_check_summary", {})
    if not shift_summary.get("uses_next_token_labels"):
        warnings.append("CRITICAL: label shift is not next-token")
    if shift_summary.get("self_token_prediction"):
        warnings.append("CRITICAL: self-token prediction detected")
    if len(dump.get("shift_check", [])) < 80 and dump.get("full_token_count", 0) >= 81:
        warnings.append("shift_check shorter than 80 rows")

    return warnings


# ---------------------------------------------------------------------------
# MLX helpers
# ---------------------------------------------------------------------------

def collate_batch(
    batch: list[dict[str, list[int]]],
    pad_id: int = 0,
    label_pad_id: int = -100,
) -> dict:
    """Pad and collate a batch into MLX arrays."""
    import mlx.core as mx

    max_len = max(len(b["input_ids"]) for b in batch)
    bs = len(batch)

    input_ids = mx.full((bs, max_len), pad_id, dtype=mx.int32)
    labels = mx.full((bs, max_len), label_pad_id, dtype=mx.int32)
    lengths = mx.array([len(b["input_ids"]) for b in batch], dtype=mx.int32)

    for i, b in enumerate(batch):
        ids = b["input_ids"]
        n = len(ids)
        input_ids[i, :n] = mx.array(ids, dtype=mx.int32)

        prompt_len = b["prompt_len"]
        tgt_len = b["target_len"]
        if tgt_len > 0:
            target_ids = ids[prompt_len:prompt_len + tgt_len]
            labels[i, prompt_len:prompt_len + tgt_len] = mx.array(target_ids, dtype=mx.int32)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "lengths": lengths,
    }


def compute_loss(model, input_ids, labels, lengths):
    """Cross-entropy loss over target tokens only (labels != -100)."""
    import mlx.core as mx
    import mlx.nn.losses as losses

    logits = model(input_ids)
    logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_labels = shifted_labels.reshape(B * T)
    ce = losses.cross_entropy(flat_logits, flat_labels)
    ce = ce.reshape(B, T)

    mask = (shifted_labels != -100).astype(mx.float32)
    token_count = mask.sum()
    loss = (ce * mask).sum() / mx.maximum(token_count, mx.array(1.0))
    return loss


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def _count_params(params) -> int:
    from mlx.utils import tree_flatten

    return int(sum(v.size for _, v in tree_flatten(params)))


def _target_modules_from_config(lora_cfg: dict) -> list[str]:
    target_modules = lora_cfg.get("target_modules")
    if not target_modules:
        return DEFAULT_LORA_TARGET_MODULES
    if isinstance(target_modules, str):
        return [m.strip() for m in target_modules.split(",") if m.strip()]
    return list(target_modules)


def _collect_lora_target_keys(model, target_modules: list[str]) -> tuple[set[str], list[str]]:
    target_set = set(target_modules)
    keys: set[str] = set()
    hits: list[str] = []

    for layer_idx, layer in enumerate(model.layers):
        for name, _module in layer.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in target_set:
                keys.add(name)
                hits.append(f"layers.{layer_idx}.{name}")

    return keys, hits


def apply_lora(model, lora_cfg: dict, verbose: bool = True) -> tuple[Any, dict]:
    """Apply LoRA adapters to model."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    num_layers = lora_cfg.get("num_layers", len(list(model.layers)))
    target_modules = _target_modules_from_config(lora_cfg)
    target_keys, target_hits = _collect_lora_target_keys(model, target_modules)
    if not target_keys:
        raise RuntimeError(f"No LoRA target modules matched: {target_modules}")

    lora_config = {
        "rank": lora_cfg.get("r", 32),
        "scale": lora_cfg.get("alpha", 32) / lora_cfg.get("r", 32),
        "dropout": lora_cfg.get("dropout", 0.0),
        "keys": target_keys,
    }

    model.freeze()
    linear_to_lora_layers(model, num_layers=num_layers, config=lora_config)

    total = _count_params(model.parameters())
    trainable = _count_params(model.trainable_parameters())
    lora_info = {
        "target_modules": target_modules,
        "matched_module_count": len(target_hits),
        "matched_modules": target_hits,
        "num_layers": num_layers,
        "trainable_param_count": trainable,
        "total_param_count": total,
    }

    if verbose:
        print(f"  LoRA target modules requested: {target_modules}")
        print(f"  LoRA matched modules ({len(target_hits)}):")
        for name in target_hits:
            print(f"    - {name}")
        print(f"  LoRA applied to {num_layers} layers")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")

    return model, lora_info


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(model, tokenizer, prompt_ids: list[int],
                 max_tokens: int = 128, temp: float = 0.0) -> str:
    """Simple greedy generation."""
    import mlx.core as mx

    tokens = list(prompt_ids)
    for _ in range(max_tokens):
        x = mx.array([tokens])
        logits = model(x)
        next_logits = logits[0, -1, :]

        if temp <= 0:
            next_id = int(mx.argmax(next_logits).item())
        else:
            scaled = next_logits / temp
            probs = mx.softmax(scaled, axis=-1)
            next_id = int(mx.argmax(probs).item())

        if next_id == tokenizer.eos_token_id:
            break
        tokens.append(next_id)

    return tokenizer.decode(tokens[len(prompt_ids):])


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render messages to prompt string for preview."""
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True,
    )
    return tokenizer.decode(prompt_ids)


# ---------------------------------------------------------------------------
# Eval using eval_math.py protocol
# ---------------------------------------------------------------------------

def _import_eval_math():
    """Import eval_math functions. Returns module or None."""
    try:
        import importlib
        spec = importlib.util.spec_from_file_location(
            "eval_math", str(Path(__file__).parent / "eval_math.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def run_eval_protocol(
    model,
    tokenizer,
    eval_samples: list[dict],
    raw_data: list[dict],
    run_dir: Path,
    eval_split: str = "val",
) -> tuple[dict, list[str]]:
    """
    Run eval using eval_math.py protocol.
    Returns (eval_summary, warnings).
    """
    eval_mod = _import_eval_math()
    warnings: list[str] = []

    # Generate predictions
    predictions_raw = []
    for item in eval_samples:
        pid = item["metadata"].get("problem_id", "")
        prompt_ids = item["input_ids"][:item["prompt_len"]]

        try:
            gen_text = generate_one(
                model, tokenizer, prompt_ids,
                max_tokens=EVAL_MAX_NEW_TOKENS, temp=EVAL_TEMPERATURE,
            )
        except Exception:
            gen_text = ""

        gen_stripped = gen_text.strip()
        if not gen_stripped:
            warnings.append(f"empty_generation:{pid}")

        ref_answer = item["metadata"].get("answer", "")
        predictions_raw.append({
            "problem_id": pid,
            "prediction": gen_text,
            "reference_answer": ref_answer,
            "split": item["metadata"].get("split", eval_split),
            "model": "qwen3_0_6b_sft_sanity",
            "backend": "mlx",
        })

    # Write eval_predictions.jsonl
    pred_path = run_dir / "metrics" / "eval_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in predictions_raw:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Compute summary using eval_math.py functions
    n = len(predictions_raw)
    pass_count = 0
    extraction_ok = 0
    format_ok = 0
    total_len = 0
    empty_count = 0

    eval_results = []
    for p in predictions_raw:
        pred_text = p["prediction"]
        ref_answer = p["reference_answer"]

        total_len += len(pred_text)
        if not pred_text.strip():
            empty_count += 1

        if eval_mod:
            predicted_answer, method = eval_mod.extract_predicted_answer(pred_text)
            extraction_success = 1 if predicted_answer is not None else 0
            is_correct = eval_mod.answers_match(predicted_answer, ref_answer)
            has_think = eval_mod.has_think_block(pred_text)
            post_think = eval_mod.get_post_think_text(pred_text)
            has_final = bool(
                eval_mod.extract_boxed_answer(post_think)
                or eval_mod.extract_answer_tag(post_think)
                or eval_mod.extract_numeric_fallback(post_think)
            )
            fa = 1 if (has_think and has_final) else 0
        else:
            has_think = bool(re.search(r"<think>.*?</think>", pred_text, re.DOTALL))
            post_think = pred_text.split("</think>")[-1] if "</think>" in pred_text else pred_text
            m = re.search(r"\\boxed\{(.*?)\}", post_think)
            predicted_answer = m.group(1) if m else None
            method = "boxed" if m else "none"
            extraction_success = 1 if predicted_answer is not None else 0
            is_correct = (predicted_answer is not None and
                          predicted_answer.strip().lower() == ref_answer.strip().lower())
            fa = 1 if has_think else 0

        if extraction_success:
            extraction_ok += 1
        if fa:
            format_ok += 1
        if extraction_success and is_correct:
            pass_count += 1

        eval_results.append({
            "problem_id": p["problem_id"],
            "reference_answer": ref_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "extraction_method": method,
            "extraction_success": extraction_success,
            "format_adherence": fa,
            "pass": 1 if (extraction_success and is_correct) else 0,
            "completion_length": len(pred_text),
        })

    # Write eval_results.jsonl
    results_path = run_dir / "metrics" / "eval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Build summary
    summary = {
        "num_samples": n,
        "pass_at_1": round(pass_count / n, 4) if n else 0,
        "answer_extraction_success": round(extraction_ok / n, 4) if n else 0,
        "format_adherence": round(format_ok / n, 4) if n else 0,
        "invalid_output_rate": round(empty_count / n, 4) if n else 0,
        "avg_completion_length": round(total_len / n, 1) if n else 0,
    }

    # Write eval_summary.json
    summary_path = run_dir / "metrics" / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Diagnose all-zero predictions
    if empty_count == n:
        warnings.append("CRITICAL: all eval predictions are empty — generation failure, not math ability")
    elif empty_count > n * 0.5:
        warnings.append(f"WARNING: {empty_count}/{n} eval predictions are empty")

    return summary, warnings


# ---------------------------------------------------------------------------
# Sample diff
# ---------------------------------------------------------------------------

def run_sample_diff(
    model,
    tokenizer,
    eval_samples: list[dict],
    raw_data_map: dict[str, dict],
    run_dir: Path,
    n_samples: int = 5,
) -> list[str]:
    """Generate sample_diff.jsonl using eval_math.py protocol. Returns warnings."""
    eval_mod = _import_eval_math()
    warnings: list[str] = []

    samples_to_review = eval_samples[:n_samples]
    diffs = []

    for item in samples_to_review:
        pid = item["metadata"].get("problem_id", "")
        ref_answer = item["metadata"].get("answer", "")
        raw = item.get("_raw", raw_data_map.get(pid, {}))
        messages = raw.get("messages", [])
        problem = raw.get("problem", "")
        if not problem and messages:
            problem = next((m["content"] for m in messages if m["role"] == "user"), "")

        prompt_ids = item["input_ids"][:item["prompt_len"]]
        rendered_prompt = render_prompt(tokenizer, messages) if messages else ""

        gen_raw = generate_one(
            model, tokenizer, prompt_ids,
            max_tokens=EVAL_MAX_NEW_TOKENS, temp=EVAL_TEMPERATURE,
        )
        gen_stripped = gen_raw.strip()

        warning = None
        if not gen_stripped:
            warning = "empty_or_whitespace_generation"

        # Extract predicted answer
        if eval_mod:
            predicted_answer, _ = eval_mod.extract_predicted_answer(gen_raw)
            extraction_success = 1 if predicted_answer is not None else 0
            is_correct = eval_mod.answers_match(predicted_answer, ref_answer)
            has_think = eval_mod.has_think_block(gen_raw)
            post_think = eval_mod.get_post_think_text(gen_raw)
            has_final = bool(
                eval_mod.extract_boxed_answer(post_think)
                or eval_mod.extract_answer_tag(post_think)
                or eval_mod.extract_numeric_fallback(post_think)
            )
            format_adherence = 1 if (has_think and has_final) else 0
        else:
            has_think = bool(re.search(r"<think>.*?</think>", gen_raw, re.DOTALL))
            post_think = gen_raw.split("</think>")[-1] if "</think>" in gen_raw else gen_raw
            m = re.search(r"\\boxed\{(.*?)\}", post_think)
            predicted_answer = m.group(1) if m else None
            extraction_success = 1 if predicted_answer is not None else 0
            is_correct = (predicted_answer is not None and
                          predicted_answer.strip().lower() == ref_answer.strip().lower())
            format_adherence = 1 if has_think else 0

        pass_ = 1 if (extraction_success and is_correct) else 0

        diffs.append({
            "sample_id": pid,
            "problem": problem[:500],
            "reference_answer": ref_answer,
            "rendered_prompt_preview": rendered_prompt[:500],
            "before": "(pre-training baseline)",
            "after_raw": repr(gen_raw[:500]),
            "after_stripped": gen_stripped[:500] if gen_stripped else "",
            "predicted_answer": predicted_answer,
            "extraction_success": extraction_success,
            "format_adherence": format_adherence,
            "pass": pass_,
            "warning": warning,
        })

        if warning:
            warnings.append(f"{warning}:{pid}")

    # Write sample_diff.jsonl
    diff_path = run_dir / "samples" / "sample_diff.jsonl"
    with diff_path.open("w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return warnings


# ---------------------------------------------------------------------------
# Generation smoke test
# ---------------------------------------------------------------------------

def run_generation_smoke(model, tokenizer) -> dict:
    """Generate from a fixed prompt before/after training."""
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SMOKE_PROMPT},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True,
    )

    gen = generate_one(model, tokenizer, prompt_ids,
                       max_tokens=EVAL_MAX_NEW_TOKENS, temp=EVAL_TEMPERATURE)
    gen_stripped = gen.strip()

    warning = None
    if not gen_stripped:
        warning = "empty_or_whitespace_generation"

    return {
        "raw": gen,
        "stripped": gen_stripped,
        "warning": warning,
    }


def write_generation_diagnostics(run_dir: Path, diagnostics: dict) -> None:
    diag_path = run_dir / "logs" / "generation_diagnostics.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)


def _is_whitespace_output(output: dict | None) -> bool | None:
    if output is None:
        return None
    raw = output.get("raw", "")
    return bool(raw) and not raw.strip()


def _is_normal_generation(output: dict | None) -> bool | None:
    if output is None:
        return None
    return bool(output.get("stripped"))


def _loss_diagnosis(step_losses: list[float]) -> tuple[bool, str]:
    if not step_losses:
        return False, "no training steps"
    if any(math.isnan(x) or math.isinf(x) for x in step_losses):
        return True, "NaN or Inf detected"
    if max(step_losses) > 100:
        return True, f"loss too high: max={max(step_losses):.6f}"
    if len(step_losses) >= 2 and step_losses[-1] > max(step_losses[0] * 10, step_losses[0] + 10):
        return True, f"loss spike: first={step_losses[0]:.6f}, last={step_losses[-1]:.6f}"
    return False, "normal range"


def _grad_norm(grads) -> float | None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = tree_flatten(grads)
    if not flat:
        return None
    total = mx.array(0.0, dtype=mx.float32)
    for _, grad in flat:
        g = grad.astype(mx.float32)
        total = total + mx.sum(g * g)
    return float(mx.sqrt(total).item())


def write_diagnosis_report(
    run_dir: Path,
    warnings: list[str],
    generation_diagnostics: dict,
    step_losses: list[float],
    lora_info: dict,
    debug_dump: dict | None,
    max_steps: int,
) -> None:
    """Write logs/diagnosis.md with concrete answers for this run."""
    diag_path = run_dir / "logs" / "diagnosis.md"
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    base_ok = _is_normal_generation(generation_diagnostics.get("base_model_output"))
    before_ok = _is_normal_generation(generation_diagnostics.get("before_training_output"))
    after_ok = _is_normal_generation(generation_diagnostics.get("after_training_in_memory_output"))
    reload_ok = _is_normal_generation(generation_diagnostics.get("after_reloading_checkpoint_output"))
    loss_bad, loss_note = _loss_diagnosis(step_losses)

    shift_summary = (debug_dump or {}).get("shift_check_summary", {})
    label_shift_ok = (
        bool(shift_summary.get("uses_next_token_labels"))
        and not bool(shift_summary.get("self_token_prediction"))
    )
    target_modules = lora_info.get("target_modules", [])
    matched_modules = lora_info.get("matched_modules", [])
    lora_reasonable = (
        target_modules == DEFAULT_LORA_TARGET_MODULES
        and bool(matched_modules)
        and all(any(f".{target}." in f".{name}." or name.endswith(f".{target}")
                    for target in DEFAULT_LORA_TARGET_MODULES)
                for name in matched_modules)
    )

    def yes_no(value: bool | None) -> str:
        if value is None:
            return "未验证"
        return "是" if value else "否"

    with diag_path.open("w", encoding="utf-8") as f:
        f.write("# SFT Smoke Diagnosis\n\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"max_steps: {max_steps}\n\n")
        f.write("## Answers\n\n")
        f.write(f"- base model 是否正常：{yes_no(base_ok)}\n")
        f.write(f"- LoRA 注入但不训练是否正常：{yes_no(before_ok)}\n")
        f.write(f"- 1 step 后是否退化：{yes_no(not after_ok) if max_steps == 1 else '不适用'}\n")
        f.write(f"- 10 step 后是否退化：{yes_no(not after_ok) if max_steps == 10 else '不适用'}\n")
        f.write(f"- checkpoint reload 是否引入问题：{yes_no(bool(after_ok and not reload_ok))}\n")
        f.write(f"- loss 是否出现 NaN/Inf/异常飙升：{yes_no(loss_bad)}，{loss_note}\n")
        f.write(f"- label shift 是否正确：{yes_no(label_shift_ok)}\n")
        f.write(f"- LoRA target modules 是否合理：{yes_no(lora_reasonable)}，"
                f"matched={lora_info.get('matched_module_count', 0)}，"
                f"trainable_param_count={lora_info.get('trainable_param_count', 0)}\n\n")
        f.write("## Generation\n\n")
        for key in [
            "base_model_output",
            "before_training_output",
            "after_training_in_memory_output",
            "after_reloading_checkpoint_output",
        ]:
            out = generation_diagnostics.get(key) or {}
            f.write(f"- {key}: warning={out.get('warning')}, "
                    f"whitespace_only={_is_whitespace_output(out)}, "
                    f"stripped_preview={out.get('stripped', '')[:120]!r}\n")

        if warnings:
            f.write("\n## Warnings\n\n")
            for w in warnings:
                f.write(f"- {w}\n")


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def write_diagnosis(run_dir: Path, warnings: list[str]) -> None:
    """Write logs/diagnosis.md with accumulated warnings."""
    if not warnings:
        return
    diag_path = run_dir / "logs" / "diagnosis.md"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with diag_path.open("w", encoding="utf-8") as f:
        f.write("# Diagnosis\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Warnings\n\n")
        for w in warnings:
            f.write(f"- {w}\n")
        f.write("\n## Interpretation\n\n")
        has_empty = any("empty" in w.lower() for w in warnings)
        has_loss_zero = any("supervised_label_count" in w for w in warnings)
        if has_loss_zero:
            f.write("- Target tokens not participating in loss — check tokenize_sample and loss mask.\n")
        if has_empty:
            f.write("- Generation output is empty/whitespace — likely a generation or adapter issue, "
                     "not a math ability problem.\n")
            f.write("- Check: was the model loaded correctly? Was LoRA applied? "
                     "Is the prompt format compatible with the model?\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    config = build_config(str(args.base_config), str(args.config))

    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    max_steps = config.get("training", {}).get("max_steps")
    if max_steps is None:
        max_steps = config["training"]["num_epochs"] * 1000

    # --- Create run via registry ---
    merged_tmp = Path("__tmp_merged_config.yaml")
    with merged_tmp.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    try:
        run_dir = create_run(
            base_path=str(merged_tmp),
            override_path=None,
            runs_dir=args.runs_dir,
        )
    finally:
        merged_tmp.unlink(missing_ok=True)
        tmp_freeze = Path("__tmp_config_freeze")
        if tmp_freeze.exists():
            import shutil
            shutil.rmtree(tmp_freeze)

    run_dir = Path(run_dir)
    run_id = run_dir.name
    print(f"Run created: {run_id} -> {run_dir}")

    # Resolve debug dump path
    debug_dump_path = args.debug_dump_path
    if debug_dump_path is None:
        debug_dump_path = run_dir / "logs" / "debug_batch.json"

    # --- Hardware snapshot ---
    try:
        snap = snapshot_hardware()
        append_hardware_log(run_dir, snap)
        print(f"  Hardware: {snap.get('platform')} / {snap.get('machine')}")
        print(f"  MLX available: {snap.get('mlx_available')}")
    except Exception as e:
        print(f"  [warn] hardware snapshot failed: {e}")

    # --- Training ---
    try:
        exit_status = _run_training(
            config, run_dir, max_steps, args.runs_dir,
            debug_dump=args.debug_dump_batch,
            debug_dump_path=debug_dump_path,
        )
    except Exception as exc:
        _handle_failure(run_dir, exc, args.runs_dir)
        return 1

    update_run_status(run_id, exit_status, args.runs_dir)
    print(f"\nRun {run_id} finished with status: {exit_status}")
    return 0


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------

def _run_training(
    config: dict,
    run_dir: Path,
    max_steps: int,
    runs_dir: str,
    debug_dump: bool = False,
    debug_dump_path: Path | None = None,
) -> str:
    """Core training logic. Returns status string: 'completed' or 'completed_with_warnings'."""
    run_id = run_dir.name
    all_warnings: list[str] = []

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load as mlx_load
        from mlx.utils import tree_map
    except ImportError as e:
        raise RuntimeError(
            f"mlx / mlx_lm not installed: {e}\n"
            "Install with: pip install mlx mlx-lm"
        ) from e

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    model_path = model_cfg.get("path", "Qwen/Qwen3-0.6B")
    data_path = Path(data_cfg.get("path", "data/math/splits/sft_v1.jsonl"))
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    batch_size = train_cfg.get("batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    lr = train_cfg.get("learning_rate", 5e-5)
    seed = train_cfg.get("seed", 42)

    mx.random.seed(seed)

    # ------ Load model ------
    print(f"Loading model: {model_path} ...")
    t0 = time.time()
    model, tokenizer = mlx_load(model_path)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    generation_diagnostics: dict[str, Any] = {
        "prompt": SMOKE_PROMPT,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "temperature": EVAL_TEMPERATURE,
    }

    # ------ Base model smoke ------
    print(f"\n[Smoke] Generating base model output: \"{SMOKE_PROMPT}\"")
    base_output = run_generation_smoke(model, tokenizer)
    generation_diagnostics["base_model_output"] = base_output
    print(f"  base_stripped: {repr(base_output['stripped'][:200])}")
    if base_output["warning"]:
        print(f"  WARNING: {base_output['warning']}")
        all_warnings.append(f"base_model_output:{base_output['warning']}")

    # ------ LoRA ------
    lora_info: dict[str, Any] = {
        "enabled": False,
        "target_modules": [],
        "matched_module_count": 0,
        "matched_modules": [],
        "trainable_param_count": _count_params(model.trainable_parameters()),
        "total_param_count": _count_params(model.parameters()),
    }
    if lora_cfg.get("enabled", True):
        model, lora_info = apply_lora(model, lora_cfg)
        lora_info["enabled"] = True
    generation_diagnostics["lora"] = lora_info

    # ------ LoRA-injected smoke ------
    print(f"\n[Smoke] Generating before training with LoRA: \"{SMOKE_PROMPT}\"")
    smoke_before = run_generation_smoke(model, tokenizer)
    generation_diagnostics["before_training_output"] = smoke_before
    print(f"  before_training_stripped: {repr(smoke_before['stripped'][:200])}")
    if smoke_before["warning"]:
        print(f"  WARNING: {smoke_before['warning']}")
        all_warnings.append(f"before_training_output:{smoke_before['warning']}")

    # ------ Load dataset ------
    print(f"\nLoading dataset: {data_path} ...")
    raw_data = load_sft_dataset(data_path)
    raw_data_map = {item.get("metadata", {}).get("problem_id", ""): item for item in raw_data}
    n_samples = len(raw_data)
    print(f"  {n_samples} samples loaded")

    # ------ Tokenize ------
    print("Tokenizing ...")
    tokenized: list[dict] = []
    skipped = 0
    for item in raw_data:
        tk = tokenize_sample(tokenizer, item["messages"], item["target"], max_seq_length)
        if tk is not None:
            tk["metadata"] = item.get("metadata", {})
            tk["_raw"] = item
            tokenized.append(tk)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} empty samples")
    print(f"  {len(tokenized)} samples tokenized")

    # ------ Debug dump ------
    current_debug_dump: dict | None = None
    if debug_dump and tokenized:
        print(f"\n[Debug] Writing batch debug dump to {debug_dump_path} ...")
        first = tokenized[0]
        dump = build_debug_dump(
            tokenizer,
            first["_raw"]["messages"],
            first["_raw"]["target"],
            first["input_ids"][:first["prompt_len"]],
            first["input_ids"],
            first["prompt_len"],
            first["target_len"],
            first["metadata"].get("problem_id", ""),
        )
        dump_warnings = validate_debug_dump(dump)
        dump["_validation_warnings"] = dump_warnings
        current_debug_dump = dump

        debug_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_dump_path.open("w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False)

        for w in dump_warnings:
            print(f"  [debug] {w}")
            all_warnings.append(f"debug_dump:{w}")

    # ------ Optimizer ------
    optimizer = optim.Adam(learning_rate=lr)

    # ------ Train loop ------
    effective_steps = min(max_steps, len(tokenized) // batch_size) if len(tokenized) >= batch_size else max_steps
    print(f"\nStarting training: {effective_steps} steps (batch_size={batch_size}, "
          f"grad_accum={grad_accum}, lr={lr})")
    print("=" * 60)

    update_run_status(run_id, "running", runs_dir)
    step_losses: list[float] = []
    data_idx = 0
    start_time = time.time()

    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    model.train()

    for step in range(1, effective_steps + 1):
        micro_losses: list[float] = []
        accumulated_grads = None
        for _ in range(grad_accum):
            if data_idx + batch_size > len(tokenized):
                data_idx = 0
            micro_batch = tokenized[data_idx:data_idx + batch_size]
            data_idx += batch_size

            collated = collate_batch(micro_batch)
            loss, grads = loss_and_grad_fn(
                model,
                collated["input_ids"],
                collated["labels"],
                collated["lengths"],
            )
            micro_losses.append(float(loss.item()))
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda x, y: x + y, accumulated_grads, grads)

        avg_loss = sum(micro_losses) / len(micro_losses)
        if grad_accum > 1:
            accumulated_grads = tree_map(lambda x: x / grad_accum, accumulated_grads)
        grad_norm = _grad_norm(accumulated_grads)
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)

        step_losses.append(avg_loss)
        append_metric(run_dir, "train", step=step,
                      train_loss=round(avg_loss, 6),
                      grad_norm=round(grad_norm, 6) if grad_norm is not None else None,
                      learning_rate=lr,
                      trainable_param_count=lora_info.get("trainable_param_count"),
                      tokens_processed=step * batch_size * grad_accum * max_seq_length)

        elapsed = time.time() - start_time
        grad_norm_text = f"{grad_norm:.4f}" if grad_norm is not None else "NA"
        print(f"  step {step}/{effective_steps}  loss={avg_loss:.4f}  "
              f"grad_norm={grad_norm_text}  elapsed={elapsed:.1f}s")

    train_time = time.time() - start_time
    print(f"\nTraining finished in {train_time:.1f}s")

    # ------ Save checkpoint ------
    ckpt_dir = run_dir / "checkpoints" / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(ckpt_dir / "model.safetensors"))
    ckpt_size = (ckpt_dir / "model.safetensors").stat().st_size
    print(f"Checkpoint saved: {ckpt_dir} ({ckpt_size:,} bytes)")
    if ckpt_size == 0:
        all_warnings.append("CRITICAL: checkpoint file is empty")

    # ------ Post-training smoke ------
    print(f"\n[Smoke] Generating after training: \"{SMOKE_PROMPT}\"")
    smoke_after = run_generation_smoke(model, tokenizer)
    generation_diagnostics["after_training_in_memory_output"] = smoke_after
    print(f"  after_stripped: {repr(smoke_after['stripped'][:200])}")
    if smoke_after["warning"]:
        print(f"  WARNING: {smoke_after['warning']}")
        all_warnings.append(f"smoke_after:{smoke_after['warning']}")

    # ------ Reload checkpoint smoke ------
    print(f"\n[Smoke] Reloading checkpoint and generating: \"{SMOKE_PROMPT}\"")
    reloaded_model, reloaded_tokenizer = mlx_load(model_path)
    if lora_cfg.get("enabled", True):
        reloaded_model, _ = apply_lora(reloaded_model, lora_cfg, verbose=False)
    reloaded_model.load_weights(str(ckpt_dir / "model.safetensors"), strict=False)
    mx.eval(reloaded_model.parameters())
    smoke_reloaded = run_generation_smoke(reloaded_model, reloaded_tokenizer)
    generation_diagnostics["after_reloading_checkpoint_output"] = smoke_reloaded
    print(f"  after_reload_stripped: {repr(smoke_reloaded['stripped'][:200])}")
    if smoke_reloaded["warning"]:
        print(f"  WARNING: {smoke_reloaded['warning']}")
        all_warnings.append(f"smoke_reload:{smoke_reloaded['warning']}")

    write_generation_diagnostics(run_dir, generation_diagnostics)

    # Write smoke before/after
    smoke_path = run_dir / "logs" / "generation_smoke_before_after.json"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    with smoke_path.open("w", encoding="utf-8") as f:
        json.dump({
            "prompt": SMOKE_PROMPT,
            "base_model": base_output["raw"],
            "before_training": smoke_before["raw"],
            "after_training_in_memory": smoke_after["raw"],
            "after_reloading_checkpoint": smoke_reloaded["raw"],
            "base_stripped": base_output["stripped"],
            "before_stripped": smoke_before["stripped"],
            "after_stripped": smoke_after["stripped"],
            "after_reload_stripped": smoke_reloaded["stripped"],
            "base_warning": base_output["warning"],
            "before_warning": smoke_before["warning"],
            "after_warning": smoke_after["warning"],
            "after_reload_warning": smoke_reloaded["warning"],
        }, f, indent=2, ensure_ascii=False)

    write_diagnosis_report(
        run_dir,
        all_warnings,
        generation_diagnostics,
        step_losses,
        lora_info,
        current_debug_dump,
        max_steps,
    )

    if effective_steps == 0:
        if all_warnings:
            print(f"\n{len(all_warnings)} warnings written to logs/diagnosis.md")
            return "completed_with_warnings"
        return "completed"

    # ------ Eval (using eval_math.py protocol) ------
    print("\nRunning eval ...")
    eval_samples = [t for t in tokenized if t.get("metadata", {}).get("split") in ("test", "val")]
    if not eval_samples:
        eval_samples = tokenized[:10]
    eval_samples = eval_samples[:10]  # max 10 for sanity

    eval_summary, eval_warnings = run_eval_protocol(
        model, tokenizer, eval_samples, raw_data, run_dir,
    )
    all_warnings.extend(eval_warnings)

    # Write train summary to eval_metrics.jsonl as well
    eval_summary["train_time_seconds"] = round(train_time, 2)
    eval_summary["train_loss_mean"] = round(sum(step_losses) / len(step_losses), 6) if step_losses else 0
    eval_summary["train_loss_first"] = step_losses[0] if step_losses else 0
    eval_summary["train_loss_last"] = step_losses[-1] if step_losses else 0
    append_metric(run_dir, "eval", step=0, **eval_summary)

    print(f"  pass_at_1: {eval_summary.get('pass_at_1', 'N/A')}")
    print(f"  answer_extraction_success: {eval_summary.get('answer_extraction_success', 'N/A')}")
    print(f"  format_adherence: {eval_summary.get('format_adherence', 'N/A')}")
    print(f"  invalid_output_rate: {eval_summary.get('invalid_output_rate', 'N/A')}")
    print(f"  avg_completion_length: {eval_summary.get('avg_completion_length', 'N/A')}")

    # ------ Sample diff ------
    print("\nGenerating sample diffs ...")
    diff_warnings = run_sample_diff(
        model, tokenizer, eval_samples, raw_data_map, run_dir, n_samples=5,
    )
    all_warnings.extend(diff_warnings)

    # ------ Diagnosis ------
    write_diagnosis_report(
        run_dir,
        all_warnings,
        generation_diagnostics,
        step_losses,
        lora_info,
        current_debug_dump,
        max_steps,
    )
    if all_warnings:
        print(f"\n{len(all_warnings)} warnings written to logs/diagnosis.md")

    # ------ Determine status ------
    critical = [w for w in all_warnings if "CRITICAL" in w]
    if critical:
        return "completed_with_warnings"
    if all_warnings:
        return "completed_with_warnings"
    return "completed"


# ---------------------------------------------------------------------------
# Failure handler
# ---------------------------------------------------------------------------

def _handle_failure(run_dir: Path, exc: Exception, runs_dir: str):
    """Mark run as failed, write error.txt."""
    run_id = run_dir.name
    print(f"\n{'='*60}")
    print(f"TRAINING FAILED: {exc}")
    print(f"{'='*60}")

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    error_path = logs_dir / "error.txt"
    with error_path.open("w", encoding="utf-8") as f:
        f.write(f"Error: {exc}\n\n")
        f.write(traceback.format_exc())

    try:
        update_run_status(run_id, "failed", runs_dir)
    except Exception as reg_err:
        print(f"  [warn] could not update run status: {reg_err}")

    print(f"Error written: {error_path}")
    print(f"Run {run_id} marked as failed.")


if __name__ == "__main__":
    raise SystemExit(main())
