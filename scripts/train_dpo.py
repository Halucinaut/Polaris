#!/usr/bin/env python3
"""
M2 DPO training script (MLX / mlx-lm).

Implements DPO loss with response-only logprob, frozen reference model,
and proper next-token prediction alignment.

Usage:
    python scripts/train_dpo.py --max-steps 10 --debug-batch
    python scripts/train_dpo.py --config configs/qwen3_0_6b/dpo_math.yaml

Exits 0 on success, 1 on handled failure (run marked failed via registry).
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from polaris.config import build_config, freeze_config
from polaris.json_records import load_json_record_stream
from polaris.monitoring.hardware import snapshot_hardware, append_hardware_log
from polaris.monitoring.metrics import append_metric
from polaris.registry import create_run, update_run_status


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem and put the final answer in \\boxed{}."
)

ASSISTANT_HEADER = "<|im_start|>assistant\n"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polaris M2 DPO Trainer")
    p.add_argument("--config", type=Path,
                   default=Path("configs/qwen3_0_6b/dpo_math.yaml"),
                   help="Experiment config YAML")
    p.add_argument("--data-path", type=Path, default=None,
                   help="Override DPO data path (dpo_v1.jsonl)")
    p.add_argument("--policy-adapter-path", type=Path, default=None,
                   help="Override the initial policy LoRA adapter directory or safetensors file")
    p.add_argument("--ref-model-path", type=Path, default=None,
                   help="Override the frozen reference base-model directory")
    p.add_argument("--ref-adapter-path", type=Path, default=None,
                   help="Override the frozen reference LoRA adapter directory or safetensors file")
    p.add_argument("--policy-checkpoint", type=Path, default=None,
                   help="Deprecated alias for --policy-adapter-path")
    p.add_argument("--ref-checkpoint", type=Path, default=None,
                   help="Deprecated alias for --ref-adapter-path")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max training steps")
    p.add_argument("--output-run-dir", type=str, default=None,
                   help="Override run output directory name")
    p.add_argument("--beta", type=float, default=None,
                   help="Override DPO beta parameter")
    p.add_argument("--debug-batch", action="store_true",
                   help="Dump first training batch to debug_dpo_batch.json")
    p.add_argument("--base-config", type=Path,
                   default=Path("configs/base.yaml"),
                   help="Base config YAML")
    p.add_argument("--runs-dir", type=str, default="runs",
                   help="Runs root directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

def acquire_training_lock(runs_dir: str) -> Path:
    lock_path = Path(runs_dir) / ".train_dpo.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Another train_dpo.py run appears to be active: {lock_path} {existing}. "
            "Remove the lock only after confirming no DPO process is running."
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")
        f.write(f"created_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _release() -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_release)
    return lock_path


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dpo_dataset(path: Path) -> list[dict]:
    return load_json_record_stream(path)


def build_microbatch_groups(
    records: list[dict],
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    max_steps: int | None,
) -> list[list[list[dict]]]:
    """Create optimizer updates without silently multiplying epoch coverage.

    A group contains the micro-batches whose gradients form one optimizer
    update. The final group in an epoch may contain fewer micro-batches, so
    every record is consumed exactly once per epoch instead of wrapping to the
    start of the dataset.
    """
    if batch_size < 1 or gradient_accumulation_steps < 1 or num_epochs < 1:
        raise ValueError("batch_size, gradient_accumulation_steps, and num_epochs must be positive")
    if not records:
        raise ValueError("Cannot build DPO batches from an empty dataset")

    micro_batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    epoch_groups = [
        micro_batches[i:i + gradient_accumulation_steps]
        for i in range(0, len(micro_batches), gradient_accumulation_steps)
    ]
    all_groups = epoch_groups * num_epochs

    if max_steps is None:
        return all_groups
    if max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    return [all_groups[i % len(all_groups)] for i in range(max_steps)]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return rendered + ASSISTANT_HEADER


def tokenize_pair(
    tokenizer,
    messages: list[dict],
    chosen: str,
    rejected: str,
    max_seq_length: int,
    prompt_suffix: str = "",
    no_eos: bool = False,
) -> dict | None:
    """Tokenize one DPO pair. Returns None if result is empty.

    Args:
        prompt_suffix: Extra text appended to prompt before tokenization.
        no_eos: If True, do not append EOS after response (single-token supervision).
    """
    prompt_text = render_prompt(tokenizer, messages)
    prompt_ids = tokenizer.encode(prompt_text)
    if prompt_suffix:
        prompt_ids = prompt_ids + tokenizer.encode(prompt_suffix)
    eos_id = tokenizer.eos_token_id
    use_eos = (eos_id is not None) and (not no_eos)

    chosen_ids = tokenizer.encode(chosen)
    rejected_ids = tokenizer.encode(rejected)

    # Build full sequences: prompt + response + eos
    chosen_full = prompt_ids + chosen_ids + ([eos_id] if use_eos else [])
    rejected_full = prompt_ids + rejected_ids + ([eos_id] if use_eos else [])

    prompt_len = len(prompt_ids)

    # Truncate if needed
    if len(chosen_full) > max_seq_length:
        chosen_full = chosen_full[:max_seq_length]
    if len(rejected_full) > max_seq_length:
        rejected_full = rejected_full[:max_seq_length]

    # Response masks: 1 for response tokens, 0 for prompt tokens
    chosen_mask = [0] * min(prompt_len, len(chosen_full)) + [1] * max(0, len(chosen_full) - prompt_len)
    rejected_mask = [0] * min(prompt_len, len(rejected_full)) + [1] * max(0, len(rejected_full) - prompt_len)

    if not chosen_full or not rejected_full or not any(chosen_mask) or not any(rejected_mask):
        return None

    return {
        "chosen_ids": chosen_full,
        "rejected_ids": rejected_full,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
        "prompt_len": prompt_len,
        "chosen_len": len(chosen_ids),
        "rejected_len": len(rejected_ids),
    }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_dpo_batch(
    batch: list[dict],
    pad_id: int = 0,
) -> dict:
    import mlx.core as mx

    max_chosen = max(len(b["chosen_ids"]) for b in batch)
    max_rejected = max(len(b["rejected_ids"]) for b in batch)
    max_len = max(max_chosen, max_rejected)
    bs = len(batch)

    chosen_ids = mx.full((bs, max_len), pad_id, dtype=mx.int32)
    rejected_ids = mx.full((bs, max_len), pad_id, dtype=mx.int32)
    chosen_mask = mx.zeros((bs, max_len), dtype=mx.float32)
    rejected_mask = mx.zeros((bs, max_len), dtype=mx.float32)

    for i, b in enumerate(batch):
        nc = len(b["chosen_ids"])
        nr = len(b["rejected_ids"])
        chosen_ids[i, :nc] = mx.array(b["chosen_ids"], dtype=mx.int32)
        rejected_ids[i, :nr] = mx.array(b["rejected_ids"], dtype=mx.int32)
        chosen_mask[i, :nc] = mx.array(b["chosen_mask"], dtype=mx.float32)
        rejected_mask[i, :nr] = mx.array(b["rejected_mask"], dtype=mx.float32)

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
    }


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def compute_response_logprob(logits, ids, mask):
    """Compute per-sample response logprob with next-token alignment.

    logits: (B, T, V) - raw model output
    ids:    (B, T)    - token ids
    mask:   (B, T)    - 1 for response tokens, 0 for prompt tokens

    Uses next-token prediction: logits[:, :-1] aligns with ids[:, 1:].
    Mask is shifted accordingly: mask[:, 1:] selects response tokens.
    """
    import mlx.nn.losses as losses

    # Next-token alignment: predict token at position t+1 from logits at position t
    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_ids = ids[:, 1:]             # (B, T-1)
    shift_mask = mask[:, 1:]           # (B, T-1) — mask the TARGET position

    B, T, V = shift_logits.shape
    flat_logits = shift_logits.reshape(B * T, V)
    flat_ids = shift_ids.reshape(B * T)

    # Per-token cross-entropy (negative log-likelihood)
    token_nll = losses.cross_entropy(flat_logits, flat_ids)  # (B*T,)
    token_nll = token_nll.reshape(B, T)

    # DPO is defined over sequence log-probabilities. Length balance belongs
    # in the pair-construction protocol and in the recorded diagnostics, not
    # in an implicit change from sequence sums to token averages.
    masked_nll = token_nll * shift_mask
    sample_logprob = -masked_nll.sum(axis=1)  # negative because CE = -logprob

    return sample_logprob


def compute_dpo_loss(
    policy_model,
    ref_model,
    chosen_ids,
    rejected_ids,
    chosen_mask,
    rejected_mask,
    beta: float,
) -> dict:
    """Compute DPO loss and metrics.

    Returns dict with loss, logprobs, margins, and preference accuracy.
    """
    import mlx.core as mx

    # Forward pass for policy
    policy_chosen_logits = policy_model(chosen_ids)
    policy_rejected_logits = policy_model(rejected_ids)

    # Forward pass for reference (frozen, no grad)
    ref_chosen_logits = ref_model(chosen_ids)
    ref_rejected_logits = ref_model(rejected_ids)

    # Response logprobs
    policy_chosen_lp = compute_response_logprob(policy_chosen_logits, chosen_ids, chosen_mask)
    policy_rejected_lp = compute_response_logprob(policy_rejected_logits, rejected_ids, rejected_mask)
    ref_chosen_lp = compute_response_logprob(ref_chosen_logits, chosen_ids, chosen_mask)
    ref_rejected_lp = compute_response_logprob(ref_rejected_logits, rejected_ids, rejected_mask)

    # Margins
    policy_margin = policy_chosen_lp - policy_rejected_lp
    ref_margin = ref_chosen_lp - ref_rejected_lp
    dpo_margin = policy_margin - ref_margin

    # DPO loss: -logsigmoid(beta * dpo_margin)
    loss = -mx.log(mx.sigmoid(beta * dpo_margin)).mean()

    # Preference accuracy: how often dpo_margin > 0
    pref_acc = (dpo_margin > 0).astype(mx.float32).mean()

    return {
        "loss": loss,
        "policy_chosen_logprob": policy_chosen_lp.mean(),
        "policy_rejected_logprob": policy_rejected_lp.mean(),
        "policy_margin": policy_margin.mean(),
        "ref_chosen_logprob": ref_chosen_lp.mean(),
        "ref_rejected_logprob": ref_rejected_lp.mean(),
        "ref_margin": ref_margin.mean(),
        "dpo_margin": dpo_margin.mean(),
        "preference_accuracy": pref_acc,
    }


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------

def build_debug_dpo_batch(
    tokenizer,
    sample: dict,
    policy_model,
    ref_model,
    beta: float,
    checkpoint_provenance: dict,
) -> dict:
    """Build debug dump for one DPO sample."""
    import mlx.core as mx

    messages = sample["_raw"]["messages"]
    chosen = sample["_raw"]["chosen"]
    rejected = sample["_raw"]["rejected"]

    prompt_text = render_prompt(tokenizer, messages)

    chosen_ids_arr = mx.array([sample["chosen_ids"]], dtype=mx.int32)
    rejected_ids_arr = mx.array([sample["rejected_ids"]], dtype=mx.int32)
    chosen_mask_arr = mx.array([sample["chosen_mask"]], dtype=mx.float32)
    rejected_mask_arr = mx.array([sample["rejected_mask"]], dtype=mx.float32)

    metrics = compute_dpo_loss(
        policy_model, ref_model,
        chosen_ids_arr, rejected_ids_arr,
        chosen_mask_arr, rejected_mask_arr,
        beta,
    )

    # Shift check: show the actual prompt/response boundary.
    shift_check = []
    full_ids = sample["chosen_ids"]
    prompt_len = sample["prompt_len"]
    shift_start = max(0, prompt_len - 4)
    shift_end = min(len(full_ids) - 1, prompt_len + 16)
    for i in range(shift_start, shift_end):
        label_pos = i + 1
        is_response = label_pos >= prompt_len
        shift_check.append({
            "input_pos": i,
            "label_pos": label_pos,
            "input_token": tokenizer.decode([full_ids[i]]),
            "label_token": tokenizer.decode([full_ids[label_pos]]) if is_response else "[masked]",
            "is_response": is_response,
        })

    def token_window(ids: list[int], mask: list[int]) -> list[dict]:
        start = max(0, prompt_len - 6)
        end = min(len(ids), prompt_len + 20)
        return [
            {
                "position": pos,
                "token_id": ids[pos],
                "token_text": tokenizer.decode([ids[pos]]),
                "response_mask": bool(mask[pos]),
            }
            for pos in range(start, end)
        ]

    # Mask summary
    chosen_mask_sum = int(sum(sample["chosen_mask"]))
    rejected_mask_sum = int(sum(sample["rejected_mask"]))

    return {
        "sample_id": sample["metadata"].get("problem_id", ""),
        "checkpoint_provenance": checkpoint_provenance,
        "prompt_text": prompt_text,
        "chosen_text": chosen,
        "rejected_text": rejected,
        "prompt_token_count": sample["prompt_len"],
        "chosen_token_ids": sample["chosen_ids"][:50],
        "rejected_token_ids": sample["rejected_ids"][:50],
        "chosen_boundary_window": token_window(sample["chosen_ids"], sample["chosen_mask"]),
        "rejected_boundary_window": token_window(sample["rejected_ids"], sample["rejected_mask"]),
        "chosen_response_mask_summary": {
            "total_tokens": len(sample["chosen_ids"]),
            "prompt_tokens": sample["prompt_len"],
            "response_tokens": chosen_mask_sum,
        },
        "rejected_response_mask_summary": {
            "total_tokens": len(sample["rejected_ids"]),
            "prompt_tokens": sample["prompt_len"],
            "response_tokens": rejected_mask_sum,
        },
        "chosen_response_logprob": round(float(metrics["policy_chosen_logprob"].item()), 6),
        "rejected_response_logprob": round(float(metrics["policy_rejected_logprob"].item()), 6),
        "ref_chosen_response_logprob": round(float(metrics["ref_chosen_logprob"].item()), 6),
        "ref_rejected_response_logprob": round(float(metrics["ref_rejected_logprob"].item()), 6),
        "policy_margin": round(float(metrics["policy_margin"].item()), 6),
        "ref_margin": round(float(metrics["ref_margin"].item()), 6),
        "dpo_margin": round(float(metrics["dpo_margin"].item()), 6),
        "shift_check": shift_check,
        "shift_check_summary": {
            "uses_next_token_labels": True,
            "self_token_prediction": False,
            "shown_input_positions": [shift_start, shift_end - 1],
        },
        "truncated": (
            len(sample["chosen_ids"]) >= 2048 or
            len(sample["rejected_ids"]) >= 2048
        ),
    }


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
        "target_keys": sorted(target_keys),
        "matched_module_count": len(target_hits),
        "matched_modules": target_hits,
        "num_layers": num_layers,
        "trainable_param_count": trainable,
        "total_param_count": total,
    }

    if verbose:
        print(f"  LoRA target modules: {target_modules}")
        print(f"  LoRA matched modules ({len(target_hits)})")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")

    return model, lora_info


def save_adapter_checkpoint(model, ckpt_dir: Path, lora_cfg: dict, lora_info: dict) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = ckpt_dir / "adapters.safetensors"
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)

    rank = lora_cfg.get("r", 32)
    scale = lora_cfg.get("alpha", 32) / rank
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": lora_info.get("num_layers"),
        "target_modules": lora_info.get("target_modules", []),
        "matched_module_count": lora_info.get("matched_module_count", 0),
        "trainable_param_count": lora_info.get("trainable_param_count", 0),
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": lora_cfg.get("dropout", 0.0),
            "keys": lora_info.get("target_keys", []),
        },
    }
    config_file = ckpt_dir / "adapter_config.json"
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    return {
        "checkpoint_type": "adapter",
        "adapter_file": str(adapter_file),
        "adapter_config_file": str(config_file),
        "adapter_file_size": adapter_file.stat().st_size,
    }


def resolve_adapter_file(adapter_path: str | Path | None) -> Path | None:
    """Resolve an adapter directory or safetensors file to its weight file."""
    if adapter_path is None:
        return None

    path = Path(adapter_path)
    adapter_file = path / "adapters.safetensors" if path.is_dir() else path
    if not adapter_file.is_file():
        raise FileNotFoundError(f"LoRA adapter weights not found: {adapter_file}")
    if adapter_file.suffix != ".safetensors":
        raise ValueError(f"LoRA adapter must be a .safetensors file: {adapter_file}")
    return adapter_file


def load_adapter_weights(model, adapter_file: Path | None) -> str | None:
    """Load a resolved LoRA adapter after the matching LoRA layers exist."""
    if adapter_file is None:
        return None
    model.load_weights(str(adapter_file), strict=False)
    return str(adapter_file)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    lock_path = acquire_training_lock(args.runs_dir)
    print(f"Training lock acquired: {lock_path}")

    config = build_config(str(args.base_config), str(args.config))

    # CLI overrides
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.beta is not None:
        config.setdefault("dpo", {})["beta"] = args.beta
    if args.data_path is not None:
        config.setdefault("data", {})["path"] = str(args.data_path)
    policy_adapter_arg = args.policy_adapter_path or args.policy_checkpoint
    ref_adapter_arg = args.ref_adapter_path or args.ref_checkpoint
    if policy_adapter_arg is not None:
        config.setdefault("dpo", {})["policy_adapter_path"] = str(policy_adapter_arg)
    if args.ref_model_path is not None:
        config.setdefault("dpo", {})["ref_model_path"] = str(args.ref_model_path)
    if ref_adapter_arg is not None:
        config.setdefault("dpo", {})["ref_adapter_path"] = str(ref_adapter_arg)

    # Create run
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

    run_dir = Path(run_dir)
    run_id = run_dir.name
    print(f"Run created: {run_id} -> {run_dir}")

    # Hardware snapshot
    try:
        snap = snapshot_hardware()
        append_hardware_log(run_dir, snap)
    except Exception as e:
        print(f"  [warn] hardware snapshot failed: {e}")

    # Run training
    try:
        exit_status = _run_training(config, run_dir, args)
    except Exception as exc:
        _handle_failure(run_dir, exc, args.runs_dir)
        return 1

    update_run_status(run_id, exit_status, args.runs_dir)
    print(f"\nRun {run_id} finished with status: {exit_status}")
    return 0


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------

def _run_training(config: dict, run_dir: Path, args) -> str:
    run_id = run_dir.name
    all_warnings: list[str] = []

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx.utils import tree_map

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    dpo_cfg = config.get("dpo", {})

    model_path = model_cfg.get("path", "models/qwen3_0_6b/mlx")
    data_path = Path(data_cfg.get("path", "data/math/splits/dpo_v1.jsonl"))
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    batch_size = train_cfg.get("batch_size", 2)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    lr = train_cfg.get("learning_rate", 5e-7)
    seed = train_cfg.get("seed", 42)
    max_steps = train_cfg.get("max_steps")
    num_epochs = int(train_cfg.get("num_epochs", 1))
    beta = dpo_cfg.get("beta", 0.1)
    ref_model_path = dpo_cfg.get("ref_model_path", model_path)
    policy_adapter_file = resolve_adapter_file(dpo_cfg.get("policy_adapter_path"))
    ref_adapter_file = resolve_adapter_file(dpo_cfg.get("ref_adapter_path"))

    if policy_adapter_file is None:
        raise ValueError(
            "DPO policy_adapter_path is required. M2 must start from the validated M1 SFT adapter."
        )
    if (policy_adapter_file or ref_adapter_file) and not lora_cfg.get("enabled", True):
        raise ValueError("Loading a LoRA adapter requires lora.enabled=true")

    mx.random.seed(seed)

    # ------ Load policy model ------
    print(f"Loading policy model: {model_path} ...")
    t0 = time.time()
    policy_model, tokenizer = mlx_load(model_path)
    print(f"  Policy model loaded in {time.time() - t0:.1f}s")

    # Apply LoRA to policy
    lora_info: dict = {}
    if lora_cfg.get("enabled", True):
        policy_model, lora_info = apply_lora(policy_model, lora_cfg)
    else:
        lora_info = {
            "enabled": False,
            "target_modules": [],
            "trainable_param_count": _count_params(policy_model.trainable_parameters()),
            "total_param_count": _count_params(policy_model.parameters()),
        }
    loaded_policy_adapter = load_adapter_weights(policy_model, policy_adapter_file)
    mx.eval(policy_model.parameters())
    print(f"  Policy adapter loaded: {loaded_policy_adapter}")

    # ------ Load reference model (frozen) ------
    print(f"Loading reference model: {ref_model_path} ...")
    t0 = time.time()
    ref_model, _ = mlx_load(ref_model_path)
    if ref_adapter_file is not None:
        ref_model, _ = apply_lora(ref_model, lora_cfg, verbose=False)
        loaded_ref_adapter = load_adapter_weights(ref_model, ref_adapter_file)
        mx.eval(ref_model.parameters())
        print(f"  Reference adapter loaded: {loaded_ref_adapter}")
    else:
        loaded_ref_adapter = None
    ref_model.freeze()
    ref_trainable = _count_params(ref_model.trainable_parameters())
    print(f"  Reference model loaded in {time.time() - t0:.1f}s")
    print(f"  Reference trainable params: {ref_trainable} (should be 0 after freeze)")

    checkpoint_provenance = {
        "policy_base_model_path": str(model_path),
        "policy_adapter_file": loaded_policy_adapter,
        "reference_base_model_path": str(ref_model_path),
        "reference_adapter_file": loaded_ref_adapter,
        "reference_frozen": True,
        "response_logprob_reduction": "sum",
    }
    provenance_path = run_dir / "logs" / "checkpoint_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_provenance, f, indent=2, ensure_ascii=False)

    # ------ Load dataset ------
    print(f"\nLoading DPO dataset: {data_path} ...")
    raw_data = load_dpo_dataset(data_path)
    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        raw_data = raw_data[:int(max_samples)]
    print(f"  {len(raw_data)} pairs loaded")

    # ------ Tokenize ------
    print("Tokenizing pairs ...")
    tokenized: list[dict] = []
    skipped = 0
    for item in raw_data:
        prompt_suffix = item.get("prompt_suffix", "")
        no_eos = item.get("no_eos", False)
        tk = tokenize_pair(
            tokenizer, item["messages"],
            item["chosen"], item["rejected"],
            max_seq_length,
            prompt_suffix=prompt_suffix, no_eos=no_eos,
        )
        if tk is not None:
            tk["metadata"] = item.get("metadata", {})
            tk["_raw"] = item
            tokenized.append(tk)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} empty pairs")
    print(f"  {len(tokenized)} pairs tokenized")

    if not tokenized:
        raise RuntimeError("No valid pairs after tokenization")

    # ------ Boundary-only debug assertions ------
    has_prompt_suffix = any(item.get("prompt_suffix") for item in raw_data)
    if has_prompt_suffix:
        print("\n[Boundary-Only] Running debug assertions ...")
        for idx, (item, tk) in enumerate(zip(raw_data, tokenized)):
            pid = item.get("metadata", {}).get("problem_id", f"pair_{idx}")
            prompt_suffix = item.get("prompt_suffix", "")
            prompt_len = tk["prompt_len"]

            # 1. Prompt must end with prompt_suffix tokens
            if prompt_suffix:
                suffix_ids = tokenizer.encode(prompt_suffix)
                for seq_key in ["chosen_ids", "rejected_ids"]:
                    actual_suffix = tk[seq_key][prompt_len - len(suffix_ids):prompt_len]
                    assert list(actual_suffix) == suffix_ids, (
                        f"{pid}/{seq_key}: prompt does not end with prompt_suffix. "
                        f"Expected {suffix_ids}, got {list(actual_suffix)}"
                    )

            # 2. Response mask must sum to 1 for both chosen and rejected
            chosen_mask_sum = sum(tk["chosen_mask"])
            rejected_mask_sum = sum(tk["rejected_mask"])
            assert chosen_mask_sum == 1, (
                f"{pid}: chosen_mask sum={chosen_mask_sum}, expected 1"
            )
            assert rejected_mask_sum == 1, (
                f"{pid}: rejected_mask sum={rejected_mask_sum}, expected 1"
            )

            # 3. The single supervised label must be the chosen/rejected token
            chosen_response_start = prompt_len
            chosen_label = tk["chosen_ids"][chosen_response_start]
            chosen_text = tokenizer.decode([chosen_label])
            assert chosen_text == item["chosen"], (
                f"{pid}: chosen supervised label '{chosen_text}' != '{item['chosen']}'"
            )
            rejected_label = tk["rejected_ids"][chosen_response_start]
            rejected_text = tokenizer.decode([rejected_label])
            assert rejected_text == item["rejected"], (
                f"{pid}: rejected supervised label '{rejected_text}' != '{item['rejected']}'"
            )

        print(f"  All {len(tokenized)} boundary-only DPO assertions passed")

    # ------ Debug dump ------
    if args.debug_batch and tokenized:
        debug_path = run_dir / "logs" / "debug_dpo_batch.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[Debug] Writing debug dump to {debug_path} ...")
        debug_dump = build_debug_dpo_batch(
            tokenizer, tokenized[0], policy_model, ref_model, beta, checkpoint_provenance,
        )
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(debug_dump, f, indent=2, ensure_ascii=False)
        print(f"  sample_id: {debug_dump['sample_id']}")
        print(f"  prompt_tokens: {debug_dump['prompt_token_count']}")
        print(f"  chosen_response_tokens: {debug_dump['chosen_response_mask_summary']['response_tokens']}")
        print(f"  rejected_response_tokens: {debug_dump['rejected_response_mask_summary']['response_tokens']}")
        print(f"  policy_chosen_logprob: {debug_dump['chosen_response_logprob']:.4f}")
        print(f"  policy_rejected_logprob: {debug_dump['rejected_response_logprob']:.4f}")
        print(f"  ref_chosen_logprob: {debug_dump['ref_chosen_response_logprob']:.4f}")
        print(f"  ref_rejected_logprob: {debug_dump['ref_rejected_response_logprob']:.4f}")
        print(f"  policy_margin: {debug_dump['policy_margin']:.4f}")
        print(f"  ref_margin: {debug_dump['ref_margin']:.4f}")
        print(f"  dpo_margin: {debug_dump['dpo_margin']:.4f}")
        print(f"  truncated: {debug_dump['truncated']}")

    # ------ Optimizer ------
    optimizer = optim.Adam(learning_rate=lr)

    # ------ DPO loss and grad function ------
    def dpo_loss_fn(model, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta_val):
        return compute_dpo_loss(
            model, ref_model,
            chosen_ids, rejected_ids,
            chosen_mask, rejected_mask,
            beta_val,
        )["loss"]

    loss_and_grad_fn = nn.value_and_grad(policy_model, dpo_loss_fn)

    # ------ Train loop ------
    update_groups = build_microbatch_groups(
        tokenized,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_epochs=num_epochs,
        max_steps=max_steps,
    )
    effective_steps = len(update_groups)

    print(f"\nStarting DPO training: {effective_steps} steps")
    print(f"  epochs={num_epochs}, batch_size={batch_size}, grad_accum={grad_accum}, lr={lr}, beta={beta}")
    print("=" * 60)

    update_run_status(run_id, "running", args.runs_dir)

    step_metrics_list: list[dict] = []
    pairs_processed = 0
    start_time = time.time()

    policy_model.train()
    ref_model.eval()

    for step, micro_batches in enumerate(update_groups, start=1):
        micro_losses: list[float] = []
        accumulated_grads = None

        for micro_batch in micro_batches:
            collated = collate_dpo_batch(micro_batch)

            loss, grads = loss_and_grad_fn(
                policy_model,
                collated["chosen_ids"],
                collated["rejected_ids"],
                collated["chosen_mask"],
                collated["rejected_mask"],
                beta,
            )

            micro_losses.append(float(loss.item()))
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda x, y: x + y, accumulated_grads, grads)

        avg_loss = sum(micro_losses) / len(micro_losses)
        if len(micro_batches) > 1:
            accumulated_grads = tree_map(lambda x: x / len(micro_batches), accumulated_grads)
        grad_norm = _grad_norm(accumulated_grads)
        optimizer.update(policy_model, accumulated_grads)
        mx.eval(policy_model.parameters(), optimizer.state)
        mx.clear_cache()

        # Compute detailed metrics on the final micro-batch in this update.
        metric_batch = micro_batches[-1]
        collated = collate_dpo_batch(metric_batch)
        detail = compute_dpo_loss(
            policy_model, ref_model,
            collated["chosen_ids"], collated["rejected_ids"],
            collated["chosen_mask"], collated["rejected_mask"],
            beta,
        )

        pairs_processed += sum(len(batch) for batch in micro_batches)
        elapsed = time.time() - start_time
        tokens_per_sec = (pairs_processed * max_seq_length) / max(elapsed, 0.001)

        step_metrics = {
            "step": step,
            "dpo_loss": round(avg_loss, 6),
            "chosen_logprob": round(float(detail["policy_chosen_logprob"].item()), 6),
            "rejected_logprob": round(float(detail["policy_rejected_logprob"].item()), 6),
            "logprob_margin": round(float(detail["policy_margin"].item()), 6),
            "ref_chosen_logprob": round(float(detail["ref_chosen_logprob"].item()), 6),
            "ref_rejected_logprob": round(float(detail["ref_rejected_logprob"].item()), 6),
            "ref_logprob_margin": round(float(detail["ref_margin"].item()), 6),
            "dpo_margin": round(float(detail["dpo_margin"].item()), 6),
            "preference_accuracy": round(float(detail["preference_accuracy"].item()), 4),
            "pairs_in_update": sum(len(batch) for batch in micro_batches),
            "micro_batches_in_update": len(micro_batches),
            "chosen_token_length": int(collated["chosen_mask"].sum().item()),
            "rejected_token_length": int(collated["rejected_mask"].sum().item()),
            "learning_rate": lr,
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
            "peak_memory": None,
            "tokens_per_sec": round(tokens_per_sec, 1),
        }
        step_metrics_list.append(step_metrics)
        append_metric(run_dir, "train", **step_metrics)

        grad_text = f"{grad_norm:.4f}" if grad_norm is not None else "NA"
        print(f"  step {step}/{effective_steps}  loss={avg_loss:.4f}  "
              f"margin={step_metrics['dpo_margin']:.4f}  "
              f"pref_acc={step_metrics['preference_accuracy']:.2f}  "
              f"grad={grad_text}  {elapsed:.1f}s")

    train_time = time.time() - start_time
    print(f"\nTraining finished in {train_time:.1f}s")

    # ------ Save checkpoint ------
    ckpt_dir = run_dir / "checkpoints" / "final"
    ckpt_info = save_adapter_checkpoint(policy_model, ckpt_dir, lora_cfg, lora_info)
    print(f"Checkpoint saved: {ckpt_info['adapter_file']} ({ckpt_info['adapter_file_size']:,} bytes)")

    # ------ Final metrics summary ------
    if step_metrics_list:
        first = step_metrics_list[0]
        last = step_metrics_list[-1]
        print(f"\n=== DPO Training Summary ===")
        print(f"  Loss: {first['dpo_loss']:.4f} -> {last['dpo_loss']:.4f}")
        print(f"  Policy margin: {first['logprob_margin']:.4f} -> {last['logprob_margin']:.4f}")
        print(f"  DPO margin: {first['dpo_margin']:.4f} -> {last['dpo_margin']:.4f}")
        print(f"  Preference accuracy: {first['preference_accuracy']:.2f} -> {last['preference_accuracy']:.2f}")
        print(f"  Ref margin: {last['ref_logprob_margin']:.4f}")

    # ------ Check for NaN/Inf ------
    losses = [m["dpo_loss"] for m in step_metrics_list]
    if any(math.isnan(x) or math.isinf(x) for x in losses):
        all_warnings.append("CRITICAL: NaN or Inf in DPO loss")
        print("\nWARNING: NaN or Inf detected in DPO loss!")

    if all_warnings:
        diag_path = run_dir / "logs" / "diagnosis.md"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("w") as f:
            f.write("# DPO Training Diagnosis\n\n")
            for w in all_warnings:
                f.write(f"- {w}\n")

    return "completed"


# ---------------------------------------------------------------------------
# Failure handler
# ---------------------------------------------------------------------------

def _handle_failure(run_dir: Path, exc: Exception, runs_dir: str):
    run_id = run_dir.name
    print(f"\n{'='*60}")
    print(f"DPO TRAINING FAILED: {exc}")
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
