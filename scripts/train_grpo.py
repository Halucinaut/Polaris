#!/usr/bin/env python3
"""
M2.5 GRPO training script (MLX / mlx-lm).

Implements real GRPO with rollout sampling, Math reward,
group-relative advantage, response-only logprob, reference KL,
and clipped policy objective.

Usage:
    # Dry-run: validate config, data schema, reward protocol, no model loading
    python scripts/train_grpo.py --dry-run

    # Real training (not yet — M2.5 sanity pending)
    python scripts/train_grpo.py --config configs/qwen3_0_6b/grpo_math.yaml

Exits 0 on success, 1 on handled failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from polaris.config import build_config
from polaris.json_records import load_json_record_stream
from polaris.rewards.math_answer import compute_math_reward
from polaris.trainers.base import compute_group_relative_advantage


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
    p = argparse.ArgumentParser(description="Polaris M2.5 GRPO Trainer")
    p.add_argument("--config", type=Path,
                   default=Path("configs/qwen3_0_6b/grpo_math.yaml"),
                   help="Experiment config YAML")
    p.add_argument("--base-config", type=Path,
                   default=Path("configs/base.yaml"),
                   help="Base config YAML")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config, data schema, and reward protocol without loading models")
    p.add_argument("--runs-dir", type=str, default="runs",
                   help="Runs root directory")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max training steps")
    p.add_argument("--mode", type=str, default="dry-run",
                   choices=["dry-run", "train"],
                   help="Execution mode: dry-run (default) or train")
    p.add_argument("--confirm-real-training", action="store_true",
                   help="Required with --mode train to confirm real model training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_grpo_config(config: dict) -> list[str]:
    """Validate GRPO-specific config fields. Returns list of errors."""
    errors: list[str] = []
    grpo_cfg = config.get("grpo", {})

    required_fields = [
        "policy_adapter_path", "ref_model_path", "ref_adapter_path",
        "group_size", "max_completion_length", "kl_coef", "clip_range",
    ]
    for field in required_fields:
        if field not in grpo_cfg:
            errors.append(f"Missing required grpo.{field}")

    # Numeric validations
    gs = grpo_cfg.get("group_size")
    if gs is not None and (not isinstance(gs, int) or gs < 1):
        errors.append(f"grpo.group_size must be a positive int, got {gs}")

    mcl = grpo_cfg.get("max_completion_length")
    if mcl is not None and (not isinstance(mcl, int) or mcl < 1):
        errors.append(f"grpo.max_completion_length must be a positive int, got {mcl}")

    kl = grpo_cfg.get("kl_coef")
    if kl is not None and (not isinstance(kl, (int, float)) or kl < 0):
        errors.append(f"grpo.kl_coef must be non-negative, got {kl}")

    cr = grpo_cfg.get("clip_range")
    if cr is not None and (not isinstance(cr, (int, float)) or cr <= 0 or cr > 1):
        errors.append(f"grpo.clip_range must be in (0, 1], got {cr}")

    temp = grpo_cfg.get("rollout_temperature", 1.0)
    if not isinstance(temp, (int, float)) or temp <= 0:
        errors.append(f"grpo.rollout_temperature must be positive, got {temp}")
    elif temp != 1.0:
        errors.append(
            f"grpo.rollout_temperature must be 1.0 for on-policy GRPO, got {temp}"
        )

    top_p = grpo_cfg.get("rollout_top_p", 1.0)
    if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1:
        errors.append(f"grpo.rollout_top_p must be in (0, 1], got {top_p}")
    elif top_p != 1.0:
        errors.append(
            f"grpo.rollout_top_p must be 1.0 for on-policy GRPO, got {top_p}"
        )

    reward_cfg = grpo_cfg.get("reward", {})
    if not isinstance(reward_cfg, dict):
        errors.append("grpo.reward must be a dict")
    else:
        for key in ["correct", "incorrect", "unparseable", "empty"]:
            if key not in reward_cfg:
                errors.append(f"Missing grpo.reward.{key}")

    return errors


def validate_data_schema(records: list[dict]) -> list[str]:
    """Validate that data records have the required fields for GRPO.

    GRPO needs prompt-only data with reference answers.
    Expected: messages (list of dicts), metadata.answer (str).
    """
    errors: list[str] = []
    if not records:
        errors.append("Dataset is empty")
        return errors

    sample = records[0]
    if "messages" not in sample:
        errors.append("Records missing 'messages' field")
    elif not isinstance(sample["messages"], list):
        errors.append("'messages' must be a list")

    meta = sample.get("metadata", {})
    if "answer" not in meta:
        errors.append("metadata missing 'answer' field (needed for reward computation)")

    return errors


# ---------------------------------------------------------------------------
# Reward protocol validation (dry-run)
# ---------------------------------------------------------------------------

def validate_reward_protocol(reward_config: dict) -> list[str]:
    """Test reward function with synthetic inputs to verify protocol."""
    errors: list[str] = []

    # Test: correct answer
    result = compute_math_reward(
        "<think>2+3=5</think>\n\\boxed{5}",
        "5",
        reward_config,
    )
    if not result.answer_correct:
        errors.append(f"Reward protocol: expected correct=True, got {result.answer_correct}")
    if result.reward <= 0:
        errors.append(f"Reward protocol: correct answer reward should be > 0, got {result.reward}")

    # Test: incorrect answer
    result = compute_math_reward(
        "<think>2+3=6</think>\n\\boxed{6}",
        "5",
        reward_config,
    )
    if result.answer_correct:
        errors.append(f"Reward protocol: expected correct=False, got {result.answer_correct}")

    # Test: empty output
    result = compute_math_reward("", "5", reward_config)
    if result.invalid_reason != "empty_output":
        errors.append(f"Reward protocol: expected empty_output, got {result.invalid_reason}")

    # Test: unparseable output
    result = compute_math_reward("I think the answer is probably around five", "5", reward_config)
    if result.invalid_reason != "unparseable":
        errors.append(f"Reward protocol: expected unparseable, got {result.invalid_reason}")

    return errors


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------

def preflight_config_and_data(config: dict) -> tuple[list[str], list[dict] | None]:
    """Validate GRPO config and data path/schema. No side effects.

    Returns (errors, records_or_None).
    """
    errors = validate_grpo_config(config)

    data_path = config.get("data", {}).get("path")
    if not data_path:
        errors.append("BLOCKED: No data.path in config")
        return errors, None
    if not Path(data_path).exists():
        errors.append(f"BLOCKED: Data file not found: {data_path}")
        return errors, None

    try:
        records = load_json_record_stream(Path(data_path))
    except Exception as exc:
        errors.append(f"Data loading failed: {exc}")
        return errors, None

    schema_errors = validate_data_schema(records)
    errors.extend(schema_errors)
    return errors, records


def preflight_train(config: dict) -> list[str]:
    """Full preflight for train mode. No side effects.

    Validates config, data, adapter paths, group_size, lora.dropout.
    Must run before create_run, lock, model loading, or hardware snapshot.
    """
    errors, _records = preflight_config_and_data(config)
    grpo_cfg = config.get("grpo", {})
    lora_cfg = config.get("lora", {})

    # Adapter paths must resolve to existing safetensors files
    for key in ["policy_adapter_path", "ref_adapter_path"]:
        adapter_path = grpo_cfg.get(key)
        if not adapter_path:
            continue
        try:
            resolve_adapter_file(adapter_path)
        except FileNotFoundError as exc:
            errors.append(f"BLOCKED: {key}: {exc}")
        except ValueError as exc:
            errors.append(f"ERROR: {key}: {exc}")

    # group_size >= 2 (need at least 2 completions for group-relative advantage)
    gs = grpo_cfg.get("group_size")
    if gs is not None and isinstance(gs, int) and gs < 2:
        errors.append(f"ERROR: grpo.group_size must be >= 2 for GRPO, got {gs}")

    # lora.dropout must be 0.0 for M2.5:
    # dropout during rollout (eval) and training would create distribution mismatch
    dropout = lora_cfg.get("dropout", 0.0)
    if dropout != 0.0:
        errors.append(
            f"ERROR: lora.dropout must be 0.0 for M2.5 GRPO "
            f"(rollout eval/train distribution must match), got {dropout}"
        )

    return errors


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(config: dict) -> int:
    """Validate config, data schema, and reward protocol without loading models."""
    print("=== GRPO Dry-Run ===\n")

    all_errors: list[str] = []

    # 1-2. Config + data path & schema (shared preflight)
    print("[1/4] Validating GRPO config ...")
    preflight_errors, records = preflight_config_and_data(config)
    config_errors = validate_grpo_config(config)
    data_errors = [e for e in preflight_errors if e not in config_errors]

    if config_errors:
        all_errors.extend(config_errors)
        for e in config_errors:
            print(f"  ERROR: {e}")
    else:
        print("  Config OK")

    print("\n[2/4] Validating data path & schema ...")
    if data_errors:
        all_errors.extend(data_errors)
        for e in data_errors:
            print(f"  {e}")
    elif records is not None:
        print(f"  Data OK: {len(records)} records, schema valid")
        print(f"  Sample problem_id: {records[0].get('metadata', {}).get('problem_id', 'N/A')}")
        print(f"  Sample answer: {records[0].get('metadata', {}).get('answer', 'N/A')[:50]}")

    # 3. Reward protocol validation
    print("\n[3/4] Validating reward protocol ...")
    reward_config = config.get("grpo", {}).get("reward", {})
    if reward_config:
        reward_errors = validate_reward_protocol(reward_config)
        if reward_errors:
            all_errors.extend(reward_errors)
            for e in reward_errors:
                print(f"  ERROR: {e}")
        else:
            print("  Reward protocol OK: correct, incorrect, empty, unparseable all behave as expected")
    else:
        print("  SKIPPED: No grpo.reward config")

    # 4. Advantage protocol validation
    print("\n[4/4] Validating advantage protocol ...")
    group_size = config.get("grpo", {}).get("group_size", 8)
    test_rewards = [1.0] * group_size
    stats = compute_group_relative_advantage(test_rewards)
    if not stats.zero_variance:
        all_errors.append("Advantage: identical rewards should produce zero_variance=True")
        print("  ERROR: identical rewards should produce zero_variance=True")
    else:
        print(f"  Advantage protocol OK: zero-variance guard works (group_size={group_size})")

    # Non-zero variance test
    test_rewards2 = [0.0, 0.5, 1.0]
    stats2 = compute_group_relative_advantage(test_rewards2)
    if stats2.zero_variance:
        all_errors.append("Advantage: varied rewards should not produce zero_variance=True")
        print("  ERROR: varied rewards should not produce zero_variance=True")
    else:
        print(f"  Advantage protocol OK: normalization works (advantages={[round(a, 3) for a in stats2.advantages]})")

    # Summary
    print(f"\n{'='*40}")
    if all_errors:
        print(f"Dry-run FAILED with {len(all_errors)} error(s)")
        return 1
    else:
        print("Dry-run PASSED: config, data schema, reward, and advantage protocols are valid")
        return 0


# ---------------------------------------------------------------------------
# LoRA helpers (reuse from train_dpo.py pattern)
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


def resolve_adapter_file(adapter_path: str | Path | None) -> Path | None:
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
    if adapter_file is None:
        return None
    model.load_weights(str(adapter_file), strict=False)
    return str(adapter_file)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def build_grpo_batch(rollout_ids_list, rollout_old_token_lps, prompt_len, tokenizer):
    """Build padded batch tensors from rollout completions.

    Handles EOS-inclusive sequences: EOS token is part of generated_ids and
    its old logprob is stored in rollout_old_token_lps at the same index.

    Returns:
        (input_ids, response_mask, old_token_logprobs_raw) as numpy arrays.
        - input_ids: (B, max_len) int32
        - response_mask: (B, max_len) float32, 1.0 for response tokens
        - old_token_logprobs_raw: (B, max_len) float32, raw per-token logprobs
          (prompt positions are 0.0; causal shift applied inside compute_grpo_loss)
    """
    import numpy as np

    # Validate input consistency
    for i, (ids, old_lps) in enumerate(zip(rollout_ids_list, rollout_old_token_lps)):
        expected = len(ids) - prompt_len
        if len(old_lps) != expected:
            raise ValueError(
                f"Sample {i}: old_token_logprobs length {len(old_lps)} "
                f"!= response length {expected} "
                f"(seq_len={len(ids)}, prompt_len={prompt_len})"
            )

    max_len = max(len(ids) for ids in rollout_ids_list)
    pad_id = tokenizer.pad_token_id or 0

    batch_ids = []
    batch_masks = []
    batch_old_lps = []
    for i, ids in enumerate(rollout_ids_list):
        seq_len = len(ids)
        resp_len = seq_len - prompt_len
        padded = ids + [pad_id] * (max_len - seq_len)
        mask = [0.0] * prompt_len + [1.0] * resp_len + [0.0] * (max_len - seq_len)
        # old lp: 0.0 for prompt + per-token lp for response (incl. EOS) + 0.0 for padding
        old_lp = [0.0] * prompt_len + list(rollout_old_token_lps[i]) + [0.0] * (max_len - seq_len)
        batch_ids.append(padded)
        batch_masks.append(mask)
        batch_old_lps.append(old_lp)

    return (
        np.array(batch_ids, dtype=np.int32),
        np.array(batch_masks, dtype=np.float32),
        np.array(batch_old_lps, dtype=np.float32),
    )


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
    config = build_config(str(args.base_config), str(args.config))

    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps

    if args.dry_run or args.mode == "dry-run":
        return dry_run(config)

    if not (args.mode == "train" and args.confirm_real_training):
        print("ERROR: Real training requires both --mode train and --confirm-real-training.")
        print("  Example: python scripts/train_grpo.py --mode train --confirm-real-training --config ...")
        return 1

    # Preflight: validate all config, data, adapters before any side effects
    preflight_errors = preflight_train(config)
    if preflight_errors:
        print("PREFLIGHT FAILED:")
        for e in preflight_errors:
            print(f"  {e}")
        return 1

    # --- Real training path (requires models + data) ---
    from polaris.monitoring.hardware import snapshot_hardware, append_hardware_log
    from polaris.monitoring.metrics import append_metric
    from polaris.registry import create_run, update_run_status

    lock_path = _acquire_lock(args.runs_dir)
    print(f"Training lock acquired: {lock_path}")

    # Create run
    merged_tmp = Path("__tmp_merged_grpo_config.yaml")
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

    try:
        snap = snapshot_hardware()
        append_hardware_log(run_dir, snap)
    except Exception as e:
        print(f"  [warn] hardware snapshot failed: {e}")

    try:
        exit_status = _run_training(config, run_dir, args)
    except Exception as exc:
        _handle_failure(run_dir, exc, args.runs_dir)
        return 1

    update_run_status(run_id, exit_status, args.runs_dir)
    print(f"\nRun {run_id} finished with status: {exit_status}")
    return 0


def _acquire_lock(runs_dir: str):
    import atexit
    import os

    lock_path = Path(runs_dir) / ".train_grpo.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Another train_grpo.py run appears to be active: {lock_path} {existing}"
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")
        f.write(f"created_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _release():
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_release)
    return lock_path


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------

def _run_training(config: dict, run_dir: Path, args) -> str:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx.utils import tree_map

    from polaris.trainers.grpo import compute_grpo_loss, compute_response_logprob, _compute_token_logprobs

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    grpo_cfg = config.get("grpo", {})

    model_path = model_cfg.get("path", "models/qwen3_0_6b/mlx")
    data_path = Path(data_cfg.get("path", "data/math/splits/math_level_3_5.jsonl"))
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    lr = train_cfg.get("learning_rate", 1e-6)
    seed = train_cfg.get("seed", 42)
    num_epochs = int(train_cfg.get("num_epochs", 1))
    max_steps = train_cfg.get("max_steps")

    group_size = grpo_cfg.get("group_size", 8)
    max_completion_length = grpo_cfg.get("max_completion_length", 256)
    rollout_temperature = grpo_cfg.get("rollout_temperature", 1.0)
    rollout_top_p = grpo_cfg.get("rollout_top_p", 1.0)
    clip_range = grpo_cfg.get("clip_range", 0.2)
    kl_coef = grpo_cfg.get("kl_coef", 0.05)
    reward_config = grpo_cfg.get("reward", {})

    ref_model_path = grpo_cfg.get("ref_model_path", model_path)
    policy_adapter_file = resolve_adapter_file(grpo_cfg.get("policy_adapter_path"))
    ref_adapter_file = resolve_adapter_file(grpo_cfg.get("ref_adapter_path"))

    if policy_adapter_file is None:
        raise ValueError(
            "GRPO policy_adapter_path is required. M2.5 must start from the validated M1 SFT adapter."
        )

    # Config hash for provenance
    import hashlib
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]

    mx.random.seed(seed)

    # ------ Load policy model ------
    print(f"Loading policy model: {model_path} ...")
    t0 = time.time()
    policy_model, tokenizer = mlx_load(model_path)
    print(f"  Policy model loaded in {time.time() - t0:.1f}s")

    lora_info: dict = {}
    if lora_cfg.get("enabled", True):
        policy_model, lora_info = apply_lora(policy_model, lora_cfg)
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

    # Verify reference is frozen
    assert ref_trainable == 0, f"Reference model not frozen: {ref_trainable} trainable params"

    # ------ Checkpoint provenance ------
    checkpoint_provenance = {
        "policy_base_model_path": str(model_path),
        "policy_adapter_file": loaded_policy_adapter,
        "reference_base_model_path": str(ref_model_path),
        "reference_adapter_file": loaded_ref_adapter,
        "reference_frozen": True,
        "reference_in_ratio": False,
        "ratio_source": "current/old (rollout-frozen)",
        "kl_source": "forward_kl(current, ref)",
        "response_logprob_reduction": "sum",
        "old_logprob_capture": "per-token at rollout time",
        "group_size": group_size,
        "max_completion_length": max_completion_length,
        "resume_supported": False,
    }
    provenance_path = run_dir / "logs" / "checkpoint_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_provenance, f, indent=2, ensure_ascii=False)

    # ------ Load dataset ------
    print(f"\nLoading GRPO dataset: {data_path} ...")
    raw_data = load_json_record_stream(data_path)
    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        raw_data = raw_data[:int(max_samples)]
    print(f"  {len(raw_data)} problems loaded")

    # ------ Tokenize prompts ------
    print("Tokenizing prompts ...")
    tokenized_prompts: list[dict] = []
    skipped = 0
    for item in raw_data:
        messages = item.get("messages", [])
        metadata = item.get("metadata", {})
        answer = metadata.get("answer", "")
        if not answer:
            skipped += 1
            continue

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = rendered + ASSISTANT_HEADER
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) >= max_seq_length - max_completion_length:
            skipped += 1
            continue

        tokenized_prompts.append({
            "prompt_ids": prompt_ids,
            "prompt_len": len(prompt_ids),
            "reference_answer": answer,
            "problem_id": metadata.get("problem_id", ""),
            "prompt_text": prompt_text,
        })

    if skipped:
        print(f"  Skipped {skipped} problems (missing answer or too long)")
    print(f"  {len(tokenized_prompts)} problems tokenized")

    if not tokenized_prompts:
        raise RuntimeError("No valid problems after tokenization")

    # ------ Optimizer ------
    optimizer = optim.Adam(learning_rate=lr)

    # ------ GRPO loss function ------
    def grpo_loss_fn(model, input_ids, response_mask, advantages, kl_coef_val, clip_range_val, old_token_logprobs):
        return compute_grpo_loss(
            model, ref_model,
            input_ids, response_mask,
            advantages, kl_coef_val, clip_range_val,
            old_token_logprobs=old_token_logprobs,
        )["loss"]

    loss_and_grad_fn = nn.value_and_grad(policy_model, grpo_loss_fn)

    # ------ Training loop ------
    effective_steps = len(tokenized_prompts) * num_epochs
    if max_steps is not None:
        effective_steps = min(effective_steps, max_steps)

    print(f"\nStarting GRPO training: {effective_steps} steps")
    print(f"  group_size={group_size}, kl_coef={kl_coef}, clip_range={clip_range}")
    print(f"  rollout_temperature={rollout_temperature}, max_completion_length={max_completion_length}")
    print("=" * 60)

    update_run_status(run_dir.name, "running", args.runs_dir)

    import json as json_mod
    step_metrics_list: list[dict] = []
    start_time = time.time()
    global_step = 0
    all_warnings: list[str] = []

    ref_model.eval()

    for epoch in range(num_epochs):
        for prompt_idx, prompt_data in enumerate(tokenized_prompts):
            global_step += 1
            if max_steps is not None and global_step > max_steps:
                break

            prompt_ids = prompt_data["prompt_ids"]
            ref_answer = prompt_data["reference_answer"]
            prompt_len = prompt_data["prompt_len"]

            # ------ Rollout phase: eval mode, sample completions ------
            policy_model.eval()

            completions: list[str] = []
            completion_lengths: list[int] = []
            rollout_ids_list: list[list[int]] = []
            rollout_old_token_lps: list[list[float]] = []
            rollout_meta: list[dict] = []

            for comp_idx in range(group_size):
                input_ids_list = prompt_ids.copy()
                generated_ids: list[int] = []
                max_gen = min(max_completion_length, max_seq_length - prompt_len)
                per_token_lp: list[float] = []
                termination = "max_length"

                for _tok_pos in range(max_gen):
                    inp = mx.array([input_ids_list + generated_ids], dtype=mx.int32)
                    logits = policy_model(inp)
                    # Handle dict output (MLX modules may return dict)
                    logits = logits["logits"] if isinstance(logits, dict) else logits
                    next_logits = logits[0, -1, :] / max(rollout_temperature, 1e-6)

                    # Top-p sampling
                    if rollout_top_p < 1.0:
                        sorted_indices = mx.argsort(-next_logits)
                        sorted_logits = next_logits[sorted_indices]
                        cumprobs = mx.cumsum(mx.softmax(sorted_logits, axis=-1))
                        mask = cumprobs > rollout_top_p
                        mask[0] = False  # keep at least one
                        sorted_logits = mx.where(mask, float("-inf"), sorted_logits)
                        next_logits = mx.zeros_like(next_logits)
                        next_logits[sorted_indices] = sorted_logits

                    probs = mx.softmax(next_logits, axis=-1)
                    next_id = int(mx.random.categorical(probs).item())

                    # Record old policy logprob for this token
                    token_logprob = float(mx.log(probs[next_id]).item())
                    per_token_lp.append(token_logprob)

                    generated_ids.append(next_id)
                    if next_id == tokenizer.eos_token_id:
                        termination = "eos"
                        break

                completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                completions.append(completion_text)
                completion_lengths.append(len(generated_ids))
                rollout_ids_list.append(prompt_ids + generated_ids)
                rollout_old_token_lps.append(per_token_lp)
                rollout_meta.append({
                    "completion_idx": comp_idx,
                    "response_token_count": len(generated_ids),
                    "termination": termination,
                    "seed": seed,
                })

            # ------ Compute rewards ------
            rewards: list[float] = []
            invalid_count = 0
            for comp in completions:
                result = compute_math_reward(comp, ref_answer, reward_config)
                rewards.append(result.reward)
                if result.invalid_reason:
                    invalid_count += 1

            # ------ Group-relative advantage ------
            group_stats = compute_group_relative_advantage(rewards)

            # ------ Build padded batch ------
            batch_ids_np, batch_masks_np, batch_old_lps_np = build_grpo_batch(
                rollout_ids_list, rollout_old_token_lps, prompt_len, tokenizer,
            )
            input_ids_arr = mx.array(batch_ids_np, dtype=mx.int32)
            response_mask_arr = mx.array(batch_masks_np, dtype=mx.float32)
            old_token_lp_arr = mx.array(batch_old_lps_np, dtype=mx.float32)
            advantages_arr = mx.array(group_stats.advantages, dtype=mx.float32)

            # ------ GRPO training step: switch to train mode ------
            policy_model.train()

            loss, grads = loss_and_grad_fn(
                policy_model,
                input_ids_arr,
                response_mask_arr,
                advantages_arr,
                kl_coef,
                clip_range,
                old_token_lp_arr,
            )

            grad_norm = _grad_norm(grads)
            optimizer.update(policy_model, grads)
            mx.eval(policy_model.parameters(), optimizer.state)
            mx.clear_cache()

            # ------ Compute detailed metrics (post-update, with old logprobs) ------
            detail = compute_grpo_loss(
                policy_model, ref_model,
                input_ids_arr, response_mask_arr,
                advantages_arr, kl_coef, clip_range,
                old_token_logprobs=old_token_lp_arr,
            )

            elapsed = time.time() - start_time
            step_metrics = {
                "step": global_step,
                "reward_mean": round(group_stats.reward_mean, 6),
                "reward_std": round(group_stats.reward_std, 6),
                "zero_variance_group_count": 1 if group_stats.zero_variance else 0,
                "valid_advantage_count": sum(1 for a in group_stats.advantages if a != 0.0),
                "policy_logprob": round(float(detail["policy_logprob"].item()), 6),
                "old_policy_logprob": round(float(detail["old_policy_logprob"].item()), 6),
                "ref_logprob": round(float(detail["ref_logprob"].item()), 6),
                "KL": round(float(detail["kl"].item()), 6),
                "entropy": round(float(detail["entropy"].item()), 6),
                "loss": round(float(loss.item()), 6),
                "completion_length": round(sum(completion_lengths) / len(completion_lengths), 1),
                "response_token_count": sum(completion_lengths),
                "invalid_output_rate": round(invalid_count / group_size, 4),
                "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
                "approx_kl": round(float(detail["approx_kl"].item()), 6),
                "clip_fraction": round(float(detail["clip_fraction"].item()), 4),
            }
            step_metrics_list.append(step_metrics)
            append_metric(run_dir, "train", **step_metrics)

            if global_step % 10 == 0 or global_step == 1:
                grad_text = f"{grad_norm:.4f}" if grad_norm is not None else "NA"
                print(
                    f"  step {global_step}/{effective_steps}  "
                    f"loss={step_metrics['loss']:.4f}  "
                    f"reward={step_metrics['reward_mean']:.3f}±{step_metrics['reward_std']:.3f}  "
                    f"KL={step_metrics['KL']:.4f}  "
                    f"grad={grad_text}  {elapsed:.1f}s"
                )

    train_time = time.time() - start_time
    print(f"\nTraining finished in {train_time:.1f}s")

    # ------ Save final checkpoint ------
    ckpt_dir = run_dir / "checkpoints" / "final"
    ckpt_info = save_adapter_checkpoint(policy_model, ckpt_dir, lora_cfg, lora_info)
    print(f"Checkpoint saved: {ckpt_info['adapter_file']} ({ckpt_info['adapter_file_size']:,} bytes)")

    # ------ Final summary ------
    if step_metrics_list:
        first = step_metrics_list[0]
        last = step_metrics_list[-1]
        print(f"\n=== GRPO Training Summary ===")
        print(f"  Loss: {first['loss']:.4f} -> {last['loss']:.4f}")
        print(f"  Reward: {first['reward_mean']:.3f} -> {last['reward_mean']:.3f}")
        print(f"  KL: {first['KL']:.4f} -> {last['KL']:.4f}")
        print(f"  Invalid output rate: {first['invalid_output_rate']:.2%} -> {last['invalid_output_rate']:.2%}")
        zero_var_count = sum(m["zero_variance_group_count"] for m in step_metrics_list)
        print(f"  Zero-variance groups: {zero_var_count}/{len(step_metrics_list)}")

    # ------ Check for NaN/Inf ------
    losses = [m["loss"] for m in step_metrics_list]
    if any(math.isnan(x) or math.isinf(x) for x in losses):
        all_warnings.append("CRITICAL: NaN or Inf in GRPO loss")
        print("\nWARNING: NaN or Inf detected in GRPO loss!")

    if all_warnings:
        diag_path = run_dir / "logs" / "diagnosis.md"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("w") as f:
            f.write("# GRPO Training Diagnosis\n\n")
            for w in all_warnings:
                f.write(f"- {w}\n")

    return "completed"


# ---------------------------------------------------------------------------
# Failure handler
# ---------------------------------------------------------------------------

def _handle_failure(run_dir: Path, exc: Exception, runs_dir: str):
    run_id = run_dir.name
    print(f"\n{'='*60}")
    print(f"GRPO TRAINING FAILED: {exc}")
    print(f"{'='*60}")

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    error_path = logs_dir / "error.txt"
    with error_path.open("w", encoding="utf-8") as f:
        f.write(f"Error: {exc}\n\n")
        f.write(traceback.format_exc())

    try:
        from polaris.registry import update_run_status
        update_run_status(run_id, "failed", runs_dir)
    except Exception as reg_err:
        print(f"  [warn] could not update run status: {reg_err}")

    print(f"Error written: {error_path}")
    print(f"Run {run_id} marked as failed.")


if __name__ == "__main__":
    raise SystemExit(main())
