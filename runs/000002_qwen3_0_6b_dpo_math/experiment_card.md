# Experiment Card: 000002_qwen3_0_6b_dpo_math

## Basic Info

| Field | Value |
|---|---|
| Run ID | 000002_qwen3_0_6b_dpo_math |
| Run Name | qwen3_0_6b_dpo_math |
| Status | running |
| Created At | 2026-06-06T12:32:06.606655+00:00 |
| Updated At | 2026-06-06T12:32:06.607204+00:00 |
| Method | dpo |
| Domain | openr1 |
| Model | Qwen3-0.6B |
| Model Size | 0.6b |
| Config Path | runs/000002_qwen3_0_6b_dpo_math/config.yaml |

## Goal

本次实验要验证什么。

## Setup

| Field | Value |
|---|---|
| Dataset | openr1_math_dpo_pairs |
| Split | train |
| LoRA | r=32, alpha=32, dropout=0.0 |
| Sequence Length | 2048 |
| Batch Size | 2 |
| Precision | bfloat16 |
| Hardware | mlx |

## Expected Signals

本次实验预期观察哪些指标变化。

## Exit Criteria

满足哪些条件视为通过。

## Notes

