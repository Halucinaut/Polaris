# Experiment Card: {{ run_id }}

## Basic Info

| Field | Value |
|---|---|
| Run ID | {{ run_id }} |
| Run Name | {{ run_name }} |
| Status | {{ status }} |
| Created At | {{ created_at }} |
| Updated At | {{ updated_at }} |
| Method | {{ method }} |
| Domain | {{ domain }} |
| Model | {{ model_name }} |
| Model Size | {{ model_size }} |
| Config Path | {{ config_path }} |

## Goal

本次实验要验证什么。

## Setup

| Field | Value |
|---|---|
| Dataset | {{ dataset_name }} |
| Split | {{ dataset_split }} |
| LoRA | r={{ lora_r }}, alpha={{ lora_alpha }}, dropout={{ lora_dropout }} |
| Sequence Length | {{ max_seq_length }} |
| Batch Size | {{ batch_size }} |
| Precision | {{ dtype }} |
| Hardware | {{ hardware_device }} |

## Expected Signals

本次实验预期观察哪些指标变化。

## Exit Criteria

满足哪些条件视为通过。

## Notes

