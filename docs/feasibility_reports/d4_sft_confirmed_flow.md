# D4/D5 SFT Confirmed Flow

日期：2026-06-12

## 固定配置

D4 sanity 使用 `configs/qwen3_0_6b/sft_math.yaml`。

D5 500 条 GSM8K SFT 使用 `configs/qwen3_0_6b/sft_gsm8k_500.yaml`。

关键训练参数：

- `model.path`: `models/qwen3_0_6b/mlx`
- D4 `data.path`: `data/math/splits/sft_v1.jsonl`
- D5 `data.path`: `data/math/splits/sft_d5_500.jsonl`
- D5 `training.num_epochs`: `3`
- `training.batch_size`: `4`
- `training.gradient_accumulation_steps`: `4`
- `training.learning_rate`: `5.0e-5`
- `training.max_seq_length`: `2048`
- `lora.r`: `32`
- `lora.alpha`: `32`
- `lora.dropout`: `0.0`
- `lora.target_modules`: `q_proj,k_proj,v_proj,o_proj`

## Prompt 协议

训练、eval generation、sample diff、generation diagnostics 统一使用 M1 continuation 协议。

构造方式：先渲染 system/user chat template，再手动追加 `<|im_start|>assistant\n`。prompt 内不注入 `<think>`，模型必须自己生成：

```text
<think>
...
</think>

\boxed{answer}
```

`format_adherence` 只检查 generated continuation：必须有 `<think>`、必须有 `</think>`、`</think>` 后必须有 `\boxed{...}`。`answer_extraction_success` 保留 numeric fallback，但 numeric fallback 不计入 `format_adherence`。

## Checkpoint 协议

SFT checkpoint 只保存 adapter：

- `checkpoints/final/adapters.safetensors`
- `checkpoints/final/adapter_config.json`

reload 验证必须从 base model 重新加载，注入同样 LoRA 结构，再加载 `adapters.safetensors`。

## 训练互斥锁

`scripts/train_sft.py` 启动时创建 `runs/.train_sft.lock`。已有锁时直接失败，避免多个 MLX SFT 进程并发占用统一内存。进程正常退出时通过 `atexit` 删除锁。

## D4 Sanity 命令

```bash
UV_CACHE_DIR=/private/tmp/polaris-uv-cache uv run python scripts/train_sft.py \
  --config configs/qwen3_0_6b/sft_math.yaml \
  --max-steps 10 \
  --debug-dump-batch
```

确认 run：`runs/000022_qwen3_0_6b_sft_math`。

关键结果：

- `train_loss`: `1.108333 -> 0.715854`
- `grad_norm`: `3.035351 -> 0.578787`
- `pass_at_1`: `0.3`
- `answer_extraction_success`: `1.0`
- `format_adherence`: `1.0`
- `invalid_output_rate`: `0.0`
- fixed prompt after training: `<think>\n2 + 3 = 5\n</think>\n\n\boxed{5}`

## D3 Baseline 复核命令

```bash
UV_CACHE_DIR=/private/tmp/polaris-uv-cache uv run python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --test-data data/math/gsm8k/split/test_converted.jsonl \
  --review-data /private/tmp/polaris-no-review.jsonl \
  --limit 30 \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --output-path runs/d3_baseline_30_format_check/results.json
```

结果：`pass_at_1=23.33%`、`answer_extraction_success=96.67%`、`format_adherence=0.00%`。baseline format 低是可接受现象。

## D5 500 条 GSM8K 命令

数据构造：

```bash
python scripts/data/gsm8k/build_train_converted_slice.py \
  --limit 500 \
  --seed 42 \
  --output data/math/gsm8k/split/train_converted_d5_500.jsonl

python scripts/prepare_math_data.py \
  --input-dir data/math/gsm8k/split \
  --input-files train_converted_d5_500.jsonl \
  --output data/math/splits/sft_d5_500.jsonl \
  --report data/math/reports/sft_d5_500_data_report.json
```

训练：

```bash
UV_CACHE_DIR=/private/tmp/polaris-uv-cache uv run python scripts/train_sft.py \
  --config configs/qwen3_0_6b/sft_gsm8k_500.yaml \
  --debug-dump-batch
```

确认 run：`runs/000030_qwen3_0_6b_sft_gsm8k_500`。

关键结果：

- 训练步数：`375`
- `train_loss`: `1.108333 -> 0.032215`
- 最大 `grad_norm`: `3.211502`
- `learning_rate`: `5.0e-5`
- `trainable_param_count`: `9175040`
- `pass_at_1`: `0.8`
- `answer_extraction_success`: `1.0`
- `format_adherence`: `0.9`
- `invalid_output_rate`: `0.0`
- after training: `<think>\n2 + 3 is 5.\n</think>\n\n\boxed{5}`
- after reload: `<think>\n2 + 3 is 5.\n</think>\n\n\boxed{5}`

完整训练期间必须持续检查：

- `logs/generation_diagnostics.json`
- `logs/debug_batch.json`
- `metrics/train_metrics.jsonl`
- `metrics/eval_metrics.jsonl`
- `metrics/eval_summary.json`
- `samples/sample_diff.jsonl`

准入阈值：`invalid_output_rate` 必须保持为低值；`train_loss` 和 `grad_norm` 不能出现 NaN、Inf 或持续异常飙升；`after_training_in_memory_output` 与 `after_reloading_checkpoint_output` 不能出现结构性分叉。
