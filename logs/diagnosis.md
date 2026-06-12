# M1 格式评估口径统一报告

生成时间：2026-06-12

## 修改文件

`scripts/train_sft.py`：统一 SFT 训练、eval generation、sample_diff generation、固定 prompt diagnostics 的 prompt 协议。

`scripts/smoke/load_model.py`：统一 baseline inference 的 chat template 渲染逻辑。

`scripts/smoke/eval_gsm8k.py`：baseline 输出和保存结果增加 `pass_at_1`、`answer_extraction_success`、`format_adherence` 字段。

`scripts/eval_math.py`：统一 M1 `format_adherence` 口径。

`logs/diagnosis.md`：记录 D3 baseline 30 条和 D4 sanity eval 的验证结果。

## Prompt 协议

统一使用 system/user chat template，调用 `apply_chat_template(..., tokenize=False, add_generation_prompt=False)`，然后手动追加 `ASSISTANT_HEADER = "<|im_start|>assistant\n"`。

模型生成的 assistant continuation 必须自己生成 `<think>`、`</think>` 和 `\boxed{answer}`。训练 tokenization 中 `prompt_text` 不包含 `<think>`，`target` 从 `<think>` 开始。

## 格式评估口径

`eval_math.has_m1_format_adherence` 是唯一格式检查函数。它要求 generated continuation 同时满足：存在 `<think>`，存在 `</think>`，`</think>` 后存在 `\boxed{...}`。

`extract_predicted_answer` 保留 `numeric_fallback`，所以 `answer_extraction_success` 可以因裸数字抽取成功而置 1。`format_adherence` 不能因 `numeric_fallback` 成功而置 1。

## 修复点

`scripts/smoke/load_model.py` 删除 `enable_thinking=False` 路径，避免 tokenizer 自动插入空 `<think></think>`。

`scripts/train_sft.py` 的 `tokenize_sample` 改为使用 `build_m1_generation_prompt_ids`，不再从 target 中剥离开头 `<think>`。`run_eval_protocol`、`run_sample_diff`、`run_generation_smoke` 使用同一 M1 prompt 口径。

`scripts/train_sft.py` 的 eval 与 sample_diff format 计算改为调用 `eval_math.has_m1_format_adherence`。

`scripts/eval_math.py` 的 `compute_metrics` 增加 `pass_at_1`、`answer_extraction_success`、`format_adherence` 聚合字段。

## 验证结果

### D3 baseline 30 条

命令输出目录：`runs/d3_baseline_30_format_check/`。

结果：`total=30`，`pass_at_1=23.33%`，`answer_extraction_success=96.67%`，`format_adherence=0.00%`。

解释：baseline 主要通过 `numeric_fallback` 抽取答案，30 条中 29 条抽取成功、7 条正确。格式遵循为 0 是可接受结果，因为未训练模型没有稳定输出 M1 target 协议。

### D4 SFT sanity eval

run：`runs/000022_qwen3_0_6b_sft_math`。

固定 prompt 训练后输出和 adapter reload 输出一致：`<think>\n2 + 3 = 5\n</think>\n\n\boxed{5}`。

`debug_batch.json`：`prompt_text` 不包含 `<think>`，末尾为 `<|im_start|>assistant\n`；`target` 从 `<think>` 开始；`shift_check_summary` 为 `uses_next_token_labels=true`、`self_token_prediction=false`。

`train_metrics.jsonl`：10 step loss 从 `1.108333` 到 `0.715854`，grad_norm 从 `3.035351` 到 `0.578787`，无 NaN、Inf 或异常飙升。

`eval_metrics.jsonl`：`pass_at_1=0.3`，`answer_extraction_success=1.0`，`format_adherence=1.0`，`invalid_output_rate=0.0`。

`sample_diff.jsonl`：5 条样例 `warning=null`。首条输出包含完整结构：`<think>...\n</think>\n\n\boxed{285}`，`format_adherence=1`，`extraction_success=1`。

`checkpoints/final`：只包含 `adapters.safetensors` 和 `adapter_config.json`，adapter 文件大小为 `36724832` bytes。

## 结论

M1 格式评估口径已统一。baseline format_adherence 可以低，且不会影响 `answer_extraction_success` 和 `pass_at_1` 正常记录。D4 sanity eval 的 `format_adherence` 不再因 prompt/template 前缀错配全 0，当前结果为 1.0。

当前链路满足进入完整 GSM8K SFT 的格式评估前置条件。

## 残余风险

D4 sanity 只覆盖 10 step 和 10 条 eval 样本。完整 SFT 仍需持续监控 `format_adherence`、`answer_extraction_success`、`pass_at_1`、`invalid_output_rate`、`train_loss`、`grad_norm`。

如果后续切换 tokenizer、chat template 或模型族，必须重新验证 `render_m1_generation_prompt` 生成的 prompt 末尾仍为 `<|im_start|>assistant\n`，且不注入 `<think>`。
