# Qwen3-0.6B Baseline Eval Check

## Checked run
- run_dir: `runs/000005_qwen3_0_6b_gsm8k_baseline`
- status: completed
- created_at: 2026-06-10T10:25:14+00:00

## Test metric
- num_samples: 30
- pass_at_1: 0.6
- answer_extraction_success: 1.0
- format_adherence: null (原始 smoke 数据未记录，见 Remaining issues)
- invalid_output_rate: 0.0
- avg_completion_length: 582.43

## Review metric
- num_samples: 30
- pass_at_1: 0.6
- answer_extraction_success: 1.0
- format_adherence: null (原始 smoke 数据未记录)
- invalid_output_rate: 0.0
- avg_completion_length: 533.53
- 注：review 指标单独存放于 `samples/review_eval_summary.json`，未与 test 合并。

## Storage check
| 文件 | 状态 |
|------|------|
| config.yaml | PASS |
| run_meta.yaml | PASS |
| metrics/eval_metrics.jsonl | PASS |
| metrics/hardware_log.jsonl | PASS (memory_total_gb 为 null，见 Remaining issues) |
| samples/baseline_predictions.jsonl | PASS |
| samples/eval_results.jsonl | PASS |
| samples/eval_summary.json | PASS |
| samples/review_predictions.jsonl | PASS |
| samples/review_eval_results.jsonl | PASS |
| samples/review_eval_summary.json | PASS |

## eval_math.py 复用检查
- scripts/smoke/eval_gsm8k.py 已通过 `eval_math.evaluate()` 和 `eval_math.compute_metrics()` 进行评估，不再直接手写 pass_at_1 统计。
- inference 结束后会输出 `{split}_predictions.jsonl`，可被 `scripts/eval_math.py` 独立读取复评。

## Manual bad case conclusion
- Checked bad cases manually.
- Dominant failure type: calculation_error.
- Extraction error observed: no.

## Remaining issues
1. **format_adherence 为 null**：本次 baseline 结果是从早期 smoke test 的 `gsm8k_smoke_result.json` 迁移而来，原始数据未记录 format_adherence。eval_gsm8k.py 已修改为调用 eval_math.evaluate()，若重新运行可自动计算该字段。
2. **hardware_log.jsonl 缺少 memory_total_gb**：环境未安装 psutil，snapshot_hardware() 返回 null。建议安装 psutil 后补采一条 hardware snapshot。
3. **config.yaml 中 data.split 为 train**：这是 base config 的默认值，对 baseline run 无实质影响，但建议未来 baseline 专用 override 中显式标注 split。

## Conclusion
D3 KR4 status: CONDITIONAL PASS — 结构、metrics、samples、eval_math 复用均符合 M1 要求；仅 format_adherence 和 memory_total_gb 需补充。
