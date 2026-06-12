# M1 SFT Current Diagnosis

日期：2026-06-12

当前有效结论见 `docs/feasibility_reports/d4_sft_confirmed_flow.md`。

保留证据：

- D3 baseline 30 条：`runs/d3_baseline_30_format_check`
- D4 confirmed sanity：`runs/000022_qwen3_0_6b_sft_math`

已清理内容：

- `runs/000006_qwen3_0_6b_sft_math` 至 `runs/000021_qwen3_0_6b_sft_math`
- 项目级 `.DS_Store`
- 项目级 `__pycache__`
- 旧 `gsm8k_smoke_result.json`
- `scripts/models/qwen3_0_6b/metadata/.cache`

保留原因：

`runs/000022_qwen3_0_6b_sft_math` 是当前 D4 格式口径确认 run，包含 generation diagnostics、debug batch、train metrics、eval metrics、sample diff 和 adapter-only checkpoint。

`runs/d3_baseline_30_format_check` 是当前 baseline 格式口径复核证据，确认 baseline `format_adherence` 可低，同时 `answer_extraction_success` 与 `pass_at_1` 正常记录。
