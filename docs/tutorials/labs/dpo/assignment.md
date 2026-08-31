# DPO 作业：从 preference data 到实验结论

目标：亲自建立并审查 DPO v2 的完整证据链。当前数据目标是“答案正确性相同条件下的格式偏好”，不能把它解释为数学正确性偏好。

## Gate D0：偏好数据与评估器，训练前完成

证据文件：`data/math/splits/dpo_v2_style_train_449.jsonl`、`data/math/splits/dpo_v2_style_stress_50.jsonl`、`data/math/quarantine/dpo_v2_style_invalid_1.jsonl`、`data/math/reports/dpo_v2_style_split_report.json`、`scripts/split_dpo_v2_style.py`、`scripts/eval_style_dpo.py`。

1. 随机抽取 3 条训练 pair。逐条判断 chosen/rejected 的答案是否一致、偏好信号来自哪里、chosen 是否存在不必要的冗长步骤。
2. 解释 449/50/1 的划分规则。说明 50 条 stress 为什么不能作为独立数学泛化测试，隔离的 1 条为什么绝不能混入训练。
3. 以一条格式正确但答案错误的输出，以及一条答案正确但旧 GSM 格式的输出，说明 `answer_correct`、`answer_extractable`、`style_adherent`、`correct_and_adherent` 的关系。
4. 解释 evaluator 为什么必须拒绝 ID 不匹配和 `Final: The answer is \boxed{42}}.`；说明这些问题若被静默接受会如何污染实验结论。

提交时点：**运行 10-step sanity 前必须完成 D1–D4。**

## Gate D1：配置与训练信号，10-step 前完成

证据文件：`configs/qwen3_0_6b/dpo_v2_style.yaml`、`scripts/train_dpo.py`、`runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final/`。

5. 从配置中定位 policy、reference、训练数据和 beta。说明为什么 policy 与 reference 在第 0 步必须加载同一份 M1 adapter，reference 为什么冻结。
6. 手算本轮完整训练的更新次数：449 条、batch size 2、gradient accumulation 4、1 epoch。写出 micro-batch 数、optimizer update 数，以及最后一次 update 使用多少样本。
7. 对单条 pair 写出 DPO 的四个 response logprob、policy margin、reference margin、DPO margin 和 loss。解释初始 policy/reference 完全相同时，DPO margin 和 loss 应接近什么。

提交时点：**D5–D7 通过后，才允许启动带 `--debug-batch` 的 10-step sanity。**

## Gate D2：10-step sanity，完整训练前完成

证据文件：本次 sanity run 的 `config.yaml`、`logs/checkpoint_provenance.json`、`logs/debug_dpo_batch.json`、`metrics/train_metrics.jsonl`。

8. 审查 frozen config：数据是否为 449 条 v2 train，policy/reference adapter 是否一致，`reference_frozen` 是否为 true。
9. 审查 debug batch：写出 prompt token、chosen/rejected response token、截断状态和首个样本 ID；判断 response mask 是否排除了 prompt。
10. 比较首末 step 的 loss、DPO margin、reference margin、gradient norm。写出“链路通过”的最小证据，以及仍不能证明的效果结论。

提交时点：**完整一轮训练前必须完成 D8–D10。**

## Gate D3：完整训练、评估与盲审，训练后完成

11. 复核完整 run 是否完成 57 次更新，是否存在 NaN/Inf，最终 checkpoint 是否存在。解释 57 与 D6 的计算如何相互验证。
12. 对同一生成参数评估 SFT 与 DPO v2：GSM8K-50 报告数学正确率；stress-50 同时报告答案正确率、格式合规率和两者同时成立的比例。解释两组测试各自能证明什么。
13. 进行至少 20 条 SFT/DPO v2 盲审。先隐藏模型名完成标注，再揭示来源；每条标记答案、格式、步骤清晰度、冗长程度和异常类型。
14. 写一段不超过 250 字的实验结论：只陈述现有证据支持的结论，明确至少一个未解决的混杂因素或下一步实验。

提交时点：**D11–D14 是 M2 DPO v2 效果验收的必要材料。完成前不进入 GRPO。**
