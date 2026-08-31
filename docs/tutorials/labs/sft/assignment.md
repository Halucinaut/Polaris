# SFT 作业

目标：能独立审查一次 SFT 从数据、token 边界、训练到评估的证据链。历史材料以 `docs/tutorials/sft_to_dpo_review.md` 的 S1–S6 为主；本作业要求把结论落到项目文件。

## Gate S0：数据与目标，训练前完成

1. 读取 `data/math/gsm8k/split/test_converted_500.jsonl` 的两条记录，说明 `problem_id`、`problem`、`answer`、`solution` 分别服务什么环节；解释为什么评测要使用 `answer`，不能直接把 `solution` 当作答案。
2. 检查 SFT 训练数据的一条记录。写出 prompt、assistant target 和监督 token 的边界；指出把 prompt token 一并计入 loss 会造成什么问题。
3. 用 `runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final/` 说明 LoRA checkpoint 保存了什么、没有保存什么，以及重新加载需要的条件。

提交：完成 S1–S3 后填 `submission.md`，再查看 `gate.md` 的 S0 门禁。

## Gate S1：sanity，10-step 后完成

4. 找到 SFT sanity 的首末两个训练指标。判断 loss 变化是否足以证明泛化；给出一个仍可能存在的失败模式。
5. 检查一次生成样本：回答最终答案提取与格式遵从是否是同一个指标，并给出一条能区分二者的输出示例。

提交：在继续完整训练前完成 S4–S5。

## Gate S2：完整训练与评估后完成

6. 从 `runs/000030_qwen3_0_6b_sft_gsm8k_500`、`runs/baseline_50_eval/` 和 `runs/sft_50_eval/` 复核 50 条统一评测的 baseline、SFT 答案正确率和格式遵从率。写清样本数与口径。
7. 人工审阅至少 10 条 baseline/SFT 输出，记录两条真实改善和两条仍失败的样本；区分“答案错”“答案无法提取”“格式错”。

提交：S6–S7 完成后，SFT 阶段作业可归档。
