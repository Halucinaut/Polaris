# M2 DPO：run 000034 训练与 50 条评估

## 结论

M2 的训练链路已完成：policy 和冻结 reference 都从 M1 SFT adapter 初始化，329 条 pair 完整遍历一次，训练稳定且无 NaN/Inf。当前单轮 DPO 没有提升前 50 条 GSM8K 的 pass\@1：SFT 与 DPO 均为 14/50；DPO 的格式遵从从 50/50 降至 48/50。因此该 run 通过工程验收，尚未通过效果验收。

## 训练设置与来源

| 项目 | 值 |
| --- | --- |
| run | `runs/000034_qwen3_0_6b_dpo_math` |
| base model | `models/qwen3_0_6b/mlx` |
| policy adapter | `runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final/adapters.safetensors` |
| reference | 相同 base model 与 M1 adapter，加载后冻结 |
| pair 数据 | `data/math/splits/dpo_v1.jsonl`，329 条 |
| batch / 累积 / epoch | 2 / 4 / 1；42 个 optimizer updates，最后一步 1 条 pair |
| 学习率 / beta | `5e-7` / `0.1` |
| response logprob | response token logprob 求和 |

权重来源和冻结状态见 `runs/000034_qwen3_0_6b_dpo_math/logs/checkpoint_provenance.json`，完整配置见 `runs/000034_qwen3_0_6b_dpo_math/config.yaml`。

## 训练证据

10-step sanity 为 `runs/000033_qwen3_0_6b_dpo_math`：初始 debug dump 的 policy margin 与 reference margin 相同，DPO margin 为 0；第 10 步 loss 为 0.397001、DPO margin 为 9.53125。

正式 run `000034` 覆盖 329 条数据各一次。第 1 步 loss 为 0.693147，第 42 步为 0.028146；末步 DPO margin 为 30.8125、preference accuracy 为 1.0。该趋势证明训练集中 policy 相对 reference 更偏向 chosen；它不等价于测试集能力提升。

## 同口径评估

评估使用 `scripts/smoke/eval_gsm8k.py`：`data/math/gsm8k/split/test_converted_500.jsonl` 前 50 条、temperature=0、max_new_tokens=512。baseline、SFT 与 DPO 使用同一问题顺序和答案提取器。

| 模型 | pass\@1 | 可提取答案 | 格式遵从 | 平均输出字符数 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 10/50 | 49/50 | 3/50 | 1678.2 |
| M1 SFT | 14/50 | 50/50 | 50/50 | 295.1 |
| M2 DPO | 14/50 | 50/50 | 48/50 | 298.6 |

DPO 相对 SFT 修复了 `gsm8k_test_0025`、`0033`、`0042`，退化了 `0001`、`0015`、`0023`，净变化为零。格式退化发生在 `0019` 与 `0039`。需要人工对这 8 条样本审阅推理、长度和最终格式，再决定是否调整 pair 筛选或训练设置。

评估原始结果位于 `runs/dpo_50_eval/results.json` 与 `runs/dpo_50_eval/test_predictions.jsonl`。这些文件未纳入 Git；复跑命令如下：

```bash
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path runs/000034_qwen3_0_6b_dpo_math/checkpoints/final \
  --test-data data/math/gsm8k/split/test_converted_500.jsonl \
  --review-data data/math/gsm8k/split/does_not_exist.jsonl \
  --limit 50 --max-new-tokens 512 --temperature 0 \
  --output-path runs/dpo_50_eval/results.json
```

## 下一次实验前的检查

先完成 D1 的人工审阅，并优先核对 71 条 `length_biased` pair 是否推动模型学习长度和格式捷径。随后增加 response-only logprob、next-token shift 和 DPO loss 的数值回归测试；当前已有加载、adapter 路径和 epoch 覆盖测试。只有确认 pair 质量后，再尝试一个可解释的单变量改动，例如过滤或分层抽样 `length_biased` pair。不要以当前 50 条持平的结果直接进入 GRPO。
