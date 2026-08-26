# Polaris

本地优先的 LLM Post-Training 能力训练项目。

## 项目定位

Polaris 在 MacBook Pro M5 Max（128 GB 统一内存）上建立可复现、可诊断的 Post-Training 实验闭环。项目目标是系统掌握 SFT、DPO、GRPO、PPO、OPD 的训练信号、数据构造、评估设计、失败诊断、实验记录和本地硬件边界。

项目优先形成经过验证的训练链路、可复现实验、数据构造经验、失败分析、MLX 适配记录和 M5 Max 本地训练能力边界；新算法研究和 benchmark 排名不进入当前主路径。

## 任务路线

项目采用两条任务线：

- **Math reasoning 主线**：使用自动判分、reward 噪声较低的数学任务，依次跑通 SFT、DPO、GRPO、PPO。
- **Tool-call 副线**：在 Math 核心闭环稳定后，验证结构化输出、工具调用、执行成功率、reward 设计和垂类迁移能力。

Qwen3-0.6B 是完整学习模型；Qwen3-4B 用于选择性验证；Qwen3-8B 只用于本地能力边界检查。执行顺序固定为 M0 → M1 → M2 → M2.5 → M3，先完成 0.6B Math 的 SFT、DPO、GRPO 核心闭环。

## 当前进度

当前处于 **M2：Qwen3-0.6B Math DPO**。M0、M1 已通过；M2 已完成数据构造、M1 adapter 初始化修复、10-step sanity 和一轮完整训练，尚未完成同口径评估与人工 sample diff 审阅。

| 里程碑 | 内容 | 状态 | 当前结果 |
|---|---|---|---|
| M0 | 工程骨架与实验协议 | 已完成 | config、run registry、指标与硬件日志、文档模板、fake sanity 闭环已建立 |
| M1 | Qwen3-0.6B Math SFT | 已完成 | GSM8K 500 条完整训练完成；pass@1 由 0.20 提升至 0.28；format adherence 由 0.06 提升至 1.00 |
| M2 | Qwen3-0.6B Math DPO | 进行中 | 329 组 DPO pairs、M1 policy/reference 初始化、10-step sanity 和一轮完整训练已完成；等待评估与人工审阅 |
| M2.5 | RL sanity | 未开始 | 等待 M2 完整验收 |
| M3 | Qwen3-0.6B Math GRPO | 未开始 | 等待 rollout、reward、logprob、KL 链路验证 |
| M4 及以后 | PPO、4B validation、Tool-call、OPD | 未开始 | 不阻塞 0.6B Math 核心闭环 |

### M1 已完成结果

M1 在 Qwen3-0.6B 上跑通了 GSM8K 数据准备、baseline eval、MLX/LoRA SFT、checkpoint、统一 prompt 协议和训练后评估。正式实验为 `runs/000030_qwen3_0_6b_sft_gsm8k_500`：训练 500 条数据、375 steps，train loss 从 1.108333 降至 0.032215；50 条同口径评估中，pass@1 从 0.20 提升至 0.28，answer extraction success 从 0.98 提升至 1.00，format adherence 从 0.06 提升至 1.00。

详细验收记录见 `docs/Project_todo/M1.md`。

### M2 已完成部分

**DPO v1（gold-vs-wrong）**：`data/math/splits/dpo_v1.jsonl`（329 组 pairs），`docs/format_notes/dpo_math_pair_protocol.md`，`scripts/train_dpo.py`。`runs/000033` 完成 10-step sanity，`runs/000034` 完整训练一轮。GSM8K-50 同口径评估：DPO 与 SFT 均为 14/50 pass\@1，格式遵从由 50/50 降为 48/50。完整分析见 `docs/experiment_notes/m2_math_dpo.md`。

**DPO v2 style-controlled（数据与评估器已完成，训练未开始）**：500 条 style-preference pairs 经严格校验后分为 train 449 / stress 50 / quarantine 1。训练配置 `configs/qwen3_0_6b/dpo_v2_style.yaml` 与评估器 `scripts/eval_style_dpo.py` 已就绪。实验计划见 `docs/experiment_notes/m2_dpo_v2_style_plan.md`。

### M2 当前缺口

进入 M2.5 前仍需完成以下工作：

1. 补充 response-only logprob、next-token alignment 和 DPO loss 的数值回归测试。
2. 完成同一测试集、prompt 协议与生成参数下的 baseline、SFT、DPO 比较。
3. 人工审阅至少 20 条 SFT 与 DPO sample diff，检查长度偏差、模板化和模式坍缩。
4. 完成 `docs/experiment_notes/m2_math_dpo.md` 与正式 run report。

M2 的完整计划和通过条件见 `docs/Project_todo/M2.md`。

## 实验原则

所有正式实验必须由 config 驱动，通过 run registry 创建目录，并保存 frozen config。指标、硬件信息和 sample diff 使用 JSONL 记录；失败实验保留错误日志和诊断，不手工修改为 completed。训练、推理和评估必须共享同一套 chat template、system prompt、输出格式和答案抽取口径。

每个阶段坚持小数据、小步验证：先检查数据和 token 边界，再跑 10-step sanity，最后执行完整训练和统一评估。loss 下降只能证明优化器在拟合当前目标，不能单独证明训练正确。

## 仓库结构

```text
polaris/
├── configs/                # base config 与模型、方法覆盖配置
├── polaris/                # 配置、run registry、日志协议
├── scripts/                # 数据准备、训练、评估和 sanity 脚本
├── runs/                   # 实验输出目录，默认不纳入 Git
├── data/                   # 数据源、split、报告和审阅材料
├── models/                 # 本地模型与 metadata
├── tests/                  # 回归测试
├── docs/                   # 里程碑、实验记录、方法卡片和模板
└── future/                 # 暂不进入主路径的可选方向
```

## 常用验证命令

```bash
# 回归测试与配置检查
make test
make config-check

# M0 工程链路
make sanity-all
make list-runs

# M1 结果检查
cat runs/000030_qwen3_0_6b_sft_gsm8k_500/run_meta.yaml
cat runs/baseline_50_eval/eval_summary.json
cat runs/sft_50_eval/eval_summary.json

# M2 数据与训练结果检查
./.venv/bin/python -c 'from pathlib import Path; from polaris.json_records import load_json_record_stream; print(len(load_json_record_stream(Path("data/math/splits/dpo_v1.jsonl"))))'
cat data/math/reports/dpo_v1_report.json
cat runs/000033_qwen3_0_6b_dpo_math/logs/checkpoint_provenance.json
tail runs/000033_qwen3_0_6b_dpo_math/metrics/train_metrics.jsonl
```

`make sanity-all` 会创建新的 fake run。只查看既有实验时使用 `make list-runs`。

## 项目文档

- `docs/tutorials/sft_to_dpo_review.md`：SFT 与 DPO 的原理、项目复盘、数据审核和代码审查练习。
- `docs/tutorials/labs/README.md`：SFT、DPO、GRPO、OPD 的阶段作业、提交和批改门禁。
- `docs/experiment_notes/m2_math_dpo.md`：M2 DPO sanity、正式训练和同口径评估记录。
- `docs/Project_todo/polaris_final_design.md`：项目目标、技术路线和完整里程碑。
- `docs/Project_todo/M0.md`：工程骨架验收结论。
- `docs/Project_todo/M1.md`：Math SFT 完整验收结论。
- `docs/Project_todo/M2.md`：Math DPO 当前计划和通过条件。
- `docs/Project_todo/Project_summary.md`：项目摘要；该文件仍需同步当前里程碑状态。
