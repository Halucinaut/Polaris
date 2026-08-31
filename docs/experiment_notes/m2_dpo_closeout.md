# M2 DPO Closeout

**日期**：2026-08-31  
**阶段状态**：工程与诊断通过；效果目标未通过；停止新的 DPO 效果实验，进入 M2.5 RL sanity 准备。  
**适用范围**：Qwen3-0.6B、M1 SFT 初始化、Math/GSM8K、当前离线 chosen/rejected 数据、temperature=0 的风格评测协议。

## 决策

M2 已完成其工程学习目标：pair 构造、policy/reference 初始化、response-only logprob、DPO loss、训练日志、风格评估、分支点诊断和 token 归因均已有可复核产物。

暂停 DPO 的原因是效果结论已稳定：当前离线 DPO 能改变已给定 chosen 轨迹上的相对概率，却未将 `Solution:` 推到自由生成分支点的 greedy 首选。继续更换风格数据或做常规超参数扫描缺少新的、可证伪的假设。

该结论的外推范围限于本项目已验证的设置，不能用于判断 DPO 在其他模型、任务或偏好轴上的能力。后续若重启 DPO，实验设计必须预先说明如何改变自由生成分支点指标，并在训练前注册该预测。

## 已验证的训练与评估证据

### DPO v1：训练链路

`runs/000033_qwen3_0_6b_dpo_math` 完成 10-step sanity，`runs/000034_qwen3_0_6b_dpo_math` 完成 329 条 pair 的完整训练。policy 与冻结 reference 均来自 M1 adapter；训练集 DPO margin 从接近 0 上升，末步 preference accuracy 为 1.0，数值稳定。

同口径 GSM8K-50 评估中，M1 SFT 与 DPO v1 均为 14/50 pass@1；DPO v1 的格式遵从为 48/50，低于 M1 的 50/50。该 run 支持训练链路正确和训练集偏好被优化，不支持数学能力提升。完整记录见 [m2_math_dpo.md](m2_math_dpo.md)。

### 风格迁移：DPO 与 SFT 对照

风格实验的偏好轴固定为：chosen 与 rejected 的数学答案均正确，chosen 使用 `Solution:`、连续编号步骤与 `Final: ... \boxed{...}` 模板，rejected 为自由格式或移除 style wrapper 的版本。训练数据、评测协议和 split 说明见 [m2_dpo_v2_style_plan.md](m2_dpo_v2_style_plan.md)。

| 实验 | 完整 run | 关键变量 | Probe-30 风格合规 | Stress-50 风格合规 | 结论 |
|---|---|---|---:|---:|---|
| SFT style control | `000039` | 以 chosen 为监督目标 | 30/30 | 49/50 | 目标模板可以被当前模型和数据学到 |
| DPO v2 | `000036` | 449 条 style pairs，lr=5e-7 | 未保留同口径 Probe 汇总 | 0/50 | 风格迁移失败 |
| DPO v3 | `000041` | 仅将 lr 提至 5e-6 | 0/30 | 0/50 | 风格迁移失败，Stress 正确率 33/50 |
| DPO v4 minimal | `000043` | chosen/rejected 仅保留 style 差异 | 0/30 | 0/50 | 风格迁移失败 |
| DPO v4 minimal 4 epoch | `000051` | v4 数据训练 4 epoch，228 updates | 0/30 | 0/50 | 增加训练仍未改变自由生成风格 |

SFT control 的作用限于可行性验证：两者目标函数、学习率和 token 监督方式不同，不能据此比较方法效果。它回答“同一 chosen 风格是否可被该模型学习”，答案为可以。

### 分支点与归因诊断

`reports/binary_prefix_ctrl_diagnosis.json` 对 30 条 Probe 样本直接检查 `<think>\n` 后目标前缀的 greedy 进入情况：M1、DPO v2、DPO v3、DPO v4 均为 0/30；SFT control 为 30/30。DPO v3 的目标首 token 平均 rank 虽从 M1 的 14274 降至 2398，仍远离 rank-1；DPO v4 为 9184。

`reports/attribution_000051_summary.json` 使用修正后的 `logits[t-1]` 对 449 对 v4 数据做全序列归因：平均 exact margin 为 50.71，`Solution` token 的平均 policy-reference shift 为 +2.12 nats，但其平均 rank 为 6913，中位 rank 为 4335，平均只贡献总 margin 的 3.9%。`Solution` 在 teacher-forced chosen 序列中获得的概率提升，不能推出它在 `prompt + <think>\n` 条件下成为生成起点。

此前存在 off-by-one 的历史归因文件，保留为 `reports/*_invalid_off_by_one.json`；不得引用其中的 rank 或 margin share 作为结论。

## 解释边界

DPO loss 对完整 chosen/rejected 序列的相对对数概率施加约束。当前实验表明，优化增益可以主要由分歧点之后的 token 累积；这种增益足以降低 loss，却不足以跨越分支点的 greedy 排名。SFT 直接在 chosen 轨迹的每个目标 token 上监督，因此能将相同模板写入生成起点。

这是一项基于当前产物的解释，模型规模、初始化、pair 分布、解码方式、beta 与目标行为均可能改变结果。

## 已冻结资产与未完成资产

保留并只读对待以下资产：`dpo_v1.jsonl`、v2/v3/v4 style 数据、所有 DPO/SFT control run、`reports/attribution_000051_*.json`、`reports/binary_prefix_ctrl_diagnosis.json`、`scripts/train_dpo.py`、评估器与对应测试。它们是后续研究复盘和 GRPO 诊断的基线。

reasoning-quality 数据构造实验已有候选资产：v2 校准得到 31 对，v3a/v3b 合计 47 对主训练 pairs；v3b 的独立 DeepSeek 复审仍未完成。该批数据不进入新的 DPO run，也不作为 M2 效果通过证据。相关记录见 [m2_rq_v2_calibration.md](m2_rq_v2_calibration.md)、[m2_rq_v3_pilot.md](m2_rq_v3_pilot.md) 与 [m2_rq_v3b_pilot.md](m2_rq_v3b_pilot.md)。

## 向 M2.5 的交接

M2 的前置能力已具备：policy/reference 来源可追踪、response mask 与 logprob 可审计、pair 和输出可检查、失败原因已定位。M2.5 的工作范围是实现真实 rollout、答案 reward、group-relative advantage、reference KL、checkpoint、resume 与日志链路；它不以提升 benchmark 为目标。

进入正式 GRPO 前，必须完成一次真实的小样本 M2.5 sanity，并验证每个 group 存在可用的 reward 方差。全对或全错 group 的 advantage 退化必须被记录和处理。
