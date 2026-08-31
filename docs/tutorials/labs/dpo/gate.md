# DPO 门禁与批改

当前状态：M2 已收口。DPO v1 证明训练链路；v2–v4 的 style experiments、SFT 对照与分支点归因支持“工程与诊断通过，风格迁移效果未通过”。不再启动新的 DPO 效果 run，下一阶段为 M2.5 RL sanity。正式记录见 `docs/experiment_notes/m2_dpo_closeout.md`。

| Gate | 必交题目 | 允许发生的实验动作 | 通过标准 | 状态 | 批改记录 |
|---|---|---|---|---|---|
| D0 | D1–D4 | 数据与评估器审计 | 偏好轴、split 与 evaluator 的边界明确 | 已完成 | v2 style 数据含 449/50/1 split；偏好轴为正确性固定下的风格差异 |
| D1 | D5–D7 | 10-step sanity | 初始化、更新次数、DPO 数学关系正确 | 已完成 | `000033` 与后续 style sanity 记录 policy/reference、mask、margin 与 provenance |
| D2 | D8–D10 | 完整 DPO run | provenance、token 边界、数值稳定性有证据 | 已完成 | `000034`、`000036`、`000041`、`000043`、`000051` 已保留为可复核 run |
| D3 | D11–D14 | 阶段收口与 M2.5 决策 | 统一评估、样例审阅与失败归因形成结论 | 已收口：工程通过，效果未通过 | 见 `m2_dpo_closeout.md`；停止新的 DPO 效果实验 |

## 已固定的解释边界

- 训练集为 449 条，stress 为 50 条，quarantine 为 1 条。
- stress 是风格压力集，不是独立泛化集。
- DPO v2 的 pair 双方答案目标相同；loss 或 margin 下降不构成数学能力提升证据。
- 任何未通过门禁的 run 保留为诊断材料，不作为阶段效果结论。
- `000051` 的 `Solution` 分支点平均 rank 为 6913；teacher-forced 的 token 概率提升不能作为自由生成风格迁移证据。
