# GRPO 门禁与批改

当前状态：未开始。进入前置条件是 DPO 的 D3 通过，并完成 M2.5 的 rollout、reward、logprob、KL、checkpoint、resume 链路验证。

| Gate | 必交题目 | 允许发生的实验动作 | 通过标准 | 状态 | 批改记录 |
|---|---|---|---|---|---|
| G0 | G1–G4 | 设计 reward、读取代码 | reward 协议可复算，风险已写明 | 未开始 | |
| G1 | G5–G7 | 真实 GRPO 小规模 run | 链路失败模式可诊断 | 未开始 | |
| G2 | G8–G10 | 阶段结论与下一轮设计 | 区分 reward gain、长度变化和能力变化 | 未开始 | |

GRPO 的核心门槛是在线数据分布与 reward 的可信度；离线 loss 曲线不能替代这一审查。
