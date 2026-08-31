# GRPO 作业

目标：在进入真实 GRPO 前，掌握在线采样、可验证 reward、group-relative advantage、KL 约束和评估混杂。当前项目尚未进入 M3；本作业的训练题必须等 M2 与 M2.5 通过后再做。

## Gate G0：方法与 reward 设计，M2 通过后完成

1. 用一个数学题的 4 个 rollout 构造 reward 表。至少包含答案正确、格式合规和无效输出三种情形；说明每一项 reward 的范围与可被投机的方式。
2. 手算 group-relative advantage：给出 group reward、均值、标准差和每个样本的标准化 advantage。说明 group 内 reward 全相同时应如何处理。
3. 解释 GRPO 相比 DPO 多出的在线环节，以及它带来的 distribution shift、采样成本和 reward hacking 风险。
4. 为 Math 主线写出最小 reward 协议：答案奖励、格式奖励、长度处理、无效输出处理。明确不能把“推理文字更多”直接作为奖励。

提交时点：**实现 rollout 或真实采样前完成 G1–G4。**

## Gate G1：M2.5 链路 sanity，真实 GRPO 前完成

证据将来自 rollout、reward、logprob、KL、checkpoint、resume 的实际日志。

5. 对一组 rollout 人工复算 reward，检查 reward 代码与文字协议一致。
6. 审查旧 policy 与当前 policy 的 logprob、KL 方向和 mask 边界；说明 KL 过大、KL 接近零分别可能意味着什么。
7. 制定失败注入：空输出、全错 group、全对 group、无法解析答案、超长 completion。写出每种情况系统应该记录的字段。

提交时点：**真实 GRPO run 前必须完成 G5–G7。**

## Gate G2：训练与评估后完成

8. 从日志复核每步的 reward、advantage、KL、entropy、completion length、invalid rate；挑选一段异常曲线解释可能原因。
9. 使用同一评测集区分数学正确率、格式奖励收益和长度变化。说明 reward 提升能否证明推理能力提升。
10. 盲审至少 20 条 pre/post rollout，并给出一项下一轮只改变一个变量的实验建议。

提交时点：**G8–G10 完成后才讨论 GRPO 的阶段结论。**
