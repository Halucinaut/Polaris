# OPD 作业

目标：完成 OPD 的论文与实现研究准备。按当前项目路线，OPD 是研究型阶段，不安排本地正式训练；本作业不要求伪造实验结果。

## Gate O0：概念与数据流，阅读后完成

1. 用自己的图或文字描述 teacher、student、student-sampled state、teacher supervision、loss 之间的数据流；注明每个张量或文本来自谁。
2. 对比 SFT、DPO、GRPO、OPD 的训练数据来源与 policy 是否在线采样。指出 OPD 的关键 on-policy 成分。
3. 写出你采用的 OPD 论文或实现版本、阅读日期和三个无法仅靠摘要回答的技术问题。

提交时点：**开始深入代码阅读前完成 O1–O3。**

## Gate O1：实现审阅，方案设计前完成

4. 选定一个实际实现，定位 rollout 生成、teacher 打分或 token-level supervision、student loss、KL 或稳定项、batch 过滤五处代码入口。
5. 选择一个 token-level supervision 样例，写出 mask 应覆盖和不应覆盖的 token；解释错位一位会怎样影响 loss。
6. 列出至少四个失败模式：teacher/student tokenizer 不一致、teacher 泄漏正确轨迹、离线数据冒充 on-policy、长度或格式捷径、rollout 过滤偏差等；为每项给出可观测诊断。

提交时点：**任何实现 proposal 前完成 O4–O6。**

## Gate O2：研究结论，暂不触发训练

7. 写一份一页的 OPD feasibility note：当前 Polaris 是否具备 teacher、rollout 成本、存储、评估和预算条件；结论可以是“不应实现”。
8. 设计一个未来最小实验，只写数据、模型、指标、预算、失败停止条件；不得默认云端训练已经获批。

提交时点：**O7–O8 完成后，本阶段作为 study 归档。**
