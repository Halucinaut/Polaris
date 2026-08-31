# Polaris 阶段作业系统

这里记录你亲自完成的证据链。每个阶段都有三份固定文件：

```text
<stage>/
├── assignment.md  # 题目、证据文件和提交时点
├── submission.md  # 你填写的答案、命令输出和判断
└── gate.md        # 阶段门禁与我的批改记录
```

使用规则：题目完成后只编辑对应的 `submission.md`；我根据仓库证据在 `gate.md` 记录通过、退回或待补证据。未通过的门禁不推进到下一次训练。训练日志、实验结果和代码仍保留在原目录，不复制到本目录。

| 阶段 | 当前定位 | 入口 |
|---|---|---|
| SFT | 已完成，作为回顾与证据审计 | [sft](sft/assignment.md) |
| DPO | 当前主线，进行 DPO v2 style-controlled 实验 | [dpo](dpo/assignment.md) |
| GRPO | 未来主线，先完成 M2.5 的 RL 链路验证 | [grpo](grpo/assignment.md) |
| OPD | 研究型学习，不进入当前本地训练主路径 | [opd](opd/assignment.md) |

每个提交都应写明：结论、证据路径或命令、数字结果、该结论不能证明什么。只写理论定义不构成提交。
