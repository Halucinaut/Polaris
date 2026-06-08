# Polaris Project Summary

## 1. 项目定位

Polaris 是一个本地优先的 LLM post-training 能力训练项目。目标是系统训练 SFT、DPO、GRPO、PPO、OPD 相关的工程与研究能力，包括训练信号理解、数据构造、实验诊断、评估设计、日志体系和硬件边界判断。

Polaris 不以提出新算法或刷 benchmark 为第一目标。项目的主要产出是可复现实验、失败诊断、数据构造经验、方法理解和本地 post-training 能力边界。

## 2. 硬件与执行约束

默认硬件是 MacBook Pro M5 Max，128GB 统一内存，2TB 存储。项目坚持本地优先，云端只作为未来可选项，不进入当前主路径。

默认策略是小模型、小数据、小步快验收。优先使用 Qwen3-0.6B 建立完整方法闭环，再选择性验证到更大模型。

## 3. 技术路线

Polaris 采用 “controlled calibration + business-proxy transfer” 路线。

Math reasoning 用作可验证校准任务。它的价值在于自动判分、reward 噪声低、评估口径清楚、失败容易归因，适合作为 SFT、DPO、GRPO 的第一实验域。

Tool-call / Structured Action 用作后续业务代理任务。它用于观察结构化输出、工具调用、schema validity、execution success、格式串扰、数据混合偏置和灾难性遗忘等更接近业务的问题。

Math 不是业务目标。Math 是 post-training calibration environment。后续会通过 Tool-call 或 structured task 建立业务迁移验证。

## 4. 当前阶段状态

M0 已完成并通过端到端验收。M0 只建立工程骨架，不包含真实模型下载、数据下载、MLX 接入或训练。

当前准备进入 M1。M1 的目标是在 Qwen3-0.6B 上完成第一组可复现 Math SFT。详细步骤、指标和验收标准见 M1.md。

M0 的具体内容、验收命令和通过标准见 M0.md。

## 5. 阶段路线图

M0：工程骨架与实验协议。已完成。

M1：Qwen3-0.6B Math SFT。目标是完成第一组可复现 SFT run，验证 loss、format adherence、eval、sample diff 和硬件日志。

M2：Math DPO。目标是学习 chosen/rejected pair 构造、preference margin、length bias、mode collapse 和 preference optimization 诊断。

M2.5：RL sanity。目标是验证 rollout、reward、logprob、KL、checkpoint、resume 和日志链路。

M3：Math GRPO。目标是建立 verifiable reward 下的 policy optimization 闭环。

M4：Math PPO。完成最小 PPO 或 failure report；理解 value、advantage、KL、reward normalization 的影响。

M4.5：混合数据类型 RL 流程。在单一 RL 训练中混合 Math + Tool-call 数据；验证多 reward 源（answer correctness + format + tool schema）的加权与冲突；记录 domain transfer 与 catastrophic forgetting。

M5：4B Math selective validation。完整 SFT/DPO；短程 GRPO；记录训练耗时、吞吐、峰值内存、checkpoint、max sequence length。

M6/M7：Tool-call SFT/DPO/GRPO demo。目标是把 post-training 能力迁移到业务代理任务。

M8：OPD study。近期以论文和实现阅读为主，不进入训练主路径。

## 6. 主要实验原则

所有实验必须由 config 驱动，并在 run 目录保存 frozen config。

所有实验必须由 registry 创建 run，不手工创建实验目录。

所有指标、硬件信息和样例变化使用 JSONL 记录。

每个正式 run 都需要保留 experiment card、metric report、sample diff、failure note 和 run report。

失败 run 必须保留错误日志和诊断信息，不能手工改成 completed。

每个阶段按小 step 推进。每个 step 必须有目标、改动文件、命令行验收方式和完成定义。


## 7. 禁止默认推进的事项

在未明确进入对应阶段前，不要实现 DPO、GRPO、PPO、OPD、Tool-call、多数据集混合、4B/8B 训练、云端训练或真实业务数据接入。

M1 阶段只服务于 Qwen3-0.6B Math SFT。