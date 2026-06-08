---
title: "Polaris 项目设计最终版"
subtitle: "本地优先的 Post-Training 能力训练项目"
date: "2026-06-02"
---

# Polaris 项目设计最终版

## 0. 项目定位

Polaris 是一个本地优先的 post-training 能力训练项目。项目第一目标是系统掌握 SFT、DPO、GRPO、PPO、OPD 相关训练信号、数据构造、实验诊断和硬件边界。对外公开材料属于副产物，从稳定实验、失败复盘和方法笔记中抽取。

项目采用两条任务线：Math reasoning 作为第一主线，Tool-call 作为第二主线。Math 线用于串通 SFT -> DPO -> GRPO -> PPO 的核心流程，因为 reward 可验证、评估简单、干扰变量少。Tool-call 线用于训练垂类领域微调能力，覆盖任务定义、轨迹合成、执行校验、偏好构造、reward 设计和 post-training 迁移。

项目不以提出新算法为目标。核心资产是方法理解、实验日志、失败诊断、数据构造能力、MLX 适配经验和 M5 Max 本地训练边界。

## 1. 约束与原则

| 维度      | 最终设定                                   |
| ------- | -------------------------------------- |
| 硬件      | MacBook Pro M5 Max，128 GB 统一内存，2 TB 存储 |
| 执行原则    | 本地可跑、本地可诊断、本地可迭代                       |
| 云端      | 不进入主路径。只作为未来可选项记录，不作为里程碑验收             |
| 精度      | BF16 默认                                |
| 微调方式    | LoRA 优先                                |
| 默认 LoRA | r=32, alpha=32                         |
| 高配 LoRA | r=64 作为后续对照，不作为默认起点                    |
| 全参数训练   | 只在 0.6B 的 SFT/DPO 阶段做对照                |
| 文档策略    | 内部研究笔记优先，公开教程后置抽取                      |

每个实验固定产出四类材料：experiment card、metric report、sample diff、failure note。实验记录先服务于个人学习和复盘，稳定后再整理成公开文档。

## 2. 模型层级

| 层级         | 模型         | 项目职责                       | 验收方式                                 |
| ---------- | ---------- | -------------------------- | ------------------------------------ |
| Primary    | Qwen3-0.6B | 完整学习模型，覆盖 SFT、DPO、GRPO、PPO | 每个方法至少有一组完整日志、曲线、样例变化和失败记录           |
| Validation | Qwen3-4B   | 选择性验证 0.6B 结论              | 完整 SFT/DPO，小规模 GRPO，PPO 可选           |
| Boundary   | Qwen3-8B   | 本地边界检查                     | 推理 baseline、可选 LoRA SFT smoke，不承诺 RL |
| Appendix   | Qwen3.5-4B | 架构差异研究备选                   | 暂不进入执行计划                             |

Qwen3-0.6B 是唯一完整 pipeline 模型。Qwen3-4B 用于验证训练经验能否部分迁移到更大模型。Qwen3-8B 只用于边界检查，避免吞掉主线调试时间。Qwen3.5-4B 暂时只保留为未来 architecture appendix，避免混合架构适配问题干扰 post-training 学习目标。

## 3. 训练框架策略

| 方法   | 本地策略                              | 风险等级 | 项目定位    |
| ---- | --------------------------------- | ---- | ------- |
| SFT  | MLX 路线优先                          | 低    | 主线必做    |
| DPO  | MLX 路线优先，4B 可缓存 reference logprob | 中低   | 主线必做    |
| GRPO | 先做 0.6B sanity，再扩展                | 中高   | 核心挑战    |
| PPO  | 只做最小实验或失败分析                       | 高    | 算法理解实验  |
| OPD  | 先做论文和实现阅读                         | 高    | 理论与实现研究 |

M0 阶段只验收工程骨架和实验协议：统一 config、logging、registry、文档模板与 fake sanity run 闭环；不下载模型、不接入 MLX/PyTorch、不执行真实训练。Qwen3 chat template、LoRA target module、loss、checkpoint、eval 等真实训练链路在 M1 的 0.6B SFT sanity 中验证。DPO 真实训练链路在 M2 验证。GRPO 在进入正式 M3 前必须完成 M2.5 sanity check，先验证 rollout、reward、logprob、KL、checkpoint、resume 和日志链路。

PPO 不作为主线阻塞项。若本地框架不稳定，输出 failure report，记录 value function、advantage estimation、KL control、reward normalization、rollout buffer 的具体问题。

OPD 暂不安排云端训练。近期目标是阅读论文、阅读 verl 实现、整理 data flow、loss structure、teacher/student interaction、on-policy rollout、dense supervision 和 failure modes。云端 OPD 实验只写入 future optional，不进入当前验收。

## 4. 数据策略

### 4.1 Math 主线

Math reasoning 贯穿第一阶段全流程。任务域保持一致，便于观察不同训练方法带来的增量变化。

| 方法   | 数据                        | 训练信号                                                            | 目的                             |
| ---- | ------------------------- | --------------------------------------------------------------- | ------------------------------ |
| SFT  | OpenR1-Math 子集，5K-10K     | `<think>` 格式与 reasoning trajectory                              | 注入结构化推理格式                      |
| DPO  | 同域 self-constructed pairs | correct / complete / concise 优于 incorrect / verbose / malformed | 学偏好构造与偏好优化                     |
| GRPO | MATH Level 3-5            | answer correctness reward + format reward                       | 学 verifiable reward 下的策略优化     |
| PPO  | 同 GRPO                    | reward + value + advantage + KL                                 | 对比 PPO 与 GRPO 的工程复杂度和稳定性       |
| OPD  | 同域概念研究                    | teacher dense supervision on student-sampled states             | 学 on-policy distillation 的训练信号 |

Math 线启动前必须做 baseline screening：抽样评估 MATH Level 2-5，在 0.6B 和 4B 上选择 baseline accuracy 约 20%-60% 的难度区间。低于 20% 的区间 reward 过稀疏，高于 60% 的区间提升空间不足。

### 4.2 Tool-call 垂类线

Tool-call 线在 Math GRPO 完成后启动。第一轮只做 30-50 个 seed tasks，先验证任务定义、schema、执行器、过滤器和 SFT 格式。pipeline 稳定后扩展到 100-200 个 seed tasks。

初始流程：定义 3-5 个 mock tools，写 30-50 个 seed tasks，用 DS V4-Pro 合成正式轨迹，用 DS V4-Flash 做调试迭代，用 MiMo-V2.5-Pro 做小规模质量对比。所有轨迹必须通过执行校验，成功轨迹进入 SFT 数据。

后续流程：用当前模型多采样生成轨迹，按执行结果、schema validity、工具调用必要性、任务完成度排序，构造 DPO pairs。最后设计 format reward 和 outcome reward，进行小规模 GRPO demo。

Tool-call 线的目标是训练垂类微调能力。重点是完整定义领域任务、生成训练数据、验证轨迹质量、构造偏好数据、设计 reward、诊断工具调用失败模式。

## 5. 里程碑

| 里程碑  | 内容                           | 验收标准                                                                                                  |
| ---- | ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| M0   | 工程骨架与实验协议                   | 统一 config、logging、registry、文档模板建立；fake SFT/DPO/GRPO/RL sanity run 通过；不下载模型、不接入 MLX/PyTorch、不执行真实训练              |
| M1   | 0.6B Math SFT                | Qwen3-0.6B baseline eval 完成；MLX/LoRA 10-step SFT sanity 通过；loss 下降；format adherence 提升；eval 无明显退化；样例输出变化可解释                                                      |
| M2   | 0.6B Math DPO                | chosen/rejected pair 构造完成；DPO 真实训练链路可运行；preference margin 可记录；长度偏移与模式坍缩被监控                                          |
| M2.5 | 0.6B RL sanity check         | 20 个 Math prompts；group size=2；max completion=128/256；10-20 update steps；reward、logprob、KL、loss 全部可记录 |
| M3   | 0.6B Math GRPO               | rollout-update-eval 闭环稳定；reward curve、pass\@1、KL、entropy、completion length、invalid rate 完整记录          |
| M4   | 0.6B Math PPO                | 完成最小 PPO 或 failure report；理解 value、advantage、KL、reward normalization 的影响                              |
| M4.5 | 混合数据类型 RL 流程              | 在单一 RL 训练中混合 Math + Tool-call 数据；验证多 reward 源（answer correctness + format + tool schema）的加权与冲突；记录 domain transfer 与 catastrophic forgetting |  
| M5   | 4B Math selective validation | 完整 SFT/DPO；短程 GRPO；记录训练耗时、吞吐、峰值内存、checkpoint、max sequence length                                      |
| M6   | Tool-call SFT/DPO            | mock tools 与执行器完成；轨迹合成与过滤完成；模型 schema validity 和 task completion 提升                                   |
| M7   | Tool-call GRPO demo          | format reward 与 outcome reward 可运行；记录 tool avoidance、tool abuse、pseudo-JSON、format reward overfitting |
| M8   | OPD study                    | 完成 OPD 技术笔记；整理流程、loss、teacher/student 交互、failure modes；不要求训练跑通                                        |

执行优先级固定为 M0 -> M1 -> M2 -> M2.5 -> M3。0.6B Math SFT/DPO/GRPO 闭环跑通后，Polaris 的第一核心验收成立。PPO、4B、Tool-call、OPD 均不能阻塞这个核心闭环。M4.5 作为探索性里程碑，在 M4 PPO 和 M5 4B validation 之间插入，不阻塞主线进度。

## 6. 指标与日志

### 6.1 Math 指标

pass\@1、format adherence、answer extraction success rate、average completion length、reward mean/std、KL、entropy、invalid output rate、tokens/sec、peak memory。

### 6.2 DPO 指标

chosen logprob、rejected logprob、logprob margin、preference accuracy、response length shift、sample quality review。

### 6.3 Tool-call 指标

schema validity、tool-call success rate、execution success rate、unnecessary tool rate、missing tool rate、JSON parse failure rate、multi-step completion rate。

### 6.4 硬件日志

每个实验必须记录 model size、LoRA rank、sequence length、batch size、gradient accumulation、tokens/sec、samples/sec、peak memory、checkpoint size、total wall time。长期目标是形成 M5 Max post-training capability table。

## 7. 仓库结构

```text
polaris/
├── README.md
├── pyproject.toml
├── Makefile
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── qwen3_0_6b/
│   │   ├── sft_math.yaml
│   │   ├── dpo_math.yaml
│   │   └── grpo_math.yaml
│   ├── qwen3_4b/
│   │   ├── sft_math.yaml
│   │   ├── dpo_math.yaml
│   │   └── grpo_math_short.yaml
│   └── qwen3_8b/
│       └── sft_smoke.yaml
│
├── data/
│   ├── math/
│   │   ├── sources/
│   │   │   └── .gitkeep
│   │   ├── splits/
│   │   │   └── .gitkeep
│   │   ├── baseline_screening/
│   │   │   └── .gitkeep
│   │   └── manifest.yaml
│   └── tool_call/
│       ├── README.md
│       └── .gitkeep
│
├── polaris/
│   ├── __init__.py
│   ├── config.py
│   ├── registry.py
│   │
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── base.py                   # M0: 接口定义
│   │
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── math_answer.py            # M1 起实现
│   │
│   ├── evaluators/
│   │   ├── __init__.py
│   │   └── math_eval.py              # M1 起实现
│   │
│   ├── data_builders/
│   │   ├── __init__.py
│   │   └── math_sft.py               # M1 起实现
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── hardware.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── chat_template.py
│       ├── checkpoint.py
│       └── lora.py
│
├── scripts/
│   ├── sanity/
│   │   ├── sanity_sft.py
│   │   ├── sanity_dpo.py
│   │   ├── sanity_grpo.py
│   │   └── sanity_ppo.py
│   │
│   ├── prepare_math_data.py
│   ├── baseline_screening.py
│   ├── build_dpo_pairs.py
│   │
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   │
│   ├── eval_math.py
│   └── generate_experiment_card.py
│
├── runs/
│   └── .gitkeep
│
├── docs/
│   ├── templates/
│   │   ├── experiment_card.md
│   │   ├── failure_note.md
│   │   ├── metric_report.md
│   │   └── sample_diff.md
│   ├── method_cards/
│   │   ├── sft.md
│   │   ├── dpo.md
│   │   ├── grpo.md
│   │   ├── ppo.md
│   │   └── opd.md
│   ├── experiment_notes/
│   ├── failure_cases/
│   ├── feasibility_reports/
│   ├── framework_adaptation/
│   │   └── _template.md
│   ├── hardware_reports/
│   └── opd_study.md
│
└── future/
    ├── optional_cloud_opd.md
    └── qwen3_5_architecture_appendix.md

```

`docs/framework_adaptation/` 用于记录 MLX 适配问题、最小复现、patch、issue、PR。`docs/feasibility_reports/` 用于记录 Apple Silicon 上 SFT/DPO/GRPO/PPO 的可行性边界。

## 8. MLX 开源适配副线

Polaris 可以把真实阻塞转化为开源维护贡献。贡献范围限定为项目需求驱动的小修复、小文档、小复现，不扩展成完整框架开发。

优先贡献类型：复现文档、配置示例、bugfix、reference logprob cache、reward interface、diagnostic logging、sanity script。暂不承诺 PPO trainer 重写、OPD trainer 实现或大规模架构改造。

每个适配问题固定记录：环境、模型、数据、最小命令、预期行为、实际行为、日志、初步定位、workaround、patch、issue/PR 状态。PR 是否合并不作为 Polaris 验收标准，高质量复现和内部 workaround 计入有效产出。

## 9. 风险登记

| 风险             | 等级 | 表现                                             | 缓解方式                                   |
| -------------- | -- | ---------------------------------------------- | -------------------------------------- |
| MLX RL 稳定性     | 高  | GRPO/PPO rollout、mask、KL、checkpoint、resume 不稳定 | 先做 M2.5；0.6B 优先；失败写 feasibility report |
| 本地 GRPO 耗时     | 高  | 4B GRPO 可能耗时过长                                 | 4B 只做 short run；0.6B 是主结果              |
| PPO 复杂度        | 高  | value、advantage、clip、KL 任一环节出错都可能无效训练          | PPO 定义为理解实验，失败报告可验收                    |
| Tool-call 数据污染 | 中高 | 合成轨迹错误、schema 偏差、执行器不严谨                        | 第一轮 30-50 seed tasks；先验证过滤链路           |
| 范围膨胀           | 中高 | Math、Tool-call、PPO、OPD、8B 同时推进                 | 严格执行 M0-M3 优先级                         |
| 8B 吞吐过低        | 中  | 本地训练成本不可接受                                     | 8B 只做推理和 SFT smoke                     |
| OPD 云端诱惑       | 中  | 云预算与主约束冲突                                      | OPD 近期只做 study；云端写入 future optional    |
| CUDA/MLX 差异    | 中  | 同配置曲线、吞吐、数值表现不同                                | eval\_cards 记录完整环境，不追求跨硬件曲线一致          |

## 10. 第一轮验收标准

第一轮只验收 0.6B Math 核心闭环：SFT、DPO、GRPO。完成条件如下：

1. Qwen3-0.6B 的 SFT、DPO、GRPO 均至少完成一组可复现实验。
2. 每组实验都有 config、日志、曲线、指标表、样例 diff、failure note。
3. MATH 难度区间经过 baseline screening。
4. GRPO 的 reward gain 能被解释，区分真实解题提升、格式奖励提升和长度策略变化。
5. 形成第一版 M5 Max post-training capability table。

第一轮完成后再判断是否进入 PPO、4B validation 和 Tool-call。PPO 不阻塞 Tool-call。4B GRPO 不阻塞 Tool-call。OPD 不阻塞任何实验主线。

## 11. 最终执行摘要

Polaris 的最终执行路径是：先在 Qwen3-0.6B 上完成 Math SFT/DPO/GRPO 闭环，建立 post-training 方法理解、日志模板、失败诊断和本地硬件边界；随后用 Qwen3-4B 做 SFT/DPO 与短程 GRPO 的选择性验证；再进入 Tool-call 垂类微调线，训练从数据构造到 reward 设计的完整能力；PPO 作为算法理解实验，OPD 作为理论与实现研究主题。整个项目坚持 MBP 本地优先，公开材料只从稳定内部资产中抽取。
