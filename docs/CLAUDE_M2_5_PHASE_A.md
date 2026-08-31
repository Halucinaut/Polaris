# Claude 执行说明：M2.5-A 真实 Rollout 与 Reward 审计

## 任务目标

在 Polaris 中实现 M2.5 的第一段真实链路：从 M1 adapter 加载 Qwen3-0.6B，对 Math prompt 按 group 采样多个 completion，执行可复算 reward，并将每条 rollout 与 group 汇总写入 registry 创建的 run 目录。

本期不实现 policy update、optimizer、GRPO loss 或正式训练。完成条件是用真实模型得到可审计的 rollout/reward 产物，确认数据分布、reward 方差、答案抽取与日志格式。

## 已知仓库事实

- `scripts/train_grpo.py`、`polaris/rewards/math_answer.py`、`polaris/trainers/base.py` 目前为空。
- `configs/qwen3_0_6b/grpo_math.yaml` 是配置草案，现有 `runs/000003`、`000004` 和 `scripts/sanity/sanity_grpo.py` 都是 fake sanity。
- M1 初始化为 `runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final`。
- `scripts/eval_math.py` 已提供 `extract_predicted_answer`、`answers_match`、`has_m1_format_adherence` 和数值归一化逻辑。奖励实现应复用这些语义，不能复制一套不一致的答案判定器。
- run 创建、冻结 config 与 JSONL 日志遵循 `scripts/train_dpo.py`、`scripts/train_sft.py`、`polaris/run_registry.py` 的现有约定。

## 必须实现

### 1. 纯奖励模块

实现 `polaris/rewards/math_answer.py`，提供无 MLX 依赖、可单测的函数：

- 输入：原始 completion、标准答案，以及由配置传入的 reward 权重。
- 输出：结构化结果，至少包含 `predicted_answer`、`extraction_method`、`answer_correct`、`format_adherent`、`invalid_output`、各 reward component 与 `reward_total`。
- 答案正确奖励固定为 `1.0`，错误且可解析为 `0.0`，空输出或无法提取答案为 `invalid_penalty`。
- 格式奖励由配置控制，默认 `0.05`；只在 `has_m1_format_adherence` 为真时生效。
- `invalid_penalty` 由配置控制，默认 `-0.1`。
- 正确答案可以同时获得格式奖励；错误答案不得因格式获得正奖励。

模块不能从 `scripts/` 导入。若需复用 `eval_math.py` 的逻辑，将无副作用的提取、归一化和比较函数移动到一个可被评测脚本与奖励模块共同导入的 `polaris` 模块，并保留 `eval_math.py` 的现有 CLI 行为。

### 2. 真实 rollout-audit CLI

实现 `scripts/train_grpo.py` 的 `--mode rollout-audit`。它必须：

- 合并 `configs/base.yaml` 与 `configs/qwen3_0_6b/grpo_math.yaml`，通过 run registry 创建新目录并冻结 config。
- 加载 base model、LoRA 层和 M1 adapter；使用与现有 Math eval 一致的 system prompt/chat template。
- 读取配置数据集，支持 `--max-prompts`、`--group-size`、`--temperature`、`--max-completion-length`、`--seed` 覆盖。
- 每题独立采样 `group_size` 个 completion。temperature 必须大于 0；固定 seed 时输出顺序和采样 seed 的记录必须可复查。
- 对每条 completion 调用奖励模块；记录 prompt 标识、group 标识、rollout 标识、completion、答案提取结果、reward component、总 reward、采样参数与 completion token 数。
- 对每个 group 记录 reward mean、reward std、min/max、正确数量、无效数量与 `nonzero_reward_variance`。全对、全错、全同 reward group 仍须完整写入，不能静默丢弃。
- 写入至少以下文件：
  - `samples/rollouts.jsonl`
  - `metrics/rollout_metrics.jsonl`
  - `logs/reward_protocol.json`
  - `logs/checkpoint_provenance.json`
  - `logs/generation_config.json`
- 日志使用现有 JSONL/JSON 风格；发生异常时保留 failure log 并将 run 标记失败。

`--mode train`、GRPO loss、optimizer update、reference KL、checkpoint/resume 留给下一期。CLI 在未显式传入 `--mode rollout-audit` 时应拒绝运行，避免被误认为已具备训练能力。

### 3. 配置与测试

更新 `configs/qwen3_0_6b/grpo_math.yaml`，补充 `grpo` 或 `reward` 配置段，包含：M1 adapter 路径、reference adapter 路径、temperature、group size、max completion length、format reward、invalid penalty。M2.5 默认建议为 `max_samples: 16`、`group_size: 4`、`temperature: 0.7`；这些值必须可被 CLI 覆盖。

新增单元测试，至少覆盖：

1. 正确且 M1 格式合规的 completion 获得 `1.05`。
2. 正确但格式不合规的 completion 获得 `1.0`。
3. 可解析但错误的 completion 获得 `0.0`。
4. 空输出和无法解析输出获得 `-0.1`。
5. 分数或 LaTex 数值答案与既有评测逻辑一致。
6. group 汇总正确处理全对、全错、混合和零方差情形。
7. 配置覆盖与输出记录字段可在无 MLX 环境下测试。

## 禁止事项

- 不启动正式 GRPO 训练，不添加 optimizer update，不声称 reward 提升或能力提升。
- 不改动 DPO 数据、DPO run、归因报告、M1 checkpoint 与现有评测口径。
- 不调用外部 API，不下载新模型，不扩大到 4B、Tool-call、PPO 或 OPD。
- 不删除、覆盖、暂存或提交当前 worktree 的无关改动。
- 不以 fake sanity 替代真实模型 rollout。

## 验收与回报格式

完成后执行相关单元测试和完整 `make test`。然后执行一次真实但小规模的 rollout audit：`--max-prompts 4 --group-size 4 --temperature 0.7`。回报中必须给出：

1. 改动文件清单与每个文件的职责。
2. 测试命令及实际结果。
3. 新 run 的目录、五个必需产物路径和每个文件的记录数。
4. 至少一个 reward 为正、一个错误或无效、一个零方差或非零方差 group 的原始 JSONL 证据；若小样本没有覆盖，明确说明并提供可复现的失败注入测试证据。
5. 已知限制与下一期实现 `reference KL + GRPO update + checkpoint/resume` 前需要解决的问题。

不要仅报告“实现完成”。所有结论必须指向代码、测试或新 run 的具体路径。
