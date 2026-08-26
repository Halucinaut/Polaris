# M2 DPO v2 Style-Controlled：实验计划

## 1. 目标

在 Qwen3-0.6B 上验证：当 chosen 与 rejected 的数学答案均为正确时，DPO 是否能在不降低答案正确率的前提下，将模型输出风格从 SFT 的自由格式迁移到 `Solution + 连续编号步骤 + Final: The answer is \boxed{...}.\` 的结构化模板。

与 DPO v1（gold-vs-wrong）的关键区别：v2 的偏好信号仅来自风格差异，不来自正确性差异。

## 2. 数据

| split | 条数 | 来源 | 用途 |
|---|---:|---|---|
| train | 449 | `data/math/splits/dpo_v2_style_train_449.jsonl` | 训练 |
| stress | 50 | `data/math/splits/dpo_v2_style_stress_50.jsonl` | 风格压力测试 |
| stress eval | 50 | `data/math/splits/dpo_v2_style_stress_eval_50.jsonl` | stress-50 的 eval_gsm8k 格式 |
| quarantine | 1 | `data/math/quarantine/dpo_v2_style_invalid_1.jsonl` | 隔离，不进入训练或指标 |

### 数据来源说明

全部 500 条源样本来自同一批 GSM8K train d5 子集（`sft_d5_500`）。chosen 由 DeepSeek-V4-Flash 按风格模板重写，rejected 为 SFT 训练目标原文。500 条经严格校验后分为：

- **train（449 条）**：答案正确 + 模板合规 + 长度比 0.55–1.60 + 相似度 ≤ 0.97。
- **stress（50 条）**：答案正确 + 模板合规，但长度比超出 0.55–1.60 范围。来自同一批 500 条源样本中的长度异常子集，只用于风格压力测试，不能表述为独立泛化能力。
- **quarantine（1 条）**：`gsm8k_train_d5_0276`，chosen boxed answer 与 gold 不一致（生成时因 API 超时手动创建）。

### DPO v1 对照背景

DPO v1（329 条 gold-vs-wrong pairs）在 GSM8K-50 上的结果为：pass@1 = 14/50（与 SFT 持平），格式遵从从 50/50 降至 48/50。该结果仅作为对照背景；DPO v2 的目标是"正确性相同条件下的格式偏好"，不是在数学能力上超越 v1。

## 3. 配置

配置文件：`configs/qwen3_0_6b/dpo_v2_style.yaml`

| 字段 | 值 | 说明 |
|---|---|---|
| `run.name` | `qwen3_0_6b_dpo_v2_style` | |
| `run.tags` | `[dpo, math, 0.6b, v2, style]` | |
| `data.name` | `gsm8k_dpo_v2_style_train_449` | |
| `data.path` | `data/math/splits/dpo_v2_style_train_449.jsonl` | 449 条 |
| `data.max_samples` | `null` | 使用全部数据 |
| `training.method` | `dpo` | |
| `training.num_epochs` | `1` | |
| `training.batch_size` | `2` | |
| `training.gradient_accumulation_steps` | `4` | |
| `training.learning_rate` | `5.0e-7` | |
| `training.max_seq_length` | `2048` | |
| `training.seed` | `42` | |
| `dpo.beta` | `0.1` | |
| `dpo.policy_adapter_path` | `runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final` | M1 SFT |
| `dpo.ref_model_path` | `models/qwen3_0_6b/mlx` | |
| `dpo.ref_adapter_path` | `runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final` | M1 SFT，冻结 |
| `lora.*` | 与 DPO v1 一致 | r=32, alpha=32, q/k/v/o_proj |

超参数延续 DPO v1 已验证值，不做调整，以确保可比性。

## 4. 执行顺序

### 4.1 10-step sanity

```bash
./.venv/bin/python scripts/train_dpo.py \
  --config configs/qwen3_0_6b/dpo_v2_style.yaml \
  --max-steps 10 \
  --debug-batch
```

该命令处理 10 × batch_size × gradient_accumulation_steps = 10 × 2 × 4 = 80 条 pair。

### 4.2 Sanity 验收口径

| 检查项 | 来源 | 预期 |
|---|---|---|
| policy/reference 初始化相同 | `logs/debug_dpo_batch.json` | policy logprob = reference logprob |
| 更新前 dpo_margin ≈ 0 | `logs/debug_dpo_batch.json` | margin ≈ 0（参数尚未更新） |
| 第 1 步 dpo_loss ≈ ln(2) | `train_metrics.jsonl` 第 1 步 | loss ≈ 0.693（更新前计算） |
| 第 1 步 dpo_margin | `train_metrics.jsonl` 第 1 步 | 允许 > 0（参数更新后计算） |
| 第 10 步 loss 下降 | `train_metrics.jsonl` 第 10 步 | loss < 第 1 步 |
| provenance 路径正确 | `logs/checkpoint_provenance.json` | `policy_adapter_file` 和 `reference_adapter_file` 指向 M1 |
| reference_frozen = true | `logs/checkpoint_provenance.json` | `reference_frozen: true` |
| response_logprob_reduction | `logs/checkpoint_provenance.json` | 字段存在 |
| debug 文件路径 | `logs/debug_dpo_batch.json` | 文件存在且可解析 |
| 无 NaN/Inf | 所有指标文件 | 无异常值 |

### 4.3 完整一轮训练

```bash
./.venv/bin/python scripts/train_dpo.py \
  --config configs/qwen3_0_6b/dpo_v2_style.yaml
```

完整一轮 = ceil(449 / (2 × 4)) = 57 个 optimizer updates，最后一组处理 1 条 pair。

### 4.4 GSM8K-50 评估（数学能力基线）

```bash
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path <run>/checkpoints/final \
  --test-data data/math/gsm8k/split/test_converted_500.jsonl \
  --limit 50 \
  --skip-review \
  --temperature 0 \
  --max-new-tokens 512 \
  --output-path <run>/eval_gsm8k_50/results.json
```

### 4.5 Stress-50 评估（风格压力测试）

Stress-50 评估分两步：先由模型在 stress 问题上生成输出，再用 `eval_style_dpo.py` 评估风格合规率。

**注意**：chosen 投影测试（将 stress split 的 chosen 直接转为 predictions）仅验证数据与评估器自洽，不代表模型效果。模型效果结论只能来自实际生成的 SFT/DPO predictions。

#### 4.5.1 准备 eval 格式数据

```bash
./.venv/bin/python scripts/prepare_dpo_v2_style_eval_data.py \
  --input data/math/splits/dpo_v2_style_stress_50.jsonl \
  --output data/math/splits/dpo_v2_style_stress_eval_50.jsonl
```

已生成，50 条，problem_id 与 stress split 完全一致。

#### 4.5.2 SFT 模型在 stress-50 上生成

```bash
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final \
  --test-data data/math/splits/dpo_v2_style_stress_eval_50.jsonl \
  --skip-review \
  --temperature 0 \
  --max-new-tokens 512 \
  --output-path <sft-stress-output>/results.json
```

#### 4.5.3 DPO v2 模型在 stress-50 上生成

```bash
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path <run>/checkpoints/final \
  --test-data data/math/splits/dpo_v2_style_stress_eval_50.jsonl \
  --skip-review \
  --temperature 0 \
  --max-new-tokens 512 \
  --output-path <dpo-v2-stress-output>/results.json
```

SFT 与 DPO v2 必须使用相同 50 个 problem_id 和完全一致的生成参数（system prompt、temperature=0、max-new-tokens=512）。

#### 4.5.4 分别评估风格合规率

```bash
# SFT 风格评估
./.venv/bin/python scripts/eval_style_dpo.py \
  --predictions <sft-stress-output>/test_predictions.jsonl \
  --references data/math/splits/dpo_v2_style_stress_50.jsonl \
  --output-dir <sft-stress-style-eval>

# DPO v2 风格评估
./.venv/bin/python scripts/eval_style_dpo.py \
  --predictions <dpo-v2-stress-output>/test_predictions.jsonl \
  --references data/math/splits/dpo_v2_style_stress_50.jsonl \
  --output-dir <dpo-v2-stress-style-eval>
```

压力集报告必须同时给出答案正确率和风格合规率。

### 4.6 人工盲审

至少审阅 20 条 SFT 与 DPO v2 sample diff，检查：
- 风格迁移是否生效（DPO 输出是否更符合模板）
- 是否出现长度偏差、模板化或模式坍缩
- 数学推理质量是否退化

## 5. 预注册通过条件

| 条件 | 说明 |
|---|---|
| 无 NaN/Inf | 训练全程数值稳定 |
| reference_frozen=true | provenance 确认 reference 未被更新 |
| 训练数据 449 条 | 确认完整遍历 |
| GSM8K-50 pass@1 ≥ 14/50 | 不低于 SFT/DPO v1 基线 |
| Stress-50：答案正确率 + 风格合规率 | 同时报告两项指标 |
| 不以训练 loss 或 margin 单独宣称效果 | 必须有下游评估支撑 |

## 6. 约束

- 不修改 `scripts/train_dpo.py`。
- 不修改 split 数据文件（449/50/1 已冻结）。
- 不修改评估器（`eval_math.py`、`eval_style_dpo.py`）。
- 不修改既有 run 目录。
- 不创建新的 API 调用。
- 隔离的 1 条（`gsm8k_train_d5_0276`）不进入训练或任何指标。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 风格模板过拟合 | stress-50 提供长度异常分布的检验 |
| 数学能力退化 | GSM8K-50 作为能力基线对照 |
| 格式合规但推理质量下降 | 人工盲审至少 20 条 |
| v2 与 v1 不可比 | 超参数完全一致，唯一变量为数据 |

## 8. M1 初始化 SFT 风格可行性对照

### 8.0 已知问题：000038 与 M1 的 accumulation 覆盖错误

**000038 (SFT v2 style control)** 训练循环使用 `data_idx` 回绕（`data_idx = 0`），
导致每个 epoch 实际消费的数据量远超 449 条。
- 旧代码中每个 step 消费 `batch_size × grad_accum = 8` 条样本
- 224 步 × 8 = 1,792 条样本 = 449 条的 4× 覆盖
- **000038 不能用于正式结论**，仅作为"style 模板可学"的定性参考

**M1 (000030)** 使用相同的旧训练循环：
- 配置：num_epochs=3, batch_size=4, grad_accum=4
- 旧代码每步消费 4 × 4 = 16 条样本
- 实际执行 375 步 × 16 = 6,000 条样本 = 500 条的 12× 覆盖
- **M1 并非 3 epoch 的实际执行结果，而是 12 epoch**
- 此事实不影响 M1 模型的有效性，但需修正 epoch 声明

### 8.1 背景与目的

DPO v2 训练后 style adherence = 0/50，需判断是"DPO 无法迁移风格"还是"模型根本无法学会该模板"。
本对照直接用 SFT 在相同 chosen 数据上训练，使用已验证的 M1 adapter 作为初始化。

**重要声明：** 这是训练内 style-feasibility control，不是 DPO vs SFT 的公平效果对比。
两者 learning rate 和目标函数不同，不可直接比较训练指标。

### 8.2 配置

- 配置：`configs/qwen3_0_6b/sft_v2_style_control.yaml`
- 数据：`data/math/splits/sft_v2_style_control_train_449.jsonl`（449 条，target = chosen）
- 初始化：`runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final`（M1 adapter）
- 学习率：5.0e-5（已验证的 SFT 学习率）
- LoRA：r=32, alpha=32, q/k/v/o_proj
- 训练：1 epoch, batch_size=2, grad_accum=4, max_seq_length=2048

### 8.3 train_sft.py 扩展

- 新增 `sft.init_adapter_path` 配置项和 `--init-adapter-path` CLI 覆盖
- 加载顺序：base model → LoRA 注入 → init adapter weights → before_training smoke
- 写入 `logs/checkpoint_provenance.json`（base_model_path, init_adapter_file, init_adapter_loaded, response_supervision）
- 未设置 init_adapter_path 时行为不变

### 8.4 执行序列

```bash
# 1. 数据转换
python3 scripts/prepare_sft_v2_style_control_data.py

# 2. 10-step sanity
./.venv/bin/python scripts/train_sft.py \
  --config configs/qwen3_0_6b/sft_v2_style_control.yaml \
  --max-steps 10 --debug-dump-batch

# 3. 完整训练
./.venv/bin/python scripts/train_sft.py \
  --config configs/qwen3_0_6b/sft_v2_style_control.yaml

# 4. Train Probe 评估
python3 scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path <run_dir>/checkpoints/final \
  --test-data data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl \
  --output-path <run_dir>/eval_train_probe/results.json \
  --temperature 0 --max-new-tokens 512 --skip-review

python3 scripts/eval_style_dpo.py \
  --predictions <run_dir>/eval_train_probe/test_predictions.jsonl \
  --references data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl \
  --output-dir <run_dir>/eval_train_probe/style_eval
```

### 8.5 通过标准

- **关键指标：** Train Probe-30 模板合规率 **明显高于 0/30**（至少 > 5/30）
- 参考：answer correctness 不应显著低于 SFT baseline（24/30）

### 8.6 决策树

```
SFT control style adherence > 5/30?
├── 是 → DPO v2 本身的问题（lr、beta、数据对齐）
│       → 下一轮设计单变量 DPO 改动
└── 否 → 模板本身或数据质量问题
        → 先修订 SFT 数据或模板格式
```

## 9. DPO v3：学习率单变量改动（lr=5e-6）

### 9.1 背景

SFT control 已证明风格模板可学（100% adherence）。
DPO v2（lr=5e-7）风格迁移失败，学习率过低是首要假设。
DPO v3 将 lr 提升 10× 至 5e-6，其余超参不变。

### 9.2 配置

- 配置：`configs/qwen3_0_6b/dpo_v3_style_lr5e6.yaml`
- 数据：同 DPO v2（449 pairs, dpo_v2_style_train_449.jsonl）
- 学习率：**5.0e-6**（唯一变更）
- beta=0.1, batch_size=2, grad_accum=4, num_epochs=1, seed=42
- LoRA r=32, alpha=32, q/k/v/o_proj
- policy/ref 均从 M1 adapter 初始化

### 9.3 DPO 训练循环

DPO script (`train_dpo.py`) 已使用正确的 pre-computed update_groups：
- 449 pairs / batch_size=2 = 224 micro-batches
- 224 / grad_accum=4 = 56 full groups + 1 partial group = 57 updates
- 梯度按实际 micro-batch 数均分（非 grad_accum）

### 9.4 预注册通过条件

| 指标 | 阈值 | 依据 |
|---|---|---|
| Probe-30 style adherence | ≥ 6/30 | SFT control=30/30，DPO 只需学会一部分 |
| Stress-50 style adherence | ≥ 10/50 | 分布外泛化的基本证据 |
| Stress-50 answer correct | ≥ 43/50 | 不低于 M1 baseline (48/50) 的 90% |

### 9.5 执行序列

```bash
# 1. 10-step sanity
./.venv/bin/python scripts/train_dpo.py \
  --config configs/qwen3_0_6b/dpo_v3_style_lr5e6.yaml \
  --max-steps 10 --debug-batch

# 2. 完整训练（57 updates）
./.venv/bin/python scripts/train_dpo.py \
  --config configs/qwen3_0_6b/dpo_v3_style_lr5e6.yaml

# 3. 评估
# Probe-30
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path <run_dir>/checkpoints/final \
  --test-data data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl \
  --output-path <run_dir>/eval_train_probe/results.json \
  --temperature 0 --max-new-tokens 512 --skip-review

python3 scripts/eval_style_dpo.py \
  --predictions <run_dir>/eval_train_probe/test_predictions.jsonl \
  --references data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl \
  --output-dir <run_dir>/eval_train_probe/style_eval

# Stress-50
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path <run_dir>/checkpoints/final \
  --test-data data/math/splits/dpo_v2_style_stress_eval_50.jsonl \
  --output-path <run_dir>/eval_stress_50/results.json \
  --temperature 0 --max-new-tokens 512 --skip-review

python3 scripts/eval_style_dpo.py \
  --predictions <run_dir>/eval_stress_50/test_predictions.jsonl \
  --references data/math/splits/dpo_v2_style_stress_eval_50.jsonl \
  --output-dir <run_dir>/eval_stress_50/style_eval

# GSM8K-50
./.venv/bin/python scripts/smoke/eval_gsm8k.py \
  --model-path models/qwen3_0_6b/mlx \
  --adapter-path <run_dir>/checkpoints/final \
  --test-data data/math/gsm8k/split/test_converted_500.jsonl \
  --output-path <run_dir>/eval_gsm8k_50/results.json \
  --temperature 0 --max-new-tokens 512 --limit 50 --skip-review
```

### 9.6 决策树

```
DPO v3 达到全部通过条件?
├── 是 → lr 是关键变量；可进一步调优 beta / 多 epoch
└── 否 → 分析具体失败项：
    ├── style 仍为 0 → DPO 目标函数本身不适合 style migration
    ├── style 有进步但不足 → 继续调参（lr/beta/epochs）
    └── correct 下降过多 → lr 过大导致能力退化，需折中

## 10. Boundary Diagnosis（无训练诊断）

### 10.1 方法

在 `<think>\n` 后、`Solution:` 分支与 rejected 首 token 的分歧点，
比较 M1、DPO v2、DPO v3、SFT control 的条件对数概率。

### 10.2 结果（10 条 probe 样本）

| 模型 | logP("Solution:") | Greedy 进入 chosen |
|---|---:|---|
| M1 (SFT) | -23.9 | 0/10 |
| DPO v2 (lr=5e-7) | -22.0 | 0/10 |
| DPO v3 (lr=5e-6) | -16.6 | 0/10 |
| SFT control | ≈ 0.0 | 10/10 |

### 10.3 结论

**DPO 的 margin 增益不发生在格式分歧点。** 即使 lr 提升 10×，
logP("Solution:") 仍为 -16.6（SFT control 为 ~0），greedy 从未进入 chosen 分支。
DPO 的偏好增益仅在 teacher-forced 的后续 token 中积累。

这解释了为什么 DPO 风格迁移失败：模型从未学会在分歧点选择 `Solution:`。

## 11. DPO v4：最小对比数据（style-only diff）

### 11.1 设计原理

Boundary diagnosis 证明 DPO 无法在分歧点学到 style。
v4 的数据设计将梯度信号集中在分歧点：
- chosen: 保持结构化模板（Solution:/numbered steps/Final:）
- rejected: 从 chosen 自动派生，只删除 style wrapper
- 推理正文和 boxed 答案逐字节一致
- chosen/rejected 相似度 0.16-0.95（mean=0.79），高相似度是目标属性

### 11.2 配置

- 配置：`configs/qwen3_0_6b/dpo_v4_style_minimal.yaml`
- 数据：`data/math/pilots/dpo_v4_minimal_449.jsonl`
- 学习率：5.0e-7（v2 原始学习率）
- 其余配置与 v2 完全一致

### 11.3 执行序列

```bash
# 1. 10-step sanity
./.venv/bin/python scripts/train_dpo.py \
  --config configs/qwen3_0_6b/dpo_v4_style_minimal.yaml \
  --max-steps 10 --debug-batch

# 2. 完整训练（57 updates）
./.venv/bin/python scripts/train_dpo.py \
  --config configs/qwen3_0_6b/dpo_v4_style_minimal.yaml

# 3. 评估（同 v3 评估流程）
```

### 11.4 停止条件

若 v4 仍出现以下任一情况，停止 DPO 风格训练：
- Probe/Stress 0% 风格合规
- 明显正确率回退（Stress correct < 43/50）
