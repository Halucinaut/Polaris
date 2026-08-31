# Polaris 实战复习：从 SFT 到 DPO

这份教程用于重新接手 Polaris。目标是恢复三种能力：解释训练目标、审查数据与代码、判断实验结论是否可信。阅读过程中需要实际打开文件、计算指标并写下判断。

教程基于当前仓库证据。M1 已完成，M2 已完成偏好数据和 10-step 训练链路验证，完整 DPO 实验尚未完成。

## 使用方法

建议分三次完成，每次约 60–90 分钟：第一次完成 SFT 数据与 loss，第二次完成 SFT 评估与故障复盘，第三次完成 DPO 数据、公式和代码审查。

每道练习先独立作答，再查看文末参考答案。答案应写明证据文件、相关字段或代码行，避免只写结论。

准备命令：

```bash
cd "/Users/halucinaut/Vault/02_Projects/Polaris"
make test
make list-runs
```

## 1. 全局路线

Polaris 使用 Math reasoning 建立可验证的 Post-Training 主线：

```text
Qwen3-0.6B Base
    │
    │ SFT：给定标准回答，最大化标准回答的 token 概率
    ▼
Qwen3-0.6B SFT
    │
    │ DPO：给定 chosen/rejected，扩大相对偏好
    ▼
Qwen3-0.6B DPO
    │
    │ RL sanity：验证 rollout、reward、logprob、KL 和恢复训练
    ▼
GRPO
```

GSM8K 的作用是提供可自动判分的校准环境。这里要验证的是训练链路、诊断方法和本地硬件边界。单次分数提升不能代表通用数学能力提升。

当前可信状态：M0 工程骨架通过；M1 SFT 通过；M2 DPO 已收口。DPO v1 证明训练链路，v2–v4 style experiments 与 SFT control 说明当前离线 DPO 设置未完成目标风格迁移；`000051` 的分支点归因解释了 teacher-forced 概率变化与自由生成失配。结论与 artifact 索引见 `docs/experiment_notes/m2_dpo_closeout.md`。下一阶段是 M2.5 的真实 RL sanity。

# 2. SFT：从标准答案到可审计训练

## 2.1 SFT 究竟优化什么

一条 SFT 样本可以写成 prompt `x` 和标准回答 `y=(y₁,...,yₜ)`。训练目标是最小化标准回答 token 的负对数似然：

$$
L_{\text{SFT}}=-\frac{1}{T}\sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t})
$$

Polaris 只监督 assistant response。system prompt 和 user question 提供条件，不进入 loss。这样做使训练目标集中在目标回答，避免大量固定 prompt token 主导平均 loss。

SFT 学到的是数据中出现的行为，包括解题路径、`<think>` 结构、`\boxed{}` 格式、措辞习惯和潜在错误。数据协议就是训练协议。

## 2.2 一条真实样本如何形成

打开第一条正式训练数据：

```bash
sed -n '1p' data/math/splits/sft_d5_500.jsonl
```

它包含三个关键部分：

```json
{
  "messages": [
    {"role": "system", "content": "...put the final answer in \\boxed{}."},
    {"role": "user", "content": "Stefan goes to a restaurant..."}
  ],
  "target": "<think>\n...\n</think>\n\n\\boxed{108}",
  "metadata": {"answer": "108"}
}
```

`scripts/prepare_math_data.py` 负责清理 GSM8K solution、去除 `<<...>>` 计算标记，并生成统一 target。`scripts/train_sft.py` 再将样本拼成：

```text
<|im_start|>system
系统指令<|im_end|>
<|im_start|>user
题目<|im_end|>
<|im_start|>assistant
<think>
推理过程
</think>

\boxed{答案}<|im_end|>
```

`render_m1_generation_prompt()` 使用 `add_generation_prompt=False`，随后手工追加 assistant header。目标回答必须自己生成 `<think>`。这样可以避免 Qwen3 模板自动产生空 think block 后，target 又产生第二个 think block。

### 练习 S1：人工审核 SFT 数据

检查前 20 条 `sft_d5_500.jsonl`，至少记录以下问题：

1. target 是否都有完整 `<think>...</think>` 和位于其后的 `\boxed{}`。
2. `metadata.answer` 是否与 boxed answer 一致。
3. reasoning 中是否残留 `<<...>>`、`####` 或明显算术错误。
4. system prompt 是否完全一致。

可以先使用下面的只读脚本辅助抽样，但最终判断需要查看原文：

```bash
python - <<'PY'
from pathlib import Path
from scripts.train_sft import load_sft_dataset

path = Path("data/math/splits/sft_d5_500.jsonl")
for index, row in enumerate(load_sft_dataset(path)[:20], 1):
    target = row["target"]
    print(
        index,
        row["metadata"]["problem_id"],
        "think=", "<think>" in target and "</think>" in target,
        "boxed=", "\\boxed{" in target,
        "markers=", "<<" in target or "####" in target,
    )
PY
```

交付物：列出样本编号、判断、证据片段。若全部通过，也要说明检查了哪些字段。

## 2.3 token、label 与 loss mask

真实 debug dump 位于：

```bash
runs/000030_qwen3_0_6b_sft_gsm8k_500/logs/debug_batch.json
```

该样本有 96 个 prompt token、86 个 target token，共 182 个 token。`tokenize_sample()` 构造：

```text
full_ids = prompt_ids + target_ids + eos
```

`collate_batch()` 先把所有 label 初始化为 `-100`，随后只将 target 区间写入真实 token id：

```text
position       0 ... 95 | 96 ... 181
input_ids      prompt    | target + eos
labels         -100      | target + eos
loss           disabled  | enabled
```

causal LM 在位置 `t` 产生的 logits 用来预测位置 `t+1`。当前实现因此使用：

```python
logits = logits[:, :-1, :]
shifted_labels = labels[:, 1:]
```

mask 来自 `shifted_labels != -100`。第一个受监督预测发生在 input position 95：assistant header 的最后一个 token 预测 position 96 的 `<think>` token。

### 练习 S2：手算 next-token loss mask

假设 token 序列如下：

```text
位置          0       1        2          3         4
token       USER     Q      ASSIST      THINK      EOS
原始 label  -100    -100     -100       THINK      EOS
```

回答：

1. `logits[:, :-1]` 对应哪些 input position？
2. `labels[:, 1:]` 的值是什么？
3. 哪两个位置的预测进入 loss？
4. 若误用 `labels[:, :-1]`，模型会被要求学习什么错误关系？

### 练习 S3：审查真实边界

打开 debug dump，找到 prompt 与 target 的交界处。确认 position 95 和 96 附近满足以下条件：

- prompt 自身不受监督；
- position 95 的 logits 预测 `<think>`；
- `<think>` 之后的 target token 持续受监督；
- EOS 属于 target。

建议命令：

```bash
python - <<'PY'
import json

p = "runs/000030_qwen3_0_6b_sft_gsm8k_500/logs/debug_batch.json"
d = json.load(open(p))
for row in d["shift_check"]:
    if 92 <= row["input_position"] <= 100:
        print(row)
print(d["loss_mask_summary"])
PY
```

交付物：写出第一个受监督 label 的位置、token 文本和判断依据。

## 2.4 LoRA 与训练循环

Polaris 在 `q_proj`、`k_proj`、`v_proj`、`o_proj` 上应用 LoRA，配置为 `r=32, alpha=32`。基础权重保持冻结，checkpoint 只保存 adapter 权重。训练后会重新加载基础模型、重建 LoRA 结构、加载 adapter，并再次生成 smoke sample。

正式 run 为：

```text
runs/000030_qwen3_0_6b_sft_gsm8k_500
```

已记录结果为 500 条训练数据、375 optimizer steps，train loss 从 1.108333 降到 0.032215，无 NaN/Inf。

这组数字只能说明代码执行并拟合了训练目标。目标是否正确还要结合 debug dump、checkpoint reload、生成结果和独立评估。

### 练习 S4：代码审查——实际遍历了几次数据

阅读 `scripts/train_sft.py` 中 `steps_per_epoch`、`effective_steps` 和梯度累积循环：

```bash
nl -ba scripts/train_sft.py | sed -n '1102,1152p'
```

配置为 500 条数据、`batch_size=4`、`gradient_accumulation_steps=4`、`num_epochs=3`。

请计算：

1. 程序得到多少个 optimizer steps？
2. 每个 optimizer step 消耗多少条样本？
3. 整个训练循环总共消费多少个样本位置？
4. 约等于完整遍历数据多少次？
5. 配置中的 `num_epochs=3` 是否准确描述实际训练量？

这是代码审核题。先根据循环推导，避免沿用里程碑文档中的表述。

## 2.5 评估协议

Polaris 拆分五个指标：

| 指标                        | 回答的问题                                  |
| ------------------------- | -------------------------------------- |
| pass\@1                   | 最终答案是否正确                               |
| answer extraction success | 能否抽取出候选答案                              |
| format adherence          | 是否存在完整 think block，且 boxed answer 位于其后 |
| invalid output rate       | 是否无法形成有效输出                             |
| average completion length | 输出长度是否异常变化                             |

`extract_predicted_answer()` 只在 `</think>` 之后查找答案，顺序为 boxed、answer tag、`####`、最后一个数字。`answers_match()` 会去除逗号和空格，并支持整数、小数、普通分数与 LaTeX 分数的数值等价。

训练和评估必须共享同一 system prompt。项目曾经使用较短的评估 prompt，导致模型没有按训练格式生成 boxed answer，format adherence 被错误评为 0。

### 练习 S5：人工判分

参考答案为 `3/4`。分别判断抽取结果、抽取方法、格式遵循和答案正确性：

```text
A. <think>1/2 + 1/4 = 3/4</think> \boxed{\frac{3}{4}}
B. <think>1/2 + 1/4 = 3/4</think> The answer is 0.75
C. \boxed{3/4}<think>calculation</think>
D. <think>The answer may be 0.70</think> I choose 0.75
E. <think>3/4</think> \boxed{0.70}
```

注意：答案正确性和格式遵循是两个独立维度。

### 练习 S6：解释两组看似冲突的指标

仓库中存在两组 SFT 指标：

```text
run 内部评估：10 条，pass@1=0.80，format adherence=0.90
统一对比评估：50 条，pass@1=0.28，format adherence=1.00
```

检查以下文件后，说明为什么不能写成“模型准确率从 0.80 降到 0.28”：

```bash
cat runs/000030_qwen3_0_6b_sft_gsm8k_500/metrics/eval_summary.json
cat runs/sft_50_eval/eval_summary.json
cat runs/baseline_50_eval/eval_summary.json
```

写结论时必须明确模型、adapter、数据集、样本数、prompt、生成参数和 evaluator。

## 2.6 M1 中真正需要掌握的内容

完成 SFT 阶段后，应能独立回答以下问题：

1. 一条原始 GSM8K 样本如何变成 prompt 和 target？
2. chat template 最终生成了哪些特殊 token？
3. 第一枚受监督 token 是什么，谁预测它？
4. prompt mask、padding mask 和 next-token shift 如何共同决定 loss？
5. LoRA adapter 如何保存和重新加载？
6. loss 下降后还要检查哪些证据？
7. baseline 与 SFT 的评估如何保证同口径？
8. 何时可以宣称格式改善，何时可以讨论解题能力改善？

如果其中任一问题只能凭记忆回答，回到对应代码和 run 文件重新验证。

# 3. DPO：从模仿标准答案到学习相对偏好

## 3.0 本项目的 DPO 学习目标

这一部分的目标是完成一个可审计的最小 DPO 闭环，为后续 GRPO 做准备。你需要能亲手检查一条 pair 如何进入 loss，确认 policy 与冻结 reference 的来源，并判断一次训练日志是否支持结论。完成这些内容后即可进入 GRPO；无需在 Polaris 中系统覆盖 DPO 的所有变体或做大规模超参数搜索。

| 必须掌握 | 用仓库中的什么证据验证 |
| --- | --- |
| 偏好数据 | 审阅 `dpo_v1.jsonl` 的 chosen、rejected、质量标签和长度差；解释偏好来自答案正确性还是格式、长度捷径。 |
| 序列概率 | 追踪 prompt/response 边界、next-token shift 和 response mask；手算 chosen/rejected 的序列 logprob 与 margin。 |
| DPO 目标 | 区分 policy margin、reference margin、DPO margin、`beta` 和 preference accuracy；解释初始两模型相同时为何 loss 约为 `ln(2)`。 |
| 权重边界 | 复核 policy 与 reference 都加载 M1 SFT adapter，reference 冻结；从 `checkpoint_provenance.json` 和 frozen config 复现该判断。 |
| 实验判断 | 看 debug dump、训练指标、同口径评估和 sample diff；将“训练可运行”与“偏好学习有效”分开。 |

与后续方法的关系为：SFT 建立基本行为，DPO 用离线的 chosen/rejected 对调整相对偏好，GRPO 用当前 policy 在线采样的一组答案及可验证 reward 更新，OPD 依赖更复杂的在线 teacher/student rollout 与分布蒸馏。DPO 的价值在于先掌握 pair、logprob、reference 和诊断这组共同基础；GRPO 才是下一阶段需要深入实现的重点。

建议按 D1、D3、D4、D5、D6、D7 的顺序学习并在仓库内核验。D2 用于审计数据构造规则，完成一次即可。每题的答案都应附上具体 sample id、文件路径或日志字段，避免只写理论定义。

## 3.1 SFT 为什么还需要 DPO

SFT 对每个 prompt 提供一个目标回答。它无法直接表达“两个都能生成的回答中，哪个更好”。DPO 数据为同一 prompt 提供 chosen 和 rejected，训练信号来自相对偏好。

在 Polaris 中，DPO 的第一目标是验证偏好数据、response-only logprob、policy/reference 对照和实验诊断。小样本 pass\@1 提升属于观察结果，不能作为首要验收条件。

DPO 必须从有效的 M1 SFT policy 出发。这样 policy 已具备目标格式和基本数学行为，DPO 再调整回答间的相对概率。

## 3.2 一条真实 DPO pair

第一条 `dpo_v1.jsonl` 使用同一个 prompt，chosen 的答案为 3，rejected 的答案为 4：

```json
{
  "pair_type": "gold_vs_sft_wrong",
  "quality_tag": "clean",
  "chosen": "<think>...2/2=1...2+1=3...</think>\\boxed{3}",
  "rejected": "<think>...2/1=2...2+2=4...</think>\\boxed{4}",
  "metadata": {
    "chosen_answer_correct": true,
    "rejected_answer_correct": false,
    "chosen_token_length": 27,
    "rejected_token_length": 37
  }
}
```

当前数据报告：329 组 pairs，其中 258 组标为 clean，71 组标为 length biased；chosen 答案正确率 1.00，rejected 答案正确率 0.00；chosen 平均长度 78.9，rejected 平均长度 84.1。另有 76 条原始构造记录因 extreme length 在 candidate 形成前被过滤。

### 练习 D1：审阅真实 pairs

从以下三类中各选 5 条，共审阅 15 条：

- `quality_tag=clean`；
- `quality_tag=length_biased`；
- chosen 与 rejected 格式不一致。

记录：答案方向、推理质量、长度差、格式差、是否值得训练、建议标签。可使用：

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
from polaris.json_records import load_json_record_stream

rows = load_json_record_stream(Path("data/math/splits/dpo_v1.jsonl"))
groups = {
    "clean": [r for r in rows if r["quality_tag"] == "clean"],
    "length_biased": [r for r in rows if r["quality_tag"] == "length_biased"],
    "format_gap": [r for r in rows if r["metadata"]["chosen_format_adherence"] != r["metadata"]["rejected_format_adherence"]],
}
for name, group in groups.items():
    print("\n###", name, len(group))
    for row in group[:5]:
        m = row["metadata"]
        print(row["id"], m["problem_id"], m["chosen_token_length"], m["rejected_token_length"])
        print("Q:", row["messages"][-1]["content"])
        print("CHOSEN:", row["chosen"])
        print("REJECTED:", row["rejected"])
PY
```

重点判断：模型会把“答案正确”与“更长、更短、更规范”中的哪种相关性当作捷径。

### 练习 D2：审查数据报告

阅读 `data/math/reports/dpo_v1_report.json`，回答：

1. `filter_breakdown.extreme_length=76` 与 `filtered_pairs=0` 为什么能够同时成立？
2. `total_candidates` 的统计发生在 hard filter 之前还是之后？
3. 71 条 `length_biased` 与 76 条 extreme length 的差异来自什么规则？
4. 当前报告缺少哪些足以复现筛选决策的信息？

随后阅读 `scripts/prepare_dpo_data.py`，用代码确认推测。

## 3.3 DPO 公式

对 prompt `x`、chosen `y_c`、rejected `y_r`，先计算 policy margin：

$$
    \Delta_\pi = \log\pi_\theta(y_c|x)-\log\pi_\theta(y_r|x)
$$

reference margin 为：

$$
\Delta_{ref}=\log\pi_{ref}(y_c|x)-\log\pi_{ref}(y_r|x)
$$

DPO margin 为：

$$
m=\Delta_\pi-\Delta_{ref}
$$

loss 为：

$$
L_{DPO}=-\log\sigma(\beta m)
$$

训练关注 policy 相对 reference 的变化。chosen 的绝对 logprob 低于 rejected 仍然可能产生正确更新，只要 policy 相对 reference 更偏向 chosen。

`beta` 控制 margin 进入 sigmoid 前的缩放。它会改变梯度尺度和 policy 偏离 reference 的强度，需要与数据质量、学习率和训练步数共同解释。

### 练习 D3：手算 DPO

给定：

```text
policy chosen logprob   = -1.2
policy rejected logprob = -1.6
ref chosen logprob      = -1.0
ref rejected logprob    = -1.1
beta                    = 0.1
```

计算 policy margin、reference margin、DPO margin、preference accuracy，并估算 DPO loss。再回答：policy 更喜欢 chosen，为什么 DPO margin 仍可能为负？

第二组：

```text
policy chosen logprob   = -2.0
policy rejected logprob = -1.5
ref chosen logprob      = -2.4
ref rejected logprob    = -1.4
```

计算后判断这一步相对 reference 是否朝 chosen 方向移动。

## 3.4 response-only logprob

`tokenize_pair()` 分别构造：

```text
prompt + chosen + eos
prompt + rejected + eos
```

两条序列共享 prompt。response mask 在 prompt 区间为 0，在 response 与 EOS 区间为 1。`compute_response_logprob()` 对 logits、ids 和 mask 同时做 next-token shift，然后只汇总 response token。

当前实现将每条 response 的 token logprob 求和。因此日志里的 `chosen_logprob` 和 `rejected_logprob` 是序列 logprob；长度平衡通过数据审阅和报告诊断处理，不能在 loss 内静默改成平均值。

### 练习 D4：比较求和与长度归一化

假设 chosen 有 2 个 token，每个 token logprob 为 `-0.5`；rejected 有 4 个 token，每个 token logprob 为 `-0.4`。

分别计算：

1. 使用 token logprob 总和的 chosen/rejected margin。
2. 使用平均 token logprob 的 chosen/rejected margin。
3. 两种定义会得出怎样不同的偏好方向？
4. 标准 DPO 目标使用哪一种？当前实现为何选择求和？改成平均值会改变什么？

这项选择必须在实验报告中明确，不能只称为“logprob”。

## 3.5 policy 与 reference 的初始化

正确的 M2 数据流应为：

```text
Base model + M1 SFT adapter
    ├── policy：加载 SFT 状态，加入可训练的 DPO adapter 或继续训练明确的参数
    └── reference：冻结的明确快照
```

reference 可以选择冻结的 SFT checkpoint 或明确记录的 base checkpoint。两者对应不同基准，必须写进 frozen config 和 run report。policy 必须从 M1 的有效 SFT 状态开始。

### 练习 D5：代码审查——checkpoint 是否真的加载

对照以下文件：

```bash
cat configs/qwen3_0_6b/dpo_math.yaml
nl -ba scripts/train_dpo.py | sed -n '520,650p'
```

回答：

1. `model.path`、`dpo.policy_adapter_path`、`dpo.ref_model_path`、`dpo.ref_adapter_path` 分别承担什么职责？
2. policy 加载 adapter 前为何必须先由 base model 建立相同的 LoRA 结构？
3. reference 加载后在哪一行冻结，`checkpoint_provenance.json` 记录了哪些可复核信息？
4. 为什么 policy 和 reference 应在 M2 开始时使用同一份 M1 adapter？
5. 旧 run `000032` 的 policy/reference 实际来自哪里？它为何不能作为 M2 效果结论？
6. 新 sanity 的初始 `dpo_margin` 接近零能支持什么判断，不能支持什么判断？

交付物：画出当前代码的真实加载路径，再画出目标加载路径。

## 3.6 当前 10-step sanity 能证明什么

run `000032_qwen3_0_6b_dpo_math` 已产生 debug dump、10 步指标和 adapter checkpoint。第一步与第十步为：

```text
step 1:  loss=0.693147, dpo_margin=-0.003697, preference_accuracy=0.0
step 10: loss=0.693350, dpo_margin= 0.001759, preference_accuracy=0.5
```

这证明模型能前向计算、反向传播、更新 LoRA 参数并保存 checkpoint。10 步中 margin 接近零且持续波动，当前证据无法支持稳定偏好学习、效果改善或 M2 通过。

### 练习 D6：审查 debug dump 的可用性

打开：

```bash
runs/000032_qwen3_0_6b_dpo_math/logs/debug_dpo_batch.json
```

检查 `chosen_token_ids` 和 `rejected_token_ids`。两者展示的前 50 个 token 完全一致。结合 `prompt_token_count=58` 回答：

1. 这是否说明 chosen 与 rejected tokenization 相同？
2. 当前 dump 是否展示了任何 response token id？
3. 怎样修改 dump 才能直接审查 prompt/response 边界？
4. `shift_check` 只展示前 20 个位置时，能否验证 response mask 的第一处边界？

这道题用于判断“生成了 debug 文件”和“debug 文件足以审计关键逻辑”之间的差距。

### 练习 D7：代码审查——训练量

阅读 `build_microbatch_groups()` 和梯度累积循环。当前有 329 组 pairs，`batch_size=2`、`gradient_accumulation_steps=4`、`num_epochs=1`。

回答：若未传 `--max-steps`，程序会得到多少 optimizer steps，最后一步包含几个 micro-batch、几个 pair？总计遍历数据多少次？`num_epochs` 如何进入计算？

## 3.7 DPO 阶段需要掌握的内容

进入 M2.5 前，应能独立回答：

1. chosen/rejected 的偏好方向如何建立和人工审阅？
2. 长度、格式和答案正确性如何形成混杂信号？
3. response mask 的第一枚受监督 token 在哪里？
4. policy margin、reference margin 和 DPO margin 各自表示什么？
5. 为什么初始 policy 与 reference 相同时 DPO margin 为零、loss 接近 `ln(2)`？
6. policy 与 reference 分别加载了哪些权重，是否冻结？
7. preference accuracy 是否可能被 batch 波动或长度偏差误导？
8. 正式 DPO 需要哪些训练、评估和 sample diff 证据？

# 4. 当前恢复路线

DPO 已收口，当前不再执行新的 DPO 效果训练。按以下顺序进入 M2.5：

1. 阅读 `docs/experiment_notes/m2_dpo_closeout.md`，复核 v1 链路证据、style 对照、分支点诊断和归因边界。
2. 完成 GRPO 的 G0 作业，手算 reward 与 group-relative advantage，先固定可复算的 Math reward 协议。
3. 实现并测试 rollout、reward、response-only logprob、reference KL、checkpoint 和 resume；对全对、全错、空输出、无法解析和超长输出注入失败。
4. 执行真实小样本 M2.5 sanity，人工复算一组 reward，并检查 reward 方差、有效 advantage、KL、entropy、长度和无效输出率。
5. M2.5 通过后才启动 M3 GRPO；正式训练继续使用固定评测集和样例盲审。

M2 提供的核心结论是：训练 loss 或 teacher-forced token 概率变化不足以证明自由生成行为已改变。M2.5 必须将同样的审计强度施加到在线 rollout 和 reward 上。

# 5. 参考答案

建议完成练习后再展开本节。

<details>
<summary>S2：next-token loss mask</summary>

`logits[:, :-1]` 对应 input position 0–3，`labels[:, 1:]` 为 `[-100, -100, THINK, EOS]`。position 2 的 logits 预测 THINK，position 3 的 logits 预测 EOS，这两项进入 loss。若使用 `labels[:, :-1]`，同一位置 logits 会对齐同一位置 token，形成 self-token prediction，模型学习错误的时序关系。

</details>

<details>
<summary>S3：真实边界</summary>

prompt 长度为 96，因此 target 从 position 96 开始。shift 后，input position 95 预测 label position 96 的 `<think>`，该项应启用 loss。position 94 预测的 label position 95 仍属于 prompt，应被 mask。EOS 位于 target 尾部并参与监督。

</details>

<details>
<summary>S4：SFT 实际训练量</summary>

`steps_per_epoch=floor(500/4)=125`，`effective_steps=3×125=375`。每个 optimizer step 执行 4 个 micro-batch，每个 micro-batch 4 条，因此消费 16 个样本位置。总消费量为 `375×16=6000`，约等于遍历 500 条数据 12 次。当前 `num_epochs=3` 没有准确描述实际数据遍历次数；计算 optimizer steps 时没有除以梯度累积步数。

</details>

<details>
<summary>S5：人工判分</summary>

A：抽取 `\frac{3}{4}`，方法 boxed，格式通过，答案正确。B：抽取 `0.75`，方法 numeric fallback，格式不通过，答案正确。C：`</think>` 位于 boxed 之后，post-think 区域没有答案，抽取失败，格式不通过，答案错误。D：post-think 区域抽取最后一个数字 `0.75`，格式不通过，答案正确。E：抽取 `0.70`，方法 boxed，格式通过，答案错误。

</details>

<details>
<summary>S6：不同评估不能直接比较</summary>

0.80 来自正式 run 内部的 10 条评估；0.28 来自独立目录中的 50 条统一对比评估。样本集合和规模不同，可能还涉及运行入口与生成配置差异。可信的阶段比较应使用同一 50 条测试数据：baseline 0.20，SFT 0.28；format adherence 从 0.06 到 1.00。

</details>

<details>
<summary>D2：数据报告风险</summary>

76 条 extreme length 在 candidate append 之前被 hard filter 排除。`total_candidates=329` 已经是第一阶段过滤后的数量；`filtered_pairs=0` 只计算 329 个 candidates 到 329 个最终 pairs 的第二阶段差值，因此两者可以同时成立，但字段名没有清楚表达不同分母。71 条 `length_biased` 通过 hard threshold，同时落在更窄的 clean ratio 区间之外。报告还应分别记录原始输入量、第一阶段和第二阶段过滤量、两套长度阈值、每条过滤原因、过滤前后 id、随机种子和构造代码版本。

</details>

<details>
<summary>D3：DPO 手算</summary>

第一组 policy margin 为 `0.4`，reference margin 为 `0.1`，DPO margin 为 `0.3`，preference accuracy 为 1。loss 为 `-log(sigmoid(0.03))`，约 0.678。若 policy margin 虽为正但小于 reference margin，DPO margin 仍为负，说明 policy 相对 reference 减弱了 chosen 偏好。

第二组 policy margin 为 `-0.5`，reference margin 为 `-1.0`，DPO margin 为 `0.5`。policy 的绝对分布仍偏向 rejected，但相对 reference 已朝 chosen 移动。

</details>

<details>
<summary>D4：求和与长度归一化</summary>

总和定义下 chosen 为 `-1.0`，rejected 为 `-1.6`，margin 为 `0.6`，偏向 chosen。平均定义下 chosen 为 `-0.5`，rejected 为 `-0.4`，margin 为 `-0.1`，偏向 rejected。标准序列 DPO 使用 response token logprob 总和，当前实现也记录 `response_logprob_reduction=sum`。平均值会削弱长度对总 logprob 的机械影响，同时改变优化目标和偏好方向；若采用它，必须显式命名并重新验证数据构造。

</details>

<details>
<summary>D5：当前与旧版 checkpoint 加载路径</summary>

当前配置中 `model.path` 与 `dpo.ref_model_path` 都指向 `models/qwen3_0_6b/mlx`；`dpo.policy_adapter_path` 与 `dpo.ref_adapter_path` 都指向 `runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final`。训练先用 base model 建立 LoRA 层，再通过 `load_weights(..., strict=False)` 加载 M1 adapter；reference 随后执行 `freeze()`。run 的 `checkpoint_provenance.json` 会记录四个路径、reference 的冻结状态和 logprob 聚合方式。

旧 run `000032` 的 policy 和 reference 都从 base model 加载。policy 随后加入新 LoRA，reference 冻结；它没有从 M1 SFT 状态开始，不能作为当前 M2 效果结论。相同初始化时初始 DPO margin 接近零只证明两条加载路径相同，不能证明 pair 质量或偏好学习效果。

</details>

<details>
<summary>D6：debug dump 的审计缺口</summary>

前 50 个 token 全部位于 58-token prompt 内，两组相同属于预期，无法说明 response tokenization 是否一致。dump 应展示交界点前后窗口，例如 `prompt_len-5:prompt_len+20`，并分别输出 token、mask 和 shift 后 label。当前 shift check 只展示前 20 个 prompt 位置，无法验证 response mask 边界。

</details>

<details>
<summary>D7：DPO 实际训练量</summary>

329 条数据按 `batch_size=2` 切成 165 个 micro-batch；每 4 个组成一个 optimizer update，所以一轮为 42 步。前 41 步各有 4 个 micro-batch、8 条 pair，最后一步有 1 个 micro-batch、1 条 pair。总计正好消费 329 条，遍历一次。`num_epochs` 通过重复完整的 epoch group 列表进入计算；若设为 2，则得到 84 步和两次完整遍历。

</details>

## 6. 证据索引

| 内容                      | 文件                                            |
| ----------------------- | --------------------------------------------- |
| M1 验收结论                 | `docs/Project_todo/M1.md`                     |
| M1 周复盘                  | `docs/weekly_report/W1.md`                    |
| SFT 数据构造                | `scripts/prepare_math_data.py`                |
| SFT token、loss、LoRA 与训练 | `scripts/train_sft.py`                        |
| 数学答案抽取与判分               | `scripts/eval_math.py`                        |
| GSM8K 生成评估              | `scripts/smoke/eval_gsm8k.py`                 |
| M1 正式 run               | `runs/000030_qwen3_0_6b_sft_gsm8k_500`        |
| M2 计划与验收标准              | `docs/Project_todo/M2.md`                     |
| DPO pair 协议             | `docs/format_notes/dpo_math_pair_protocol.md` |
| DPO 数据报告                | `data/math/reports/dpo_v1_report.json`        |
| DPO 训练实现                | `scripts/train_dpo.py`                        |
| DPO 10-step run         | `runs/000032_qwen3_0_6b_dpo_math`             |
