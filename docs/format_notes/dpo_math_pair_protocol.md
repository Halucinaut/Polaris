# DPO Math Pair Protocol

## 1. 文档目的

本文档定义 M2 阶段 DPO 训练数据的 chosen/rejected pair 构造协议。所有 pair 必须遵循本协议，确保训练信号方向明确、噪声可控、可人工审阅。

---

## 2. 基本定义

### 2.1 Pair 结构

一个 DPO pair 包含：
- **prompt**：用户问题（system + user messages）
- **chosen**：偏好响应（模型应学习的输出）
- **rejected**：非偏好响应（模型应避免的输出）
- **answer**：标准答案（用于判分）
- **pair_type**：pair 来源类型
- **quality_tag**：质量标签

### 2.2 格式要求

chosen 和 rejected 必须遵循 M1 SFT 格式协议：
```
<think>
{reasoning}
</think>
\boxed{{{answer}}}
```

- 必须包含 `<think>` 和 `</think>` 标签
- 最终答案必须在 `\boxed{}` 中
- 格式由配置决定，不硬编码

---

## 3. Chosen 合格标准

chosen 响应必须满足以下条件：

| 条件 | 说明 |
|------|------|
| 答案正确 | extracted_answer 与标准答案匹配 |
| 格式遵循 | 包含完整的 `<think>...</think>\n\n\boxed{}` |
| 推理清晰 | reasoning 过程可读、逻辑连贯 |
| 无截断 | 输出完整，不是残缺文本 |
| 长度合理 | 不极端过长（> 2000 tokens） |

---

## 4. Rejected 合格标准

rejected 响应必须满足以下条件：

| 条件 | 说明 |
|------|------|
| 答案错误 | extracted_answer 与标准答案不匹配 |
| 或格式不遵循 | 缺少 `<think>` 标签或 `\boxed{}` |
| 有实质内容 | 不是空输出或纯噪声 |
| 方向明确 | 与 chosen 的差异可被明确解释 |

---

## 5. 允许的 Pair Type

| pair_type | 说明 | 优先级 |
|-----------|------|--------|
| `gold_vs_wrong_model_output` | 标准答案 vs 模型错误输出 | 高 |
| `correct_boxed_vs_wrong_boxed` | 正确 boxed 答案 vs 错误 boxed 答案 | 高 |
| `sft_correct_vs_baseline_wrong` | SFT 正确输出 vs baseline 错误输出 | 高 |
| `sft_improved_vs_sft_regressed` | SFT 改进的输出 vs SFT 退化的输出 | 中 |
| `correct_formatted_vs_unformatted_correct` | 格式正确的正确答案 vs 格式错误的正确答案 | 低 |

**M2 优先级**：answer-correctness pair 优先于 format-only pair。

---

## 6. 禁止进入训练的数据类型

以下样本必须过滤，不能进入 dpo_v1.jsonl：

| 类型 | 说明 |
|------|------|
| 内容几乎相同 | chosen 和 rejected 只有个别词差异 |
| 只有长度差异 | 偏好信号仅来自长度，无实质内容差异 |
| 都答案错误 | chosen 和 rejected 都答错，无法区分优劣 |
| 都答案正确且无法区分 | 两者都对，但解释质量无法客观判断 |
| rejected 为空 | rejected 是空字符串或纯空白 |
| 截断残缺 | rejected 是截断导致的不完整文本 |
| 极端长度差 | chosen 极长（> 1500 tokens）而 rejected 极短（< 50 tokens） |
| 方向需主观猜测 | 无法明确判断哪个更好 |

---

## 7. 长度偏差控制规则

长度偏差是 DPO 的常见问题，必须显式控制：

### 7.1 长度统计

每对 pair 必须记录：
- `chosen_length`：chosen 的 token 数
- `rejected_length`：rejected 的 token 数
- `length_ratio`：chosen_length / rejected_length

### 7.2 过滤规则

| 规则 | 阈值 | 处理 |
|------|------|------|
| 极端长度差 | length_ratio > 3.0 或 < 0.33 | 过滤 |
| chosen 过长 | chosen_length > 1500 | 过滤或截断 |
| rejected 过短 | rejected_length < 30 | 过滤 |

### 7.3 平衡要求

- dpo_v1_report.json 必须报告 `avg_length_gap` 和 `avg_length_ratio`
- 平均 length_ratio 应在 0.7-1.5 之间
- 如果 length_bias 明显，需要增加 length-balanced pair

---

## 8. 格式偏差控制规则

### 8.1 格式一致性

- chosen 和 rejected 应使用相同的 chat template
- chosen 和 rejected 应使用相同的 system prompt
- 格式差异不应成为唯一偏好信号

### 8.2 Format-only pair 限制

- M2 允许少量 format-only pair（correct_formatted_vs_unformatted_correct）
- format-only pair 占比不超过 20%
- 主导数据必须是 answer-correctness pair

---

## 9. 人工审阅标准

### 9.1 审阅任务

人工必须审阅至少 20 组候选 pair，每组标注：

| 标签 | 说明 |
|------|------|
| `clean` | pair 方向明确，偏好信号清晰 |
| `weak` | pair 方向可判断，但信号较弱 |
| `length-biased` | 偏好信号主要来自长度差异 |
| `format-only` | 偏好信号主要来自格式差异 |
| `invalid` | pair 无效，不能进入训练 |

### 9.2 审阅重点

- 答案正确性是否确实是偏好信号来源
- reasoning 质量是否有实质差异
- 长度差异是否掩盖了实质偏好
- 是否存在 chosen 不如 rejected 的情况

### 9.3 合格标准

- clean 或 high_confidence pair 占主要比例（> 60%）
- invalid pair 被过滤
- length-biased pair 被标记或过滤

---

## 10. Pair 构造与 M1 SFT 格式协议的关系

### 10.1 格式继承

- DPO pair 的 chosen/rejected 必须遵循 M1 的 target_template
- 不能引入新的格式变体
- eval_math.py 的 format checker 必须能正确评估 DPO 输出

### 10.2 Prompt 一致性

- DPO 的 system prompt 必须与 M1 SFT 一致
- DPO 的 user prompt 格式必须与 M1 SFT 一致
- 避免 prompt 不一致导致的虚假偏好信号

### 10.3 评估一致性

- DPO 后评估必须使用与 M1 相同的 test set
- DPO 后评估必须使用与 M1 相同的 system prompt
- DPO 后评估必须使用与 M1 相同的 eval_math.py

---

## 11. 数据集规模要求

| 阶段 | 最小规模 | 推荐规模 |
|------|----------|----------|
| DPO sanity | 20-50 pairs | 50 pairs |
| 完整 DPO | 200 pairs | 300-500 pairs |

如果最终只能构造少量高质量 pair，可以先跑小规模 DPO sanity，但不能宣称完成完整 M2。

---

## 12. dpo_v1.jsonl Schema

### 12.1 字段定义

| 字段 | 类型 | 必填 | 说明 | 约束 |
|------|------|------|------|------|
| `id` | string | 是 | 唯一标识符 | 格式：`dpo_{6位数字}`，如 `dpo_000001` |
| `messages` | array | 是 | 对话上下文（system + user） | 必须包含 system 和 user 各一条；chosen/rejected 共享同一个 messages |
| `chosen` | string | 是 | 偏好响应（assistant） | 只包含 assistant response，不包含 prompt；必须符合 target_template 格式 |
| `rejected` | string | 是 | 非偏好响应（assistant） | 只包含 assistant response，不包含 prompt；必须符合 target_template 格式 |
| `answer` | string | 是 | 标准答案 | 用于自动判分和人工审阅，不代表 chosen 文本中唯一允许的内容 |
| `source` | string | 是 | 数据来源 | 当前仅允许 `gsm8k` |
| `pair_type` | string | 是 | pair 来源类型 | 见 12.2 允许值 |
| `quality_tag` | string | 是 | 质量标签 | 见 12.3 允许值 |
| `metadata` | object | 是 | 元数据 | 见 12.4 子字段定义 |

### 12.2 pair_type 允许值

| 值 | 说明 |
|------|------|
| `gold_vs_wrong_model_output` | 标准答案 vs 模型错误输出 |
| `correct_boxed_vs_wrong_boxed` | 正确 boxed 答案 vs 错误 boxed 答案 |
| `correct_formatted_vs_unformatted_correct` | 格式正确的正确答案 vs 格式错误的正确答案 |
| `sft_correct_vs_baseline_wrong` | SFT 正确输出 vs baseline 错误输出 |

### 12.3 quality_tag 允许值

| 值 | 说明 |
|------|------|
| `clean` | pair 方向明确，偏好信号清晰 |
| `weak` | pair 方向可判断，但信号较弱 |
| `length_biased` | 偏好信号主要来自长度差异 |
| `format_only` | 偏好信号主要来自格式差异 |
| `invalid` | pair 无效，不能进入训练 |

### 12.4 metadata 子字段定义

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `chosen_answer_correct` | bool | 是 | chosen 答案是否正确 |
| `rejected_answer_correct` | bool | 是 | rejected 答案是否正确 |
| `chosen_format_adherence` | bool | 是 | chosen 是否遵循格式协议 |
| `rejected_format_adherence` | bool | 是 | rejected 是否遵循格式协议 |
| `chosen_length` | int | 是 | chosen 的 token 数 |
| `rejected_length` | int | 是 | rejected 的 token 数 |
| `length_gap` | int | 是 | chosen_length - rejected_length |
| `length_ratio` | float | 是 | chosen_length / rejected_length（保留 2 位小数） |
| `chosen_extraction_method` | string | 是 | chosen 答案抽取方法（如 `boxed`, `answer_tag`, `numeric`） |
| `rejected_extraction_method` | string | 是 | rejected 答案抽取方法 |

### 12.5 合法 JSONL 示例

```json
{
  "id": "dpo_000001",
  "messages": [
    {"role": "system", "content": "You are a helpful math assistant. Solve the problem and put the final answer in \boxed{}."},
    {"role": "user", "content": "A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he receive?"}
  ],
  "chosen": "<think>\nJohn buys 5 apples at $2 each, so he spends 5 × 2 = $10.\nHe pays with a $20 bill.\nHis change is $20 - $10 = $10.\n</think>\n\\boxed{10}",
  "rejected": "<think>\nJohn buys 5 apples at $2 each.\n5 × 2 = 10.\nHe pays with $20.\n20 - 10 = 8.\n</think>\n\\boxed{8}",
  "answer": "10",
  "source": "gsm8k",
  "pair_type": "gold_vs_wrong_model_output",
  "quality_tag": "clean",
  "metadata": {
    "chosen_answer_correct": true,
    "rejected_answer_correct": false,
    "chosen_format_adherence": true,
    "rejected_format_adherence": true,
    "chosen_length": 65,
    "rejected_length": 42,
    "length_gap": 23,
    "length_ratio": 1.55,
    "chosen_extraction_method": "boxed",
    "rejected_extraction_method": "boxed"
  }
}
```

### 12.6 Schema Validation 最小规则

构造 dpo_v1.jsonl 后，必须验证以下规则：

| 规则 | 说明 | 失败处理 |
|------|------|----------|
| 必填字段存在 | 9 个顶层字段 + 10 个 metadata 子字段必须存在 | 报错，不进入训练 |
| id 唯一 | 每条记录的 id 在数据集中唯一 | 报错，去重 |
| messages 格式 | 必须是 array，包含 system 和 user 各一条 | 报错，过滤 |
| chosen/rejected 格式 | 必须是 string，不为空 | 报错，过滤 |
| chosen/rejected 不含 prompt | 不能包含 `<|im_start|>system` 或 `<|im_start|>user` | 报错，过滤 |
| chosen/rejected 符合 target_template | 必须包含 `<think>` 和 `</think>`，答案在 `\boxed{}` 中 | 警告，标记 |
| pair_type 合法 | 必须是 4 个允许值之一 | 报错，过滤 |
| quality_tag 合法 | 必须是 5 个允许值之一 | 报错，过滤 |
| metadata 类型正确 | bool/int/float/string 类型匹配 | 报错，修复或过滤 |
| length_ratio 合理 | 在 0.33-3.0 之间（极端长度差过滤） | 警告，标记 length_biased |
| chosen_answer_correct 为 true | M2 要求 chosen 答案正确 | 报错，过滤 |
| rejected_answer_correct 为 false | M2 要求 rejected 答案错误（answer-correctness pair） | 警告，检查 pair_type |

### 12.7 与 dpo_v1_report.json 的关系

schema 中的字段直接服务于 dpo_v1_report.json 的统计：

| report 字段 | 来源 |
|-------------|------|
| total_pairs | 记录总数 |
| kept_pairs | quality_tag != "invalid" 的记录数 |
| filtered_pairs | total_pairs - kept_pairs |
| pair_type_distribution | 按 pair_type 分组计数 |
| quality_tag_distribution | 按 quality_tag 分组计数 |
| chosen_avg_length | metadata.chosen_length 平均值 |
| rejected_avg_length | metadata.rejected_length 平均值 |
| avg_length_gap | metadata.length_gap 平均值 |
| avg_length_ratio | metadata.length_ratio 平均值 |
| chosen_format_adherence | metadata.chosen_format_adherence 为 true 的比例 |
| rejected_format_adherence | metadata.rejected_format_adherence 为 true 的比例 |
| chosen_answer_correct_rate | metadata.chosen_answer_correct 为 true 的比例 |
| rejected_answer_correct_rate | metadata.rejected_answer_correct 为 true 的比例 |
| duplicate_prompt_count | messages 相同的记录数 |
| same_answer_pair_count | chosen_answer_correct == rejected_answer_correct 的记录数 |
| extreme_length_gap_count | length_ratio < 0.33 或 > 3.0 的记录数 |

---

## 13. dpo_v1_report.json 必需字段

| 字段 | 说明 |
|------|------|
| total_pairs | 总 pair 数 |
| kept_pairs | 保留的 pair 数 |
| filtered_pairs | 过滤的 pair 数 |
| pair_type_distribution | 各 pair_type 的数量 |
| quality_tag_distribution | 各 quality_tag 的数量 |
| chosen_avg_length | chosen 平均长度 |
| rejected_avg_length | rejected 平均长度 |
| avg_length_gap | 平均长度差 |
| avg_length_ratio | 平均长度比 |
| chosen_format_adherence | chosen 格式遵循率 |
| rejected_format_adherence | rejected 格式遵循率 |
| chosen_answer_correct_rate | chosen 答案正确率 |
| rejected_answer_correct_rate | rejected 答案正确率 |
| duplicate_prompt_count | 重复 prompt 数量 |
| same_answer_pair_count | 同答案 pair 数量 |
| extreme_length_gap_count | 极端长度差 pair 数量 |

---

## 14. 审阅清单

构造 pair 后，逐项检查：

- [ ] chosen 答案正确
- [ ] rejected 答案错误
- [ ] chosen/rejected 格式都遵循
- [ ] pair_type 正确标注
- [ ] quality_tag 正确标注
- [ ] 长度比在 0.33-3.0 之间
- [ ] 不是 content-duplicate pair
- [ ] 不是 length-only bias pair
- [ ] 人工审阅至少 20 组
- [ ] clean pair 占主要比例

---

## 15. 版本记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06-15 | 初始版本 |
