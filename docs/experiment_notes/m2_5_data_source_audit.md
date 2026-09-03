# M2.5 Data Source Audit

**日期**：2026-08-31（v2 修正版）
**阶段**：M2.5 GRPO 数据准备
**状态**：数据源审计完成；尚未构造最终训练 split；`math_level_3_5.jsonl` 未创建

---

## 1. 本地 JSONL 数据源清单

审计脚本：`scripts/data/audit_grpo_candidates.py [--json]`
测试：`tests/test_grpo_data_audit.py`（53 tests, all pass）

### 1.1 GRPO 直接兼容文件（全部记录满足 messages + metadata.answer）

| 文件 | 记录数 | valid | invalid | 来源 |
|------|-------|-------|---------|------|
| `splits/sft_v1.jsonl` | 300 | 300 | 0 | GSM8K train (M1 SFT v1) |
| `splits/sft_d5_500.jsonl` | 500 | 500 | 0 | GSM8K train d5 (M1 SFT 实际训练) |
| `splits/sft_v2_style_control_train_449.jsonl` | 449 | 449 | 0 | GSM8K train d5 (SFT style control) |

以上文件可直接作为 `train_grpo.py` 的 `data.path` 输入，无需转换。但三者均与排除集高度重合，不适合作为 GRPO 训练候选。

### 1.2 可经确定性转换后兼容的文件

| 文件 | 记录数 | 缺失字段 | 转换方式 |
|------|-------|----------|---------|
| `gsm8k/train.jsonl` | 7,473 | messages, metadata.answer | 包装 question 到 messages，从 `#### N` 提取 answer 到 metadata |
| `gsm8k/split/train_converted_d5_500.jsonl` | 500 | messages, metadata.answer | 包装 problem 到 messages，移 answer 到 metadata |
| `gsm8k/split/test_converted_500.jsonl` | 500 | messages, metadata.answer | 同上 |

### 1.3 不兼容且无可复用转换的文件

DPO 文件（`dpo_v1.jsonl` 等）全部缺少 `metadata.answer`（answer 在顶层），且包含 chosen/rejected 列，不适合 GRPO。问题集文件（pilots/problems）缺少 `messages` 字段。

---

## 2. 污染排除清单

排除集使用 `canonical_question_hash`（归一化问题文本的 SHA256 前 16 位）作为唯一标识。每条候选问题最多计数一次。

### 2.1 排除集构成

| 组 | 排除集 | 文件 | 问题数 |
|----|--------|------|-------|
| training | m1_sft_train | sft_d5_500.jsonl | 500 |
| training | dpo_v1_train | dpo_v1.jsonl | 329 |
| training | dpo_v2_style_train | dpo_v2_style_train_449.jsonl | 449 |
| training | dpo_v4_minimal_train | dpo_v4_minimal_449.jsonl | 449 |
| training | dpo_v4_minimal_pilot | dpo_v4_minimal_pilot_30.jsonl | 30 |
| training | boundary_only_dpo | boundary_only_dpo_480.jsonl | 480 |
| training | binary_prefix_dpo_ctrl | binary_prefix_dpo_control_480.jsonl | 480 |
| training | boundary_only_sft | boundary_only_sft_480.jsonl | 480 |
| eval | probe_30 | dpo_v2_style_train_probe_30.jsonl | 30 |
| eval | probe_30_eval | dpo_v2_style_train_probe_30_eval.jsonl | 30 |
| eval | stress_50 | dpo_v2_style_stress_50.jsonl | 50 |
| eval | stress_eval_50 | dpo_v2_style_stress_eval_50.jsonl | 50 |
| eval | gsm8k_50_eval_first50 | test_converted_500.jsonl 前 50 | 50 |
| pilots | rq_v2_problems_50 | dpo_rq_v2_problems_50.jsonl | 50 |
| pilots | rq_v2b_problems_100 | dpo_rq_v2b_problems_100.jsonl | 100 |
| pilots | rq_v2c_problems_100 | dpo_rq_v2c_problems_100.jsonl | 100 |
| pilots | rq_v3_problems | dpo_rq_v3/problems.jsonl | 250 |
| pilots | rq_v3b_problems | dpo_rq_v3_b/problems.jsonl | 500 |
| pilots | rq_problems_50 | dpo_reasoning_quality_problems_50.jsonl | 50 |

### 2.2 候选源审计结果（canonical_question_hash 去重）

| 候选源 | 问题数 | training | eval | pilots | union | clean |
|--------|-------|----------|------|--------|-------|-------|
| `gsm8k/train.jsonl` | 7,473 | 500 | 80 | 1,027 | **1,430** | **6,043** |
| `train_converted_d5_500` | 500 | 500 | 80 | 97 | 500 | 0 |
| `sft_v1.jsonl` | 300 | 300 | 47 | 58 | 300 | 0 |
| `sft_d5_500.jsonl` | 500 | 500 | 80 | 97 | 500 | 0 |
| `sft_v2_style_ctrl_449` | 449 | 449 | 30 | 96 | 449 | 0 |
| `test_converted_500` | 500 | 329 | 50 | 0 | 343 | 157 |

**关键发现**：
- 唯一有 clean remaining 的训练候选源是 `gsm8k/train.jsonl`：经全部排除集过滤后剩余 **6,043** 条
- 仅排除 d5_500（m1_sft_train）时剩余 **6,973** 条；加入全部排除集后进一步减少至 6,043
- `test_converted_500` 有 157 条 clean，但属于 test set，不适合训练
- 所有 SFT/DPO split 文件的 clean remaining 均为 0

---

## 3. `math_level_3_5` 设计含义

### 3.1 本地定义

唯一出处：`docs/Project_todo/polaris_final_design.md` 第 70 行：

> | GRPO | MATH Level 3-5 | answer correctness reward + format reward | 学 verifiable reward 下的策略优化 |

第 74 行补充：

> Math 线启动前必须做 baseline screening：抽样评估 MATH Level 2-5，在 0.6B 和 4B 上选择 baseline accuracy 约 20%-60% 的难度区间。

### 3.2 解读

- **数据源**：Hendrycks MATH 数据集（`EleutherAI/hendrycks_math`）
- **难度筛选**：Level 3、4、5（Hendrycks MATH 的 `level` 字段，整数 1-5）
- **设计意图**：选择 baseline accuracy 20-60% 的难度区间

### 3.3 本地状态

- `data/math/splits/math_level_3_5.jsonl` **不存在**
- `data/math/sources/` 目录仅有 `.gitkeep`，Hendrycks MATH **未下载**
- 无任何脚本可以生成 `math_level_3_5.jsonl`
- MATH Level 3-5 是 Polaris 设计文档中 GRPO 的原定数据源，当前未实现

---

## 4. 候选构造方案

### 方案 A：GSM8K train 全量排除后

- **数据来源**：`gsm8k/train.jsonl`（7,473 条）
- **排除后可用题数**：6,043（经全部排除集 canonical hash 去重后）
- **污染风险**：零。已排除所有已知训练、评估、pilot 问题
- **难度风险**：GSM8K 全部为 grade-school math，0.6B baseline 约 20-28%
- **是否适合 M2.5 audit**：是。数据量充足，schema 转换确定性可复现
- **评测集**：`test_converted_500.jsonl` 前 50 条（GSM8K-50），与训练集零重合

### 方案 B：Hendrycks MATH Level 3-5（需下载）

- **数据来源**：`EleutherAI/hendrycks_math`，筛选 `level` ∈ {3, 4, 5}
- **预计可用题数**：~7,500（需下载后确认）
- **污染风险**：极低（与 GSM8K 数据零重合）
- **难度风险**：中高。baseline accuracy 可能 < 20%，reward 过稀疏
- **是否适合 M2.5 audit**：有条件。需先下载、做 baseline screening
- **评测集**：从 MATH Level 3-5 中预留 200-500 条

### 方案 C：GSM8K + Hendrycks MATH Level 3 混合

- **数据来源**：GSM8K clean 6,043 + MATH Level 3 ~2,500
- **污染风险**：低（两个数据集天然不重合）
- **难度风险**：中。混合 easy-moderate (GSM8K) + moderate-hard (MATH L3)
- **是否适合 M2.5 audit**：需先下载 MATH 数据

---

## 5. 推荐方案

**推荐方案 A 作为 M2.5 首个 audit 候选。**

理由：
1. 零外部依赖，所有数据已本地可用
2. 经全部排除集过滤后仍有 6,043 条干净题目
3. schema 转换确定性可复现（复用 `prepare_math_data.py` 逻辑）
4. GSM8K baseline ~20-28%，在设计文档建议的 20-60% 区间内
5. GSM8K-50 eval 与训练集零重合

### 筛选规则（可复现）

```
输入：data/math/gsm8k/train.jsonl（7,473 条）
排除：本报告 §2.1 列出的全部 19 个排除集的 canonical_question_hash union
保留：6,043 条
转换：question → messages[system+user]，#### N → metadata.answer
输出：待构造（本审计不创建 split）
评测：data/math/gsm8k/split/test_converted_500.jsonl 前 50 条
```

**注意**：当前仅完成数据源审计，尚未构造最终训练 split。

### 后续扩展

M2.5 audit 通过后，方案 B（Hendrycks MATH Level 3-5）应作为升级路径：
1. 下载 Hendrycks MATH
2. 做 Level 2-5 baseline screening
3. 构造 `math_level_3_5.jsonl`

---

## 6. 验收

### 6.1 审计脚本

- 脚本：`scripts/data/audit_grpo_candidates.py`
- 命令：`python scripts/data/audit_grpo_candidates.py [--json]`
- 测试：`python -m unittest tests.test_grpo_data_audit -v`（53 tests, all pass）
- 机器摘要：`data/math/reports/grpo_data_audit_summary.json`

### 6.2 测试覆盖

| 测试类 | 测试数 | 覆盖内容 |
|--------|-------|---------|
| TestCanonicalQuestionHash | 5 | 哈希确定性、规范化、别名 |
| TestExtractProblemId | 5 | problem_id 各种 schema 提取 |
| TestExtractProblemText | 5 | 问题文本提取 |
| TestExtractAnswer | 4 | 答案提取 |
| TestGrpoCompatibility | 7 | GRPO schema 兼容性，含 trainer 一致性 |
| TestAuditFile | 7 | 真实文件审计，含后续记录 invalid 检测 |
| TestLoadIdsAndHashes | 3 | ID/hash 加载，含 raw GSM8K |
| TestComputeOverlap | 5 | canonical 去重，不双计数 |
| TestRunAudit | 8 | 集成测试：6973、6043、raw train 纳入 |
| TestTrainerSchemaConsistency | 4 | audit schema 与 train_grpo.py 一致性 |
| **Total** | **53** | |

### 6.3 全量测试

```
make test → 446 pass, 0 fail, 11 skip
```

### 6.4 关键数字复现验证

| 检查项 | 预期值 | 实际值 | 状态 |
|--------|-------|-------|------|
| raw GSM8K train 问题数 | 7,473 | 7,473 | OK |
| 仅排除 d5_500 后 | 6,973 | 6,973 | OK |
| 全排除集 union 后 | 6,043 | 6,043 | OK |
| `math_level_3_5.jsonl` 存在 | False | False | OK |
| GRPO 直接兼容文件数 | 3 | 3 | OK |

---

## 附录：GSM8K-50 Eval Set 定义

GSM8K-50 不是独立文件，而是 `test_converted_500.jsonl` 的前 50 条，在评估时通过 `--limit 50` 参数切片：

```bash
python scripts/smoke/eval_gsm8k.py \
  --test-data data/math/gsm8k/split/test_converted_500.jsonl \
  --limit 50 --temperature 0 --max-new-tokens 512
```

problem_id 范围：`gsm8k_test_0001` 至 `gsm8k_test_0050`。

M1 eval 结果：Baseline 10/50, SFT 14/50, DPO v1 14/50。
