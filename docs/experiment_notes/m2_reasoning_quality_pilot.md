# M2 DPO Reasoning Quality Pilot

**Date**: 2026-08-27
**Status**: FAILED — did not meet ≥30 pair threshold
**Base seed**: 20260827

## 概述

尝试构造"解题步骤质量"DPO pairs：从 M1 adapter (SFT GSM8K 500) 采样 4 个候选，用 Claude 盲审评分筛选质量差异 pairs。

## 数据

- **来源**: `dpo_v2_style_train_449.jsonl`，排除 Probe-30 + Stress-50 共 80 IDs
- **采样**: seed=20260827 抽取 50 problem_ids
- **候选生成**: M1 adapter (`runs/000030_qwen3_0_6b_sft_gsm8k_500/checkpoints/final`)，temperature=0.7，每题 4 候选

## 候选生成结果

| 指标 | 值 |
|------|-----|
| 总候选数 | 200 |
| 正确候选 | 156 (78.0%) |
| 全对题目 | 27/50 |
| ≥2 correct（可配对）| 44/50 |
| 0 correct | 3 (gsm8k_train_d5_0124, 0264, 0277) |
| 1 correct | 3 (0158, 0200, 0353) |

## 审稿结果

### 初审（宽松标准）
- 201 pairs 审稿
- **181 ties (90%)**，20 non-ties
- 仅 7 pairs 通过过滤标准（chosen≥6, gap≥2, rejected≥2）

### 复审（严格标准）
- 201 pairs 严格审稿
- **121 ties (60%)**，80 non-ties
- 严格审稿后 10 pairs 通过评分标准
- 机械过滤（长度比 0.75-1.33，去重，去极相似）后 **8 pairs**

## 失败原因分析

### 核心问题：GSM8K 对 M1 太简单

M1 SFT 模型在 GSM8K 上的推理质量高度一致：
1. **推理路径单一**: 同一问题的4个候选几乎总是走相同的解题路径
2. **差异仅在措辞**: temperature=0.7 产生措辞变化，但不产生方法论差异
3. **评分天花板**: 80%+ 候选获得 8/8 (2,2,2,2)，差异空间极小

### 审稿评分分布

| 指标 | 初审 | 严格审 |
|------|------|--------|
| Tie rate | 90% | 60% |
| Non-tie pairs | 20 | 80 |
| Gap≥2 pairs | 7 | 10 |
| 最终通过 | 7 | 8 |

### 严格审稿中的典型 tie 案例

```
Pair rq_0224812f73d5 (gsm8k_train_d5_0476):
  A: [2,2,2,2] = 8  B: [2,2,2,2] = 8
  "Both candidates have identical reasoning: compute supermarket round-trip (10 mi),
   aborted farm trip (4 mi), final farm trip (6 mi), total 20 mi on 10 gal = 2 mpg."
```

### 严格审稿中的典型 non-tie 案例

```
Pair rq_c38c76f8c825 (gsm8k_train_d5_0238):
  A: [0,0,2,1] = 3  B: [2,2,2,2] = 8
  "Candidate A completely ignores Melany (3rd person) and divides by 2 instead of 3,
   producing the right numerical answer by coincidence but with fundamentally flawed reasoning."
```

## 通过的 8 pairs

| pair_id | problem_id | chosen | rejected | gap | length_ratio | similarity |
|---------|-----------|--------|----------|-----|--------------|------------|
| rq_c38c76f8c825 | 0238 | 8 | 3 | 4 | 0.764 | 0.660 |
| rq_450b272838cf | 0325 | 8 | 2 | 6 | 0.752 | 0.540 |
| rq_0c607890f83e | 0470 | 8 | 5 | 3 | 1.140 | 0.269 |
| rq_b4b1a1cd349e | 0145 | 8 | 5 | 3 | 0.846 | 0.373 |
| rq_56513b75560a | 0409 | 8 | 3 | 5 | 0.776 | 0.546 |
| rq_87af8ee4e4bc | 0171 | 8 | 5 | 3 | 0.884 | 0.790 |
| rq_a37e0c1a5116 | 0496 | 8 | 5 | 3 | 0.884 | 0.790 |
| rq_d45db0a71e04 | 0388 | 8 | 6 | 2 | 0.976 | 0.290 |

## 结论

**Pilot 未通过**。8 pairs < 30 target。

### 根因

GSM8K 是小学数学题，M1 SFT 模型的推理质量已经很高且一致。temperature 采样只能产生措辞差异，无法产生有意义的推理质量差异。

### 建议

1. **换用更难的问题集**: MATH、AMC、AIME 等竞赛题会产生更多样的解题路径
2. **增加候选数**: 每题8-16个候选，增大发现质量差异的概率
3. **降低过滤阈值**: gap≥1 而非 ≥2（但会降低 pair 质量）
4. **使用更强的模型**: 0.6B 模型能力有限，推理路径单一
5. **结合 style+quality**: 在 style 差异的基础上叠加 quality 差异

## 文件清单

| 文件 | 说明 |
|------|------|
| `data/math/pilots/dpo_reasoning_quality_problems_50.jsonl` | 50 problems |
| `data/math/pilots/dpo_reasoning_quality_candidates_50.jsonl` | 200 candidates |
| `data/math/pilots/dpo_reasoning_quality_all_pairs.jsonl` | 201 pairs |
| `data/math/pilots/dpo_reasoning_quality_pairs_50.jsonl` | 8 final DPO pairs |
| `data/math/reports/dpo_reasoning_quality_pilot_report.json` | 完整报告 |
| `data/math/pilots/review_batch_*.jsonl` | 审稿批次 (7) |
| `data/math/pilots/review_results_strict_*.jsonl` | 严格审稿结果 (7) |
| `scripts/generate_reasoning_quality_candidates.py` | 候选生成脚本 |
| `scripts/review_reasoning_quality_pairs.py` | 配对脚本 |
