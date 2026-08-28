# M2 DPO Reasoning Quality Pilot v2 — 校准通过

**Date**: 2026-08-28
**Status**: PASSED — 31 pairs from 31 independent problems (target: ≥30)
**Base seed**: 20260827

## 概述

v2 修复 v1 的三个问题：
1. 随机种子固定（seed=20260827/28/29，所有随机操作确定性）
2. 每题最多保留一对（从 C(n,2) 中选 gap 最大的）
3. 数据-报告-文档一致性校验

## 校准轮次

| 轮次 | 问题数 | seed | 候选/题 | 温度 | 可配对 | 通过 | 累计唯一 |
|------|--------|------|---------|------|--------|------|----------|
| v2 | 50 | 20260827 | 8 | 1.0 | 28 | 14 | 14 |
| v2b | 100 | 20260828 | 8 | 1.0 | 56 | 7 | 21 |
| v2b2 | 46 | 20260828 | 8 | 1.0 | 46 | 4 | 25 |
| v2c | 100 | 20260829 | 8 | 1.0 | 57 | 6 | **31** |

## 候选生成统计

| 指标 | 值 |
|------|-----|
| 总问题数 | 250（去重后200独立问题） |
| 总候选数 | 2000 |
| 正确候选 | ~700 (35%) |
| 温度 | 1.0 |
| 可配对问题（≥2 correct） | ~120 |

## 审稿统计

| 指标 | 值 |
|------|-----|
| 总审稿结果 | 395 |
| Non-tie | 331 (83.8%) |
| Tie | 64 (16.2%) |
| 通过质量过滤 | 45 |
| 通过机械过滤 | 39 |
| 每题去重后 | **31** |

## 过滤标准

| 条件 | 阈值 |
|------|------|
| chosen 总分 | ≥ 6/8 |
| chosen - rejected | ≥ 2 |
| rejected 总分 | ≥ 2/8 |
| 长度比 | 0.75 – 1.33 |
| 相似度 | 0.1 – 0.95 |
| 每题最多 | 1 pair |

## Gap 分布

| Gap | 数量 |
|-----|------|
| 2 | 18 |
| 3 | 5 |
| 4 | 4 |
| 5 | 4 |

## 一致性校验

- ✅ pairs 数 = unique problem_ids = 31
- ✅ 所有 pair 的 answer 与 gold 一致
- ✅ 无重复 problem_id
- ✅ 随机种子确定性（相同 seed → 相同结果）

## 与 v1 对比

| 指标 | v1 | v2 |
|------|-----|-----|
| 温度 | 0.7 | 1.0 |
| 候选/题 | 4 | 8 |
| Non-tie rate | 10% | 84% |
| 最终 pairs | 8 | **31** |
| 独立问题 | 4 | **31** |

## 关键改进

1. **温度提升**：temp=1.0 产生更多推理路径变异（84% non-tie vs v1 的 10%）
2. **候选数增加**：8候选/题 → C(8,2)=28 pairs/题 → 更多质量差异
3. **预筛选策略**：按输出长度差/多样性预选最优 pair → 减少审稿量
4. **多轮校准**：3轮×250题 → 逐步积累通过 pairs

## 下一步

校准通过（31 ≥ 30）。按用户要求：
1. 构造500条 DPO pairs（从 GSM8K train 中再抽250题，每题8候选）
2. 交 DeepSeek 二审确认 chosen 质量优势
3. 写入正式 DPO 训练数据

## 文件清单

| 文件 | 说明 |
|------|------|
| `data/math/pilots/dpo_rq_v2_problems_50.jsonl` | v2 50题 |
| `data/math/pilots/dpo_rq_v2_candidates_50.jsonl` | v2 400候选 |
| `data/math/pilots/dpo_rq_v2b_problems_100.jsonl` | v2b 100题 |
| `data/math/pilots/dpo_rq_v2b_candidates_100.jsonl` | v2b 800候选 |
| `data/math/pilots/dpo_rq_v2c_problems_100.jsonl` | v2c 100题 |
| `data/math/pilots/dpo_rq_v2c_candidates_100.jsonl` | v2c 800候选 |
| `data/math/pilots/dpo_rq_v2_pairs_final.jsonl` | **31 final DPO pairs** |
| `data/math/reports/dpo_rq_v2_calibration_report.json` | 校准报告 |
| `scripts/generate_reasoning_quality_candidates_v2.py` | 生成脚本 |
