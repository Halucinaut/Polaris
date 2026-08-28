# M2 DPO Reasoning Quality v3b — 500 题扩展

**Date**: 2026-08-28
**Status**: PASSED — 47 main pairs (v3a 20 + v3b 27) ≥ 30 target

## v3b 改进

- 500 题 × 8 候选 = 4000 candidates (39.1% correct)
- 逐次 `mx.random.seed(seed_i)` 设置
- 问题哈希零重合断言 (Probe-30/Stress-50/v3a)
- chosen 必须 0 语义问题，rejected 必须 ≥1 语义问题

## 生成统计

| 指标 | v3a | v3b | 合计 |
|------|-----|-----|------|
| 问题数 | 250 | 500 | 750 |
| 候选数 | 2000 | 4000 | 6000 |
| 正确率 | 37.4% | 39.1% | — |
| 可配对 | 138 | 301 | 439 |

## 审稿统计

| 指标 | v3a (re-review) | v3b |
|------|-----------------|-----|
| 总审稿 | 44 | 301 |
| Non-tie | — | 139 |
| Tie | — | 162 (53.8%) |
| chosen=0 语义 | 20 | 27 |
| rejected≥1 语义 | 20 | 27 |

## 分类结果

| 类别 | v3a | v3b | 合计 |
|------|-----|-----|------|
| **主训练** (1-3类差异) | **20** | **27** | **47** |
| 辅助 (4类差异) | 0 | 32 | 32 |

## 一致性校验

- ✅ problem_hash 唯一身份
- ✅ mx.random.seed(seed_i) 逐次设置
- ✅ 零 Probe-30/Stress-50/v3a 重合
- ✅ chosen 0 语义问题
- ✅ rejected ≥1 语义问题
- ✅ 无重复 phash

## DeepSeek 复审

复审批次已准备：`data/math/pilots/dpo_rq_v3_deepseek_review_batch.jsonl` (47 pairs)
需要设置 `DEEPSEEK_API_KEY` 环境变量后运行。

## 文件清单

| 文件 | 说明 |
|------|------|
| `data/math/pilots/dpo_rq_v3/` | v3a 产物（冻结） |
| `data/math/pilots/dpo_rq_v3_b/` | v3b 产物 |
| `data/math/pilots/dpo_rq_v3_b/dpo_pairs_main.jsonl` | 27 主训练 DPO pairs |
| `data/math/pilots/dpo_rq_v3_deepseek_review_batch.jsonl` | DeepSeek 复审批次 (47) |
| `data/math/pilots/dpo_rq_v3_b/report.json` | v3b 报告 |
| `polaris/problem_hash.py` | 哈希工具 |
| `tests/test_rq_v3_determinism.py` | 确定性测试 |
