# M2 DPO Reasoning Quality Pilot v3

**Date**: 2026-08-28
**Status**: Main pairs 21 < 30 target; total unique 44

## v3 改进

1. **problem_hash**：SHA256(normalized_text)[:16] 作为跨文件唯一身份
2. **mx.random.seed(seed_i)**：每次生成前实际调用，determinism test 通过
3. **零重合断言**：candidates ∩ Probe-30/Stress-50 = ∅
4. **4类审稿标签**：
   - 条件遗漏 (condition_omission)
   - 逻辑错误 (logic_error)
   - 单位/量纲错误 (unit_error)
   - 冗余/表达 (redundancy)
5. **主训练/辅助分离**：仅1-3类差异进主训练，4类差异单列

## 生成统计

| 指标 | 值 |
|------|-----|
| 问题数 | 250 |
| 候选数 | 2000 (8/题) |
| 正确候选 | 747 (37.4%) |
| 可配对问题 | 138 |
| 温度 | 1.0 |
| seed | mx.random.seed(20260827 + pidx*1000 + ci) |

## 审稿统计

| 指标 | 值 |
|------|-----|
| 总审稿 | 138 |
| Non-tie | 88 (63.8%) |
| Tie | 50 (36.2%) |

## 分类结果

| 类别 | pairs | 去重后唯一问题 |
|------|-------|----------------|
| 主训练 (1-3类差异) | 21 | 21 |
| 辅助 (仅4类差异) | 23 | 23 |
| **总计** | **44** | **44** |

## 一致性校验

- ✅ problem_hash 作为唯一 ID
- ✅ mx.random.seed(seed_i) 实际设置
- ✅ determinism test 6/6 通过
- ✅ 零 Probe-30/Stress-50 重合
- ✅ 主训练集无重复 phash
- ✅ 所有 pair 两侧答案正确

## 问题

主训练集21对 < 30目标。根因：
- 条件遗漏/逻辑错误/单位错误（1-3类）差异在 GSM8K 上较少
- 大部分质量差异属于冗余/表达（4类），不进主训练

## 文件清单

| 文件 | 说明 |
|------|------|
| `polaris/problem_hash.py` | 哈希工具 |
| `scripts/generate_rq_v3_candidates.py` | 候选生成 |
| `scripts/review_rq_v3.py` | 配对+分类审稿 |
| `tests/test_rq_v3_determinism.py` | 确定性测试 |
| `data/math/pilots/dpo_rq_v3/problems.jsonl` | 250题 |
| `data/math/pilots/dpo_rq_v3/candidates.jsonl` | 2000候选 |
| `data/math/pilots/dpo_rq_v3/pairs_main.jsonl` | 21 主训练 pairs |
| `data/math/pilots/dpo_rq_v3/pairs_auxiliary.jsonl` | 23 辅助 pairs |
| `data/math/pilots/dpo_rq_v3/dpo_pairs_main.jsonl` | DPO 格式主训练 |
| `data/math/pilots/dpo_rq_v3/report.json` | 完整报告 |
