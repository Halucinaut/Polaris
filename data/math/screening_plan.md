# M1 Math Baseline Screening Plan

## 1. Goal

为 Qwen3-0.6B Math SFT 构造一个小规模、可自动判分、可人工审阅的数据子集。该子集服务于 M1 baseline、SFT sanity、full SFT 和 sample diff review。

## 2. Selected source

- Primary source: GSM8K
- Reason:
  - 有 question、answer 字段，结构简单。
  - answer 最后的 #### 后的内容为最终答案，便于抽取。且全部是数字。
  - 非socratic题。
- Fallback source: MATH
- Fallback trigger:
  - GSM8K 的答案抽取成功率低于 90%。
  - 无法观察到明显的SFT结果，包括loss曲线或accuracy曲线。

## 3. Target subset size

D2 最小版本：

- raw\_subset: 300
- train: 210
- val: 30
- test: 30
- review: 30

M1 full SFT 可扩展版本：

- train: 7k
- val: 0.47k
- test: 1k
- review: 0.32k

## 4. Filtering rules

保留样本：

- question 非空。
- answer 非空且可以抽取最终答案。
- question 长度不超过 2000 字符。
- answer 长度不超过 4000 字符。
- 最终答案优先为整数、分数、小数或短表达式。

排除样本：

- 最终答案无法稳定抽取。
- 题目或解答过长。
- question 归一化后重复。

## 5. Answer extraction rule

从 answer 中按以下优先级抽取最终答案：

1. `#### ...`

D2 最小版本只强制支持 `#### ...`。抽取失败的样本不进入 split。

## 6. Split protocol

先对过滤后的样本做去重，再用固定 seed 打乱。

- seed: 42
- train: 70%
- val: 10%
- test: 10%
- review: 10%

约束：

- 同一个 normalized\_question 只能出现在一个 split。
- review split 固定用于人工审阅，不参与训练。
- test split 只用于 baseline 与 SFT 后对比。
- val split 用于训练过程检查。

## 7. Manual review protocol

人工抽查至少 30 条 raw/filtered 样本，记录：

- question 是否完整。
- answer 是否对应题目。
- answer 是否能抽取最终结果。
- 答案是否适合自动判分。
- 是否需要图形。
- 难度是否适合 0.6B M1。

人工审阅结论写入：8.Manual\_review\_notes

## 8. Manual\_review\_notes

- Reviewed file: 'data/math/gsm8k/split/review\.jsonl'
- Reviewed samples: 30
- Findings:
  - All reviewed samples contain a final answer after `####`.
  - The final answer format is standard numeric text.
  - All 30 reviewed final answers are integers.
  - No obvious malformed problem or answer was found.
  - Difficulty is relatively consistent and fits grade-school arithmetic reasoning.
  - Most problems require roughly 1-4 reasoning steps.
  - Problem length varies from short prompts to prompts over 100 words.
- Conclusion: GSM8K passes D2 manual review for M1 minimal data construction.

