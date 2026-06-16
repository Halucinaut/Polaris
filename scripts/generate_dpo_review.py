#!/usr/bin/env python3
"""
Generate DPO pair review template from baseline and SFT predictions.

This script loads baseline and SFT predictions, compares them, and generates
a review template with 20 candidate DPO pairs for human review.
"""

import json
import re
from pathlib import Path
from typing import Optional


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_answer(prediction: str) -> tuple[Optional[str], str]:
    """Extract answer from prediction."""
    # Try boxed first
    match = re.search(r"\\boxed\{(.*?)\}", prediction)
    if match:
        return match.group(1), "boxed"

    # Try numeric fallback
    candidates = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?:/\d+)?(?!\w)", prediction)
    if candidates:
        return candidates[-1], "numeric_fallback"

    return None, "none"


def answers_match(pred: Optional[str], ref: str) -> bool:
    """Check if predicted answer matches reference."""
    if pred is None:
        return False

    # Normalize
    pred_norm = pred.strip().lower().replace(" ", "")
    ref_norm = ref.strip().lower().replace(" ", "")

    if pred_norm == ref_norm:
        return True

    # Try numeric comparison
    try:
        from fractions import Fraction
        pred_num = Fraction(pred_norm)
        ref_num = Fraction(ref_norm)
        return pred_num == ref_num
    except:
        pass

    return False


def has_format_adherence(prediction: str) -> bool:
    """Check if prediction follows think+boxed format."""
    has_think = bool(re.search(r"<think>.*?</think>", prediction, re.DOTALL))
    has_boxed = bool(re.search(r"\\boxed\{.*?\}", prediction))
    return has_think and has_boxed


def estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def generate_review_template():
    """Generate DPO pair review template."""

    # Load predictions
    baseline_preds = load_jsonl(Path("runs/baseline_50_eval/test_predictions.jsonl"))
    sft_preds = load_jsonl(Path("runs/sft_50_eval/test_predictions.jsonl"))

    # Load reference answers
    test_data = load_jsonl(Path("data/math/gsm8k/split/test_converted_500.jsonl"))
    ref_map = {rec["problem_id"]: rec for rec in test_data[:50]}

    # Create prediction maps
    baseline_map = {rec["problem_id"]: rec for rec in baseline_preds}
    sft_map = {rec["problem_id"]: rec for rec in sft_preds}

    # Find candidate pairs
    pairs = []

    for problem_id in baseline_map.keys():
        if problem_id not in sft_map or problem_id not in ref_map:
            continue

        baseline = baseline_map[problem_id]
        sft = sft_map[problem_id]
        ref = ref_map[problem_id]

        baseline_answer, baseline_method = extract_answer(baseline["prediction"])
        sft_answer, sft_method = extract_answer(sft["prediction"])

        baseline_correct = answers_match(baseline_answer, ref["answer"])
        sft_correct = answers_match(sft_answer, ref["answer"])

        baseline_format = has_format_adherence(baseline["prediction"])
        sft_format = has_format_adherence(sft["prediction"])

        # Generate pair based on correctness
        if sft_correct and not baseline_correct:
            # SFT correct, baseline wrong
            pair_type = "sft_correct_vs_baseline_wrong"
            auto_tag = "clean"
        elif sft_correct and baseline_correct:
            # Both correct - skip (not useful for DPO)
            continue
        elif not sft_correct and not baseline_correct:
            # Both wrong - skip
            continue
        else:
            # SFT wrong, baseline correct - skip (unusual)
            continue

        # Estimate tokens
        chosen_tokens = estimate_tokens(sft["prediction"])
        rejected_tokens = estimate_tokens(baseline["prediction"])
        length_ratio = round(chosen_tokens / max(rejected_tokens, 1), 2)

        # Check length bias
        if length_ratio > 3.0 or length_ratio < 0.33:
            auto_tag = "length_biased"

        # Check format difference
        if sft_format and not baseline_format:
            if auto_tag == "clean":
                auto_tag = "clean"  # Still clean, format is part of the preference

        pair = {
            "id": f"dpo_review_{len(pairs)+1:04d}",
            "problem_id": problem_id,
            "question": ref["problem"],
            "answer": ref["answer"],
            "pair_type": pair_type,
            "chosen": sft["prediction"],
            "rejected": baseline["prediction"],
            "chosen_answer": sft_answer,
            "rejected_answer": baseline_answer,
            "chosen_answer_correct": sft_correct,
            "rejected_answer_correct": baseline_correct,
            "chosen_format_adherence": sft_format,
            "rejected_format_adherence": baseline_format,
            "chosen_token_length": chosen_tokens,
            "rejected_token_length": rejected_tokens,
            "token_length_ratio": length_ratio,
            "auto_tag": auto_tag,
            "reviewer_tag": "",
            "reviewer_note": "",
            "decision": ""
        }

        pairs.append(pair)

    # Take first 20 pairs
    pairs = pairs[:20]

    # Save JSONL
    output_path = Path("data/math/review/dpo_pair_review_20.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Generate markdown table
    md_path = Path("docs/experiment_notes/dpo_pair_review_20.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# DPO Pair Review Template (20 Pairs)\n\n")
        f.write("## 概览\n\n")
        f.write(f"- 总 pair 数: {len(pairs)}\n")
        f.write(f"- pair_type: sft_correct_vs_baseline_wrong\n")
        auto_tag_counts = {}
        for p in pairs:
            tag = p['auto_tag']
            auto_tag_counts[tag] = auto_tag_counts.get(tag, 0) + 1
        f.write(f"- auto_tag 分布: {auto_tag_counts}\n")
        f.write(f"- 输出文件: {output_path}\n\n")

        f.write("## 审阅说明\n\n")
        f.write("1. 检查 chosen 和 rejected 的答案是否正确\n")
        f.write("2. 检查 reasoning 质量是否有实质差异\n")
        f.write("3. 判断 pair 方向是否明确\n")
        f.write("4. 填写 reviewer_tag（clean/weak/length_biased/format_only/invalid）\n")
        f.write("5. 填写 decision（keep/discard/needs_review）\n\n")

        f.write("## Pair 列表\n\n")

        for i, pair in enumerate(pairs, 1):
            f.write(f"### Pair {i}: {pair['problem_id']}\n\n")
            f.write(f"**Question:** {pair['question']}\n\n")
            f.write(f"**Answer:** {pair['answer']}\n\n")
            f.write(f"**Pair Type:** {pair['pair_type']}\n\n")
            f.write(f"**Auto Tag:** {pair['auto_tag']}\n\n")
            f.write(f"**Token Length:** chosen={pair['chosen_token_length']}, rejected={pair['rejected_token_length']}, ratio={pair['token_length_ratio']}\n\n")

            f.write("**Chosen (SFT):**\n")
            f.write(f"```\n{pair['chosen'][:500]}{'...' if len(pair['chosen']) > 500 else ''}\n```\n")
            f.write(f"Answer: {pair['chosen_answer']} ({'✓' if pair['chosen_answer_correct'] else '✗'})\n")
            f.write(f"Format: {'✓' if pair['chosen_format_adherence'] else '✗'}\n\n")

            f.write("**Rejected (Baseline):**\n")
            f.write(f"```\n{pair['rejected'][:500]}{'...' if len(pair['rejected']) > 500 else ''}\n```\n")
            f.write(f"Answer: {pair['rejected_answer']} ({'✓' if pair['rejected_answer_correct'] else '✗'})\n")
            f.write(f"Format: {'✓' if pair['rejected_format_adherence'] else '✗'}\n\n")

            f.write("**Reviewer:**\n")
            f.write(f"- reviewer_tag: {pair['reviewer_tag'] or '___'}\n")
            f.write(f"- reviewer_note: {pair['reviewer_note'] or '___'}\n")
            f.write(f"- decision: {pair['decision'] or '___'}\n\n")
            f.write("---\n\n")

    print(f"Generated {len(pairs)} pairs")
    print(f"JSONL: {output_path}")
    print(f"Markdown: {md_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Auto tag distribution:")
    for tag in sorted(set(p['auto_tag'] for p in pairs)):
        count = sum(1 for p in pairs if p['auto_tag'] == tag)
        print(f"    {tag}: {count}")


if __name__ == "__main__":
    generate_review_template()
