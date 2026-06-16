#!/usr/bin/env python3
"""
Construct DPO v1 dataset using gold-vs-model-wrong strategy.

Chosen: GSM8K gold solution reformatted to <think> + \boxed{} target format.
Rejected: Model's wrong output (SFT preferred, then baseline).

Strategy priority:
  1. gold_vs_sft_wrong (SFT got it wrong)
  2. gold_vs_baseline_wrong (baseline got it wrong, SFT also wrong or not available)
  3. sft_correct_vs_baseline_wrong (limited, with strict length/format checks)

Filtered out:
  - baseline_correct_vs_sft_wrong → sft_regression_repair pool (not in dpo_v1)
"""

import json
import re
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem and put the final answer in \\boxed{}."
)

# Length ratio bounds for "clean" vs "length_biased"
LENGTH_RATIO_CLEAN_MIN = 0.5
LENGTH_RATIO_CLEAN_MAX = 2.0
LENGTH_RATIO_HARD_MIN = 0.25
LENGTH_RATIO_HARD_MAX = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def extract_answer(text: str) -> tuple[Optional[str], str]:
    """Extract answer from model output or gold solution."""
    # Try \boxed{...}
    m = re.search(r"\\boxed\{(.*?)\}", text)
    if m:
        return m.group(1).strip(), "boxed"
    # Try #### marker (gold solution format)
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip(), "hash_answer"
    # Numeric fallback (last standalone number)
    nums = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?:/\d+)?(?!\w)", text)
    if nums:
        return nums[-1], "numeric_fallback"
    return None, "none"


def answers_match(pred: Optional[str], ref: str) -> bool:
    if pred is None:
        return False
    p = pred.strip().lower().replace(",", "").replace(" ", "")
    r = ref.strip().lower().replace(",", "").replace(" ", "")
    if p == r:
        return True
    try:
        return Fraction(p) == Fraction(r)
    except (ValueError, ZeroDivisionError):
        return False


def has_format(text: str) -> bool:
    has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    has_boxed = bool(re.search(r"\\boxed\{.*?\}", text))
    return has_think and has_boxed


def est_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def is_truncated(text: str) -> bool:
    return "[truncated]" in text or text.endswith("...") and len(text) > 500


def text_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity on word sets."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def gold_to_chosen(gold_solution: str, gold_answer: str) -> str:
    """Convert GSM8K gold solution to target format.

    Strip the #### line, clean up <<calc>> markers, wrap in <think> + \boxed{}.
    GSM8K format: <<expr=result>>result — remove markers, keep result after marker.
    """
    # Remove #### line
    lines = gold_solution.strip().split("\n")
    reasoning_lines = []
    for line in lines:
        if line.strip().startswith("####"):
            continue
        # Clean <<expr=result>> markers: remove the marker, keep text after it
        cleaned = re.sub(r"<<[^>]*>>", "", line)
        reasoning_lines.append(cleaned.strip())

    reasoning = "\n".join(reasoning_lines).strip()
    if not reasoning:
        reasoning = f"The answer is {gold_answer}."

    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{gold_answer}}}"


def make_pair_id(pid: str, pair_type: str) -> str:
    h = hashlib.md5(f"{pid}:{pair_type}".encode()).hexdigest()[:8]
    return f"dpo_{h}"


# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------

def main():
    # Load data
    test_data = load_jsonl(Path("data/math/gsm8k/split/test_converted_500.jsonl"))
    test_map = {r["problem_id"]: r for r in test_data}

    sft_preds = load_jsonl(Path("runs/sft_500_eval/test_predictions.jsonl"))
    sft_map = {r["problem_id"]: r for r in sft_preds}

    bl_preds = load_jsonl(Path("runs/baseline_500_eval/test_predictions.jsonl"))
    bl_map = {r["problem_id"]: r for r in bl_preds}

    # --- Phase 1: Build candidates ---
    candidates = []
    sft_regression_pool = []
    filter_counts = Counter()

    for pid, ref in test_map.items():
        sft = sft_map.get(pid)
        bl = bl_map.get(pid)

        # Extract answers
        sft_ans, sft_method = extract_answer(sft["prediction"]) if sft else (None, "none")
        bl_ans, bl_method = extract_answer(bl["prediction"]) if bl else (None, "none")
        gold_ans = ref["answer"]

        sft_correct = answers_match(sft_ans, gold_ans)
        bl_correct = answers_match(bl_ans, gold_ans)
        sft_fmt = has_format(sft["prediction"]) if sft else False
        bl_fmt = has_format(bl["prediction"]) if bl else False

        gold_chosen = gold_to_chosen(ref["solution"], gold_ans)
        gold_tokens = est_tokens(gold_chosen)

        # --- Strategy 1: gold_vs_sft_wrong ---
        if sft and not sft_correct:
            rejected = sft["prediction"]
            rej_ans, rej_method = sft_ans, sft_method
            rej_fmt = sft_fmt
            rej_tokens = est_tokens(rejected)
            pair_type = "gold_vs_sft_wrong"
            chosen_origin = "gsm8k_gold"
            rejected_origin = "sft_eval"

        # --- Strategy 2: gold_vs_baseline_wrong (when SFT is also wrong or missing) ---
        elif bl and not bl_correct:
            rejected = bl["prediction"]
            rej_ans, rej_method = bl_ans, bl_method
            rej_fmt = bl_fmt
            rej_tokens = est_tokens(rejected)
            pair_type = "gold_vs_baseline_wrong"
            chosen_origin = "gsm8k_gold"
            rejected_origin = "baseline_eval"

        # --- Strategy 3: sft_correct_vs_baseline_wrong (limited) ---
        elif sft and bl and sft_correct and not bl_correct:
            # Only keep if no truncation, reasonable length, both formatted
            if is_truncated(sft["prediction"]) or is_truncated(bl["prediction"]):
                filter_counts["truncation"] += 1
                continue
            if not sft_fmt or not bl_fmt:
                filter_counts["format_mismatch"] += 1
                continue
            rej_tokens = est_tokens(bl["prediction"])
            sft_tokens = est_tokens(sft["prediction"])
            ratio = sft_tokens / max(rej_tokens, 1)
            if ratio < LENGTH_RATIO_HARD_MIN or ratio > LENGTH_RATIO_HARD_MAX:
                filter_counts["extreme_length"] += 1
                continue
            rejected = bl["prediction"]
            rej_ans, rej_method = bl_ans, bl_method
            rej_fmt = bl_fmt
            pair_type = "sft_correct_vs_baseline_wrong"
            chosen_origin = "sft_eval"
            rejected_origin = "baseline_eval"

        # --- baseline_correct_vs_sft_wrong → regression pool ---
        elif sft and bl and bl_correct and not sft_correct:
            sft_regression_pool.append({
                "problem_id": pid,
                "answer": gold_ans,
                "sft_prediction": sft["prediction"],
                "sft_answer": sft_ans,
                "baseline_prediction": bl["prediction"],
                "baseline_answer": bl_ans,
                "reason": "sft_regression",
            })
            continue

        else:
            # Both correct, or missing data
            continue

        # --- Apply filters ---
        chosen = gold_chosen if pair_type != "sft_correct_vs_baseline_wrong" else sft["prediction"]
        if pair_type == "sft_correct_vs_baseline_wrong":
            chosen_tokens = est_tokens(chosen)
        else:
            chosen_tokens = gold_tokens

        # Truncation
        if is_truncated(chosen) or is_truncated(rejected):
            filter_counts["truncation"] += 1
            continue

        # Empty
        if not chosen.strip() or not rejected.strip():
            filter_counts["empty"] += 1
            continue

        # Extractability
        c_ans, c_method = extract_answer(chosen)
        if c_ans is None:
            filter_counts["unextractable_chosen"] += 1
            continue
        if rej_ans is None:
            filter_counts["unextractable_rejected"] += 1
            continue

        # Rejected must be wrong (except for sft_correct_vs_baseline_wrong)
        rej_correct = answers_match(rej_ans, gold_ans)
        if rej_correct and pair_type != "sft_correct_vs_baseline_wrong":
            filter_counts["rejected_correct"] += 1
            continue

        # Content similarity
        sim = text_similarity(chosen, rejected)
        if sim > 0.85:
            filter_counts["similar_content"] += 1
            continue

        # Length ratio
        ratio = chosen_tokens / max(rej_tokens, 1)
        if ratio < LENGTH_RATIO_HARD_MIN or ratio > LENGTH_RATIO_HARD_MAX:
            filter_counts["extreme_length"] += 1
            continue

        quality_tag = "clean"
        if ratio < LENGTH_RATIO_CLEAN_MIN or ratio > LENGTH_RATIO_CLEAN_MAX:
            quality_tag = "length_biased"

        # Format adherence
        c_fmt = has_format(chosen)

        candidate = {
            "id": make_pair_id(pid, pair_type),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ref["problem"]},
            ],
            "chosen": chosen,
            "rejected": rejected,
            "answer": gold_ans,
            "source": "gsm8k",
            "pair_type": pair_type,
            "quality_tag": quality_tag,
            "metadata": {
                "problem_id": pid,
                "pair_origin": pair_type,
                "chosen_origin": chosen_origin,
                "rejected_origin": rejected_origin,
                "generation_run_id": "000030_qwen3_0_6b_sft_gsm8k_500" if "sft" in rejected_origin else "000005_qwen3_0_6b_gsm8k_baseline",
                "chosen_answer_correct": True,
                "rejected_answer_correct": rej_correct,
                "chosen_format_adherence": c_fmt,
                "rejected_format_adherence": rej_fmt,
                "chosen_token_length": chosen_tokens,
                "rejected_token_length": rej_tokens,
                "token_length_gap": chosen_tokens - rej_tokens,
                "token_length_ratio": round(ratio, 2),
                "chosen_extraction_method": c_method,
                "rejected_extraction_method": rej_method,
                "filter_reason": "",
            },
        }
        candidates.append(candidate)

    # --- Phase 2: Apply final quality filter to produce dpo_v1 ---
    dpo_v1 = []
    for c in candidates:
        pt = c["pair_type"]
        qt = c["quality_tag"]
        ratio = c["metadata"]["token_length_ratio"]

        # For sft_correct_vs_baseline_wrong, only keep clean (not length_biased)
        if pt == "sft_correct_vs_baseline_wrong" and qt == "length_biased":
            c["metadata"]["filter_reason"] = "sft_vs_bl_length_biased"
            continue

        dpo_v1.append(c)

    # --- Phase 3: Save outputs ---
    cand_path = Path("data/math/splits/dpo_candidates.jsonl")
    v1_path = Path("data/math/splits/dpo_v1.jsonl")
    report_path = Path("data/math/reports/dpo_v1_report.json")
    reg_path = Path("data/math/splits/dpo_sft_regression_pool.jsonl")

    cand_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cand_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(v1_path, "w", encoding="utf-8") as f:
        for c in dpo_v1:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(reg_path, "w", encoding="utf-8") as f:
        for r in sft_regression_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Phase 4: Generate report ---
    total_cand = len(candidates)
    kept = len(dpo_v1)
    filtered = total_cand - kept

    pt_dist = Counter(c["pair_type"] for c in dpo_v1)
    qt_dist = Counter(c["quality_tag"] for c in dpo_v1)

    if dpo_v1:
        c_lens = [c["metadata"]["chosen_token_length"] for c in dpo_v1]
        r_lens = [c["metadata"]["rejected_token_length"] for c in dpo_v1]
        gaps = [c["metadata"]["token_length_gap"] for c in dpo_v1]
        ratios = [c["metadata"]["token_length_ratio"] for c in dpo_v1]
        c_fmt_count = sum(1 for c in dpo_v1 if c["metadata"]["chosen_format_adherence"])
        r_fmt_count = sum(1 for c in dpo_v1 if c["metadata"]["rejected_format_adherence"])
        c_ans_correct = sum(1 for c in dpo_v1 if c["metadata"]["chosen_answer_correct"])
        r_ans_correct = sum(1 for c in dpo_v1 if c["metadata"]["rejected_answer_correct"])
    else:
        c_lens = r_lens = gaps = ratios = [0]
        c_fmt_count = r_fmt_count = c_ans_correct = r_ans_correct = 0

    report = {
        "total_candidates": total_cand,
        "kept_pairs": kept,
        "filtered_pairs": filtered,
        "filter_breakdown": dict(filter_counts),
        "pair_type_distribution": dict(pt_dist),
        "quality_tag_distribution": dict(qt_dist),
        "chosen_avg_token_length": round(sum(c_lens) / len(c_lens), 1),
        "rejected_avg_token_length": round(sum(r_lens) / len(r_lens), 1),
        "avg_token_length_gap": round(sum(gaps) / len(gaps), 1),
        "avg_token_length_ratio": round(sum(ratios) / len(ratios), 2),
        "chosen_format_adherence": round(c_fmt_count / max(kept, 1), 3),
        "rejected_format_adherence": round(r_fmt_count / max(kept, 1), 3),
        "chosen_answer_correct_rate": round(c_ans_correct / max(kept, 1), 3),
        "rejected_answer_correct_rate": round(r_ans_correct / max(kept, 1), 3),
        "truncation_filtered_count": filter_counts.get("truncation", 0),
        "extreme_length_gap_count": filter_counts.get("extreme_length", 0),
        "same_answer_pair_count": filter_counts.get("similar_content", 0),
        "sft_regression_pool_count": len(sft_regression_pool),
        "notes": "",
    }

    if kept < 200:
        report["notes"] = (
            f"Only {kept} clean pairs from 200 test samples. "
            "Gold-vs-model-wrong strategy yields 1 pair per wrong answer. "
            "To reach 200+ pairs, expand eval to full 500 test set or use SFT multi-sample."
        )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"=== DPO v1 Construction ===")
    print(f"Test samples:     {len(test_map)}")
    print(f"Candidates:       {total_cand}")
    print(f"Kept (dpo_v1):    {kept}")
    print(f"Filtered:         {filtered}")
    print(f"Regression pool:  {len(sft_regression_pool)}")
    print(f"\nPair type distribution:")
    for pt, cnt in sorted(pt_dist.items()):
        print(f"  {pt}: {cnt}")
    print(f"\nQuality tag distribution:")
    for qt, cnt in sorted(qt_dist.items()):
        print(f"  {qt}: {cnt}")
    print(f"\nLength stats (dpo_v1):")
    print(f"  chosen avg:  {report['chosen_avg_token_length']} tokens")
    print(f"  rejected avg: {report['rejected_avg_token_length']} tokens")
    print(f"  avg ratio:   {report['avg_token_length_ratio']}")
    print(f"\nFilter breakdown:")
    for reason, cnt in sorted(filter_counts.items()):
        print(f"  {reason}: {cnt}")
    print(f"\nOutputs:")
    print(f"  {cand_path}")
    print(f"  {v1_path}")
    print(f"  {report_path}")
    print(f"  {reg_path}")


if __name__ == "__main__":
    main()
