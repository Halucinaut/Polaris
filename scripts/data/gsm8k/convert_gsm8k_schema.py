import json
import os
import re

SPLIT_DIR = "data/math/gsm8k/split"


def extract_final_answer(answer_text: str) -> str:
    """Extract final answer after #### marker."""
    match = re.search(r"####\s*(.+)$", answer_text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def convert_split(split_name: str) -> None:
    in_path = os.path.join(SPLIT_DIR, f"{split_name}.jsonl")
    out_path = os.path.join(SPLIT_DIR, f"{split_name}_converted.jsonl")

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            raw = json.loads(line.strip())
            converted = {
                "problem_id": f"gsm8k_{split_name}_{idx:04d}",
                "problem": raw["question"],
                "answer": extract_final_answer(raw["answer"]),
                "solution": raw["answer"],
                "source": "gsm8k",
                "domain": "grade_school_math",
                "split": split_name,
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"{split_name}: converted -> {out_path}")


for split in ["train", "val", "test", "review"]:
    convert_split(split)
