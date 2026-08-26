"""Convert train probe JSONL to eval_gsm8k compatible format."""
import json, sys

input_path = "data/math/probes/dpo_v2_style_train_probe_30.jsonl"
output_path = "data/math/probes/dpo_v2_style_train_probe_30_eval.jsonl"

records = []
with open(input_path, encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))

with open(output_path, "w", encoding="utf-8") as f:
    for r in records:
        out = {
            "problem_id": r["problem_id"],
            "problem": r["problem"],
            "answer": r["answer"],
            "source": "dpo_v2_style_train_probe",
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Converted {len(records)} records → {output_path}")
