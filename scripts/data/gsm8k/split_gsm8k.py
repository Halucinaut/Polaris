import json
import os
import random

os.makedirs("data/math/gsm8k/split", exist_ok=True)

samples = []
with open("data/math/gsm8k/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line.strip()))

random.seed(42)
random.shuffle(samples)

train_n = 210
val_n = 30
test_n = 30
review_n = 30

splits = {
    "train": samples[:train_n],
    "val": samples[train_n : train_n + val_n],
    "test": samples[train_n + val_n : train_n + val_n + test_n],
    "review": samples[train_n + val_n + test_n : train_n + val_n + test_n + review_n],
}

for name, data in splits.items():
    out_path = f"data/math/gsm8k/split/{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{name}: {len(data)} samples -> {out_path}")
