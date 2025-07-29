import json
def run(train, val, test,prefix="data_set/final/egypt_pdf_qa"):
    with open(f"{prefix}_train.jsonl", "a", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"{prefix}_val.jsonl", "a", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"{prefix}_test.jsonl", "a", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(train)} train and {len(val)} validation samples.")
