import json

INPUT_FILE = "qaoa_gpt_dataset.jsonl"
OUTPUT_FILE = "qaoa_gpt_dataset_elite.jsonl"

kept = 0
discarded = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] Line {line_num}: invalid JSON, skipping")
            continue

        if entry.get("tier") == "elite":
            fout.write(json.dumps(entry) + "\n")
            kept += 1
        else:
            discarded += 1

print(f"Done.")
print(f"Kept elite circuits     : {kept}")
print(f"Discarded non-elite     : {discarded}")
print(f"Output written to       : {OUTPUT_FILE}")
