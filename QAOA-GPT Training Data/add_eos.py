import json

INPUT_FILE = "qaoa_gpt_dataset_elite.jsonl"
OUTPUT_FILE = "qaoa_gpt_dataset_eos.jsonl"

kept = 0
discarded = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):
        line.append("<end_of_circuit>")
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] Line {line_num}: invalid JSON, skipping")
            continue

        fout.write(json.dumps(entry) + "\n")


print(f"Done.")

