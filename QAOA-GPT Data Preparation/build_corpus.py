import json

INPUT = "../QAOA-GPT Training Data/qaoa_gpt_dataset_canonical.jsonl"
OUTPUT = "train.txt"

with open(INPUT, "r") as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        entry = json.loads(line)

        tier = entry.get("tier", "unknown")
        tier_token = f"<tier_{tier}>"

        tokens = entry["tokens"]

        # prepend tier token
        full_sequence = [tier_token] + tokens

        fout.write(" ".join(full_sequence) + "\n")

print("Corpus written to train.txt")
