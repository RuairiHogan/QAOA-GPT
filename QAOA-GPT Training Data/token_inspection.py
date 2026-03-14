import json
from collections import Counter
INPUT_FILE = "qaoa_gpt_dataset_elite.jsonl"
OUTPUT_FILE = "qaoa_gpt_dataset_canonical.jsonl"

def canonicalize_token(tok):
    """
    Convert all tokens to strings.
    Floats and ints are stringified with no loss of meaning.
    """
    if isinstance(tok, str):
        return tok
    elif isinstance(tok, (int, float)):
        return str(tok)
    else:
        raise TypeError(f"Unsupported token type: {type(tok)}")

num_examples = 0

with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
    for line in fin:
        example = json.loads(line)

        # Canonicalize tokens
        example["tokens"] = [canonicalize_token(t) for t in example["tokens"]]

        fout.write(json.dumps(example) + "\n")
        num_examples += 1

print(f"Canonicalized {num_examples} examples.")
print(f"Saved to: {OUTPUT_FILE}")

# Checking data
token_counts = Counter()
lengths = []

with open("qaoa_gpt_dataset_elite.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        tokens = ex["tokens"]
        lengths.append(len(tokens))
        for t in tokens:
            token_counts[str(t)] += 1

print("Num examples:", len(lengths))
print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Avg length:", sum(lengths)/len(lengths))
print("Top 20 tokens:", token_counts.most_common(20))

with open("qaoa_gpt_dataset_canonical.jsonl") as f:
    ex = json.loads(next(f))
    print(type(ex["approx_ratio"]))        # should be float
    print(type(ex["tokens"][2]))   
