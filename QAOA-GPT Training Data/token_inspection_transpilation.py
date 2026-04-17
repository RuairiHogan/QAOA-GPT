import json
from collections import Counter

INPUT_FILE = "transpilation_dataset_0_01.jsonl"
OUTPUT_FILE = "transpilation_dataset_0_01_canonical.jsonl"


def canonicalize_token(tok):
    if isinstance(tok, str):
        return tok
    if isinstance(tok, (int, float)):
        return str(tok)
    raise TypeError(f"Unsupported token type: {type(tok)}")


def build_transpilation_prefix(example):
    depth = int(example.get("transpiled_depth", -1))
    two_qubit_count = int(example.get("transpiled_two_qubit_count", -1))

    return [
        "<transpilation_metrics>",
        "<transpiled_depth>",
        depth,
        "<transpiled_two_qubit_count>",
        two_qubit_count,
        "<end_of_transpilation_metrics>",
    ]


def split_graph_and_circuit(tokens):
    """
    Split tokens into:
      graph_part (before <end_of_graph>)
      circuit_part (after <end_of_graph>)
    """
    if "<end_of_graph>" not in tokens:
        raise ValueError("Missing <end_of_graph>")

    idx = tokens.index("<end_of_graph>")
    graph_part = tokens[:idx]
    circuit_part = tokens[idx + 1:]

    return graph_part, circuit_part


def wrap_graph(graph_part):
    return ["<maxcut_graph>"] + graph_part + ["<end_of_maxcut_graph>"]


def wrap_circuit(circuit_part):
    return ["<circuit>"] + circuit_part + ["<end_of_circuit>"]


num_examples = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        example = json.loads(line)

        original_tokens = example["tokens"]

        if original_tokens and original_tokens[0] == "<bos>":
            after_bos = original_tokens[1:]
            graph_part, circuit_part = split_graph_and_circuit(after_bos)

            new_tokens = ["<bos>"]
            new_tokens.extend(build_transpilation_prefix(example))
            new_tokens.extend(wrap_graph(graph_part))
            new_tokens.extend(wrap_circuit(circuit_part))
        else:
            graph_part, circuit_part = split_graph_and_circuit(original_tokens)

            new_tokens = []
            new_tokens.extend(build_transpilation_prefix(example))
            new_tokens.extend(wrap_graph(graph_part))
            new_tokens.extend(wrap_circuit(circuit_part))

        example["tokens"] = [canonicalize_token(t) for t in new_tokens]

        fout.write(json.dumps(example) + "\n")
        num_examples += 1

print(f"Canonicalized {num_examples} examples.")
print(f"Saved to: {OUTPUT_FILE}")

# -------- CHECKING --------

token_counts = Counter()
lengths = []

with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        tokens = ex["tokens"]
        lengths.append(len(tokens))
        for t in tokens:
            token_counts[t] += 1

print("Num examples:", len(lengths))
print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Avg length:", sum(lengths) / len(lengths))
print("Top 20 tokens:", token_counts.most_common(20))

with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    ex = json.loads(next(f))
    print(type(ex["approx_ratio"]))
    print(type(ex["tokens"][2]))
    print(ex["tokens"][:40])
