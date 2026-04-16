import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = "old_dataset.jsonl"
CIRCUITS_OUT_FILE = BASE_DIR / "generated_circuits_adapt_optimal_50.txt"
GRAPHS_OUT_FILE = BASE_DIR / "test_graphs_trans_adapt_optimal_50.txt"
AR_TOL = 1e-9


def tokens_to_circuit_line(tokens):
    return " ".join(str(token) for token in tokens) + " <tier_elite>"


def tokens_to_trans_graph_line(tokens):
    graph_tokens = ["<score_elite>", "<bos>", "<maxcut_graph>"]

    i = 1  # skip <bos>
    while i < len(tokens) and tokens[i] != "<end_of_graph>":
        graph_tokens.append(str(tokens[i]))
        graph_tokens.append(str(tokens[i + 1]))
        i += 2

    graph_tokens.append("<end_of_maxcut_graph>")
    return " ".join(graph_tokens)


def main():
    written = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as in_f, open(
        CIRCUITS_OUT_FILE, "w", encoding="utf-8"
    ) as circuits_f, open(GRAPHS_OUT_FILE, "w", encoding="utf-8") as graphs_f:
        for line_num, raw in enumerate(in_f, start=1):
            raw = raw.strip()
            if not raw:
                continue

            entry = json.loads(raw)
            ar = float(entry.get("approx_ratio", 0.0))

            if abs(ar - 1.0) > AR_TOL:
                continue

            tokens = entry["tokens"]
            circuits_f.write(tokens_to_circuit_line(tokens) + "\n")
            graphs_f.write(tokens_to_trans_graph_line(tokens) + "\n")
            written += 1

            print(f"Wrote AR=1 circuit from line {line_num} ({written} total)")

    print()
    print(f"Total AR=1 circuits written: {written}")
    print(f"Circuits file: {CIRCUITS_OUT_FILE}")
    print(f"Graphs file: {GRAPHS_OUT_FILE}")


if __name__ == "__main__":
    main()
