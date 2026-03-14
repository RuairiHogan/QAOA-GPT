import json
import numpy as np
import networkx as nx
from tqdm import tqdm

from qaoa_feather import QAOA_FEATHER


# ---------------------------------
# CONFIG
# ---------------------------------
JSONL_PATH = "../QAOA-GPT Training Data/qaoa_gpt_dataset_canonical.jsonl"
OUT_PATH   = "graph_embeddings.npy"

THETA_MAX   = 2.5
EVAL_POINTS = 8      # controls embedding size
ORDER       = 4      # 4 × 2 × 8 × 2 = 128 dims
POOLING     = "mean"


# ---------------------------------
# GRAPH PARSER
# ---------------------------------
def parse_graph_from_tokens(tokens):
    G = nx.Graph()

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "<end_of_graph>":
            break

        if tok.startswith("(") and tok.endswith(")"):
            u, v = tok[1:-1].split(",")
            w = float(tokens[i + 1])
            G.add_edge(int(u), int(v), weight=w)
            i += 2
        else:
            i += 1

    return G


# ---------------------------------
# LOAD GRAPHS
# ---------------------------------
graphs = []

with open(JSONL_PATH, "r") as f:
    for line in tqdm(f, desc="Reading graphs"):
        obj = json.loads(line)

        tokens = obj["tokens"]
        G = parse_graph_from_tokens(tokens)

        # sanity check: nodes must be 0..N-1
        nodes = sorted(G.nodes)
        if nodes and nodes != list(range(max(nodes) + 1)):
            raise ValueError(f"Non-contiguous node indices: {nodes}")

        graphs.append(G)

print(f"Loaded {len(graphs)} graphs")


# ---------------------------------
# FEATHER EMBEDDING
# ---------------------------------
model = QAOA_FEATHER(
    theta_max=THETA_MAX,
    eval_points=EVAL_POINTS,
    order=ORDER,
    pooling=POOLING,
)

model.fit(graphs)
embeddings = model.get_embedding().astype(np.float32)


# ---------------------------------
# SAVE
# ---------------------------------
np.save(OUT_PATH, embeddings)

print("Saved embeddings:")
print(" ", OUT_PATH)
print(" shape:", embeddings.shape)
