import random
import time
import os
import numpy as np
import networkx as nx
from tqdm import tqdm

from qaoa_feather import QAOA_FEATHER as FEATHERG   

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
NUM_GRAPHS = 1000
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.5, 0.9)

OUT_GRAPH_FILE = "test_graphs.txt"
OUT_EMB_FILE   = "test_graph_emb.npy"

FEATHER_THETA_MAX = 2.5
FEATHER_EVAL_POINTS = 8
FEATHER_ORDER = 4
FEATHER_POOLING = "mean"

# ---------------------------------------------------
# GRAPH GENERATION
# ---------------------------------------------------
def generate_connected_graph(seed):
    random.seed(seed)

    while True:
        p = random.uniform(*EDGE_PROB_RANGE)
        G = nx.erdos_renyi_graph(NUM_QUBITS, p, seed=seed)
        if nx.is_connected(G):
            break

    # Assign random edge weights
    for u, v in G.edges():
        G[u][v]["weight"] = round(random.uniform(0.01, 1.0), 2)

    return G

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
graphs = []
graph_lines = []

base_seed = int(time.time())

print("Generating graphs...")
for i in tqdm(range(NUM_GRAPHS)):
    seed = base_seed + i
    G = generate_connected_graph(seed)
    graphs.append(G)

    # ---- serialize graph to tokens ----
    tokens = []
    tokens.append("<bos>")

    for u, v in G.edges():
        tokens.append(f"({u},{v})")
        tokens.append(f"{G[u][v]['weight']}")

    tokens.append("<end_of_graph>")
    tokens.append(f"<seed={seed}>")

    graph_lines.append(" ".join(tokens))

# ---------------------------------------------------
# WRITE GRAPH FILE
# ---------------------------------------------------
with open(OUT_GRAPH_FILE, "w", encoding="utf-8") as f:
    for line in graph_lines:
        f.write(line + "\n")

print(f"✅ Wrote {NUM_GRAPHS} graphs to {OUT_GRAPH_FILE}")

# ---------------------------------------------------
# FEATHER GRAPH EMBEDDINGS
# ---------------------------------------------------
print("Computing FEATHER graph embeddings...")

feather = FEATHERG(
    theta_max=FEATHER_THETA_MAX,
    eval_points=FEATHER_EVAL_POINTS,
    order=FEATHER_ORDER,
    pooling=FEATHER_POOLING,
)

feather.fit(graphs)
embeddings = feather.get_embedding().astype(np.float32)

# ---------------------------------------------------
# SAVE EMBEDDINGS
# ---------------------------------------------------
np.save(OUT_EMB_FILE, embeddings)

print(f"✅ Saved graph embeddings to {OUT_EMB_FILE}")
print(f"Embedding shape: {embeddings.shape}")

# ---------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------
assert embeddings.shape[0] == NUM_GRAPHS, "Mismatch between graphs and embeddings!"

print("🎯 Graphs and embeddings are index-aligned.")
