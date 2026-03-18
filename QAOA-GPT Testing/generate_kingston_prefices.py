import random
import time
import networkx as nx

NUM_GRAPHS = 1000
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.5, 0.9)
OUT_FILE = "test_graphs_kingston.txt"

# ---------------------------------------------------
# Kingston 7-qubit logical coupling map
# ---------------------------------------------------
KINGSTON_COUPLING_MAP = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (3, 6),
]

# ---------------------------------------------------
# Generate connected random graph
# ---------------------------------------------------

def generate_connected_graph(seed):
    random.seed(seed)

    while True:
        p = random.uniform(*EDGE_PROB_RANGE)
        G = nx.erdos_renyi_graph(NUM_QUBITS, p, seed=seed)
        if nx.is_connected(G):
            break

    # Assign random weights
    for u, v in G.edges():
        w = round(random.uniform(0.01, 1.0), 2)
        G[u][v]["weight"] = w

    return G

# ---------------------------------------------------
# Write dataset
# ---------------------------------------------------

with open(OUT_FILE, "w", encoding="utf-8") as f:
    base_seed = int(time.time())

    for i in range(NUM_GRAPHS):
        seed = base_seed + i
        G = generate_connected_graph(seed)

        tokens = []

        # ---- BOS ----
        tokens.append('"<bos>"')

        # ---- HARDWARE ----
        tokens.append('"<IBM_KINGSTON>"')
        tokens.append('"<hardware_graph>"')

        for (u, v) in KINGSTON_COUPLING_MAP:
            tokens.append(f'"({u},{v})"')

        tokens.append('"<end_of_hardware_graph>"')

        # ---- MAXCUT GRAPH ----
        tokens.append('"<maxcut_graph>"')

        for u, v in G.edges():
            tokens.append(f'"({u},{v})"')
            tokens.append(f'"{G[u][v]["weight"]}"')

        tokens.append('"<end_of_maxcut_graph>"')

        # # ---- CIRCUIT START ----
        # tokens.append('"<circuit>"')

        # # ---- SEED ----
        # tokens.append(f'"<seed={seed}>"')

        # Write line
        f.write(",".join(tokens) + "\n")

print(f"✅ Wrote {NUM_GRAPHS} graphs with Kingston hardware to {OUT_FILE}")