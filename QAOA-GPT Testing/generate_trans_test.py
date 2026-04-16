import random
import time
import networkx as nx

NUM_GRAPHS = 1000
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.5, 0.9)
OUT_FILE = "test_graphs_trans.txt"


def generate_connected_graph(seed):
    random.seed(seed)

    while True:
        p = random.uniform(*EDGE_PROB_RANGE)
        graph = nx.erdos_renyi_graph(NUM_QUBITS, p, seed=seed)
        if nx.is_connected(graph):
            break

    for u, v in graph.edges():
        weight = round(random.uniform(0.01, 1.0), 2)
        graph[u][v]["weight"] = weight

    return graph


with open(OUT_FILE, "w", encoding="utf-8") as f:
    base_seed = int(time.time())

    for i in range(NUM_GRAPHS):
        seed = base_seed + i
        graph = generate_connected_graph(seed)

        tokens = ["<score_elite>", "<bos>", "<maxcut_graph>"]

        for u, v in graph.edges():
            tokens.append(f"({u},{v})")
            tokens.append(str(graph[u][v]["weight"]))

        tokens.append("<end_of_maxcut_graph>")

        f.write(" ".join(tokens) + "\n")

print(f"Wrote {NUM_GRAPHS} graphs to {OUT_FILE}")
