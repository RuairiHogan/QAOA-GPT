import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from qiskit import quantum_info as qi
from scipy.optimize import minimize

from model import GPTConfig, GPT

#########################################################
# CONFIG
#########################################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "test_graphs_trans.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "timing_comparison_trans_20.json")
OUTPUT_PLOT = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "timing_comparison_trans_20.png")

# plot mode: "stacked" or "grouped"
PLOT_MODE = "stacked"

NUM_GRAPHS = 20
NUM_QUBITS = 7

# ADAPT-QAOA settings
MAX_DEPTH = 9
GAMMA0_GRID = [0.01, 0.1, 0.5, 1.0]
TARGET_AR = 0.8
OP_POOL_MODE = "PDUAL"

# LLM settings
out_dir = "out-qaoa"
checkpoint_file = "second_transpilation.pt"
data_dir = "data/qaoa"
meta_file = "meta_for_both_trans.pkl"

device = "cpu"
max_new_tokens = 50
temperature = 0.8
top_k = 50
END_CIRCUIT_TOKEN = "<end_of_circuit>"

# Plot styling to match the trans test plotting file.
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
TICK_FONT_SIZE = 18

#########################################################
# SHARED PARSING HELPERS
#########################################################

def get_num_qubits_from_graph_tokens(tokens):
    nodes = set()
    for t in tokens:
        if t.startswith("(") and "," in t and t.endswith(")"):
            try:
                u, v = t.strip("()").split(",")
                nodes.add(int(u))
                nodes.add(int(v))
            except ValueError:
                pass
    return max(nodes) + 1 if nodes else 0


def parse_graph_from_prompt_tokens(tokens):
    if "<maxcut_graph>" not in tokens or "<end_of_maxcut_graph>" not in tokens:
        raise ValueError("Missing graph section")

    start = tokens.index("<maxcut_graph>") + 1
    end = tokens.index("<end_of_maxcut_graph>")
    graph_tokens = tokens[start:end]

    edges = []
    i = 0
    while i < len(graph_tokens):
        if i + 1 >= len(graph_tokens):
            raise ValueError("Incomplete edge-weight pair")

        edge_tok = graph_tokens[i]
        weight_tok = graph_tokens[i + 1]

        if not (edge_tok.startswith("(") and edge_tok.endswith(")") and "," in edge_tok):
            raise ValueError(f"Invalid edge token: {edge_tok}")

        u_str, v_str = edge_tok.strip("()").split(",")
        u, v = int(u_str), int(v_str)
        w = float(weight_tok)
        edges.append((u, v, w))
        i += 2

    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


def load_test_graph_prompts(input_file, num_graphs):
    prompts = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            tokens = [t.strip() for t in line.split() if t.strip() and not t.startswith("<seed=")]
            if not tokens:
                continue

            n_qubits = get_num_qubits_from_graph_tokens(tokens)
            if n_qubits != NUM_QUBITS:
                continue

            try:
                G = parse_graph_from_prompt_tokens(tokens)
            except Exception:
                continue

            prompts.append({
                "source_line": line_idx,
                "tokens": tokens,
                "graph": G,
            })

            if len(prompts) >= num_graphs:
                break

    if len(prompts) < num_graphs:
        raise ValueError(f"Only found {len(prompts)} valid graphs, needed {num_graphs}")

    return prompts

#########################################################
# ADAPT-QAOA CODE
#########################################################

def maxcut_opt_bruteforce(graph):
    n = graph.number_of_nodes()
    best = 0.0
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        cut = 0.0
        for (u, v) in graph.edges:
            w = float(graph[u][v]["weight"])
            if bits[u] != bits[v]:
                cut += w
        best = max(best, cut)
    return best


def cost_hamiltonian_Hc(graph):
    n = graph.number_of_nodes()
    paulis = []
    coeffs = []
    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])

        paulis.append(("I" * n)[::-1])
        coeffs.append(0.5 * w)

        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        paulis.append("".join(s)[::-1])
        coeffs.append(-0.5 * w)

    return qi.SparsePauliOp(paulis, coeffs).to_matrix()


def build_single_qubit_paulis(n):
    X, Y, Z = [], [], []
    for i in range(n):
        X.append(qi.Pauli(("I" * i + "X" + "I" * (n - i - 1))[::-1]).to_matrix())
        Y.append(qi.Pauli(("I" * i + "Y" + "I" * (n - i - 1))[::-1]).to_matrix())
        Z.append(qi.Pauli(("I" * i + "Z" + "I" * (n - i - 1))[::-1]).to_matrix())
    I = np.eye(2**n, dtype=complex)
    return I, X, Y, Z


def precompute_edge_ZZ(graph):
    n = graph.number_of_nodes()
    edge_terms = []
    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        ZZ = qi.Pauli("".join(s)[::-1]).to_matrix()
        edge_terms.append((w, ZZ))
    return edge_terms


def apply_pauli_exp_to_state(psi, P, theta, sign=-1):
    c = np.cos(theta)
    s = np.sin(theta)
    if sign == -1:
        return c * psi - 1j * s * (P @ psi)
    return c * psi + 1j * s * (P @ psi)


def apply_cost_to_state(psi, edge_terms, gamma):
    for (w, ZZ) in edge_terms:
        theta = gamma * w / 2.0
        psi = apply_pauli_exp_to_state(psi, ZZ, theta, sign=+1)
    return psi


def build_operator_list_and_mats(n, I, X, Y, Z, coupling_map=None):
    op_names = []
    op_mats = []

    if OP_POOL_MODE.upper() != "PDUAL":
        raise ValueError("This timing script expects OP_POOL_MODE='PDUAL'")

    for i in range(n):
        op_names.append(f"X_{i}")
        op_mats.append(X[i])
        op_names.append(f"Y_{i}")
        op_mats.append(Y[i])
        op_names.append(f"Z_{i}")
        op_mats.append(Z[i])

    paulis = {"X": X, "Y": Y, "Z": Z}

    if coupling_map is None:
        coupling_map = [(i, j) for i in range(n) for j in range(i + 1, n)]

    edge_set = sorted({
        tuple(sorted(e))
        for e in coupling_map
        if 0 <= e[0] < n and 0 <= e[1] < n and e[0] != e[1]
    })

    for (i, j) in edge_set:
        for B in ["X", "Y", "Z"]:
            for C in ["X", "Y", "Z"]:
                op_names.append(f"{B}_{i}_{C}_{j}")
                op_mats.append(paulis[B][i] @ paulis[C][j])

    return op_names, op_mats


def adapt_gradient(phi, Hc, Aj):
    comm = Hc @ Aj - Aj @ Hc
    return np.imag(np.vdot(phi, comm @ phi))


def apply_mixer_to_state(psi, op_index, beta, op_names, op_mats, X):
    P = op_mats[op_index]
    return apply_pauli_exp_to_state(psi, P, beta, sign=-1)


def build_state(n, edge_terms, op_names, op_mats, X, op_indices, betas, gammas):
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for k in range(len(op_indices)):
        psi = apply_cost_to_state(psi, edge_terms, gammas[k])
        psi = apply_mixer_to_state(psi, op_indices[k], betas[k], op_names, op_mats, X)
    return psi


def approximation_ratio(psi, Hc, opt_val):
    val = np.real(np.vdot(psi, Hc @ psi))
    return val / opt_val if opt_val > 0 else 0.0


def run_adapt_qaoa_once(graph, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, max_depth):
    n = graph.number_of_nodes()

    op_indices = []
    betas = []
    gammas = []

    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    for _ in range(max_depth):
        phi = apply_cost_to_state(psi, edge_terms, gamma0)

        grads = []
        for j in range(len(op_names)):
            Aj = op_mats[j]
            grads.append(abs(adapt_gradient(phi, Hc, Aj)))

        best_j = int(np.argmax(grads))
        op_indices.append(best_j)

        def objective_neg(x):
            beta_new, gamma_new = x
            psi_tmp = build_state(
                n, edge_terms, op_names, op_mats, X,
                op_indices,
                betas + [beta_new],
                gammas + [gamma_new],
            )
            return -np.real(np.vdot(psi_tmp, Hc @ psi_tmp))

        res = minimize(objective_neg, x0=[0.1, 0.1], method="Nelder-Mead")
        betas.append(float(res.x[0]))
        gammas.append(float(res.x[1]))

        psi = build_state(n, edge_terms, op_names, op_mats, X, op_indices, betas, gammas)
        ar = approximation_ratio(psi, Hc, opt_val)

        if ar >= TARGET_AR:
            break

    return op_indices, betas, gammas, ar


def generate_adapt_circuit_for_graph(G, opt_val):
    n = G.number_of_nodes()
    I, X, Y, Z = build_single_qubit_paulis(n)
    edge_terms = precompute_edge_ZZ(G)
    Hc = cost_hamiltonian_Hc(G)

    op_names, op_mats = build_operator_list_and_mats(n, I, X, Y, Z)

    best_result = None
    best_ar = -np.inf

    for gamma0 in GAMMA0_GRID:
        op_indices, betas, gammas, ar = run_adapt_qaoa_once(
            G, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, MAX_DEPTH
        )
        if ar > best_ar:
            best_ar = ar
            best_result = (op_indices, betas, gammas, ar)

    return best_result

#########################################################
# LLM CODE
#########################################################

with open(os.path.join(data_dir, meta_file), "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]


def encode(tokens):
    return torch.tensor([stoi[t] for t in tokens], dtype=torch.long)


def decode(indices):
    return [itos[i] for i in indices]


def trim_generated_circuit(out_tokens):
    cleaned = []
    for token in out_tokens:
        cleaned.append(token)
        if token == END_CIRCUIT_TOKEN:
            return cleaned
    return cleaned


checkpoint = torch.load(
    os.path.join(out_dir, checkpoint_file),
    map_location=device,
)

gptconf = GPTConfig(
    vocab_size=vocab_size,
    block_size=checkpoint["model_args"]["block_size"],
    n_layer=checkpoint["model_args"]["n_layer"],
    n_head=checkpoint["model_args"]["n_head"],
    n_embd=checkpoint["model_args"]["n_embd"],
    bias=checkpoint["model_args"]["bias"],
)

model = GPT(gptconf)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)


def generate_one_llm_circuit(prompt_tokens):
    x = encode(prompt_tokens)[None, :].to(device)

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    out_tokens = decode(y[0].tolist())
    generated_tokens = out_tokens[len(prompt_tokens):]
    return trim_generated_circuit(generated_tokens)

#########################################################
# TIMING + PLOTTING
#########################################################

def plot_times(graph_ids, adapt_times, llm_times, output_path):
    plt.rcParams["font.family"] = "Times New Roman"

    x = np.arange(len(graph_ids))
    width = 0.7

    fig, ax = plt.subplots(figsize=(13, 6.5))

    if PLOT_MODE == "stacked":
        ax.bar(x, llm_times, width, label="QAOA-GPT")
        ax.bar(x, adapt_times, width, bottom=llm_times, label="ADAPT-QAOA")
    else:
        w = 0.38
        ax.bar(x - w / 2, llm_times, w, label="QAOA-GPT")
        ax.bar(x + w / 2, adapt_times, w, label="ADAPT-QAOA")

    ax.set_xlabel("Circuit / graph ID", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Generation time (seconds)", fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(graph_ids, rotation=45)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)

    forced_ticks = [0.5]
    existing_ticks = list(ax.get_yticks())
    combined_ticks = sorted(set(existing_ticks + forced_ticks))
    ax.set_yticks(combined_ticks)

    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

#########################################################
# MAIN
#########################################################

def main():
    prompts = load_test_graph_prompts(INPUT_FILE, NUM_GRAPHS)

    graph_ids = []
    adapt_times = []
    llm_times = []
    results = []

    print(f"Timing {NUM_GRAPHS} trans graphs...\n")

    for idx, item in enumerate(prompts, start=1):
        prompt_tokens = item["tokens"]
        G = item["graph"]

        graph_id = f"G{idx}"
        graph_ids.append(graph_id)

        print(f"{graph_id}: source line {item['source_line']}")

        opt_val = maxcut_opt_bruteforce(G)

        t0 = time.perf_counter()
        adapt_result = generate_adapt_circuit_for_graph(G, opt_val=opt_val)
        adapt_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        llm_result = generate_one_llm_circuit(prompt_tokens)
        llm_elapsed = time.perf_counter() - t1

        adapt_times.append(adapt_elapsed)
        llm_times.append(llm_elapsed)

        layer_tokens = [tok for tok in llm_result if tok.startswith("<new_layer_")]

        results.append({
            "graph_id": graph_id,
            "source_line": item["source_line"],
            "adapt_time_seconds_excluding_bruteforce": adapt_elapsed,
            "llm_time_seconds": llm_elapsed,
            "adapt_num_layers": len(adapt_result[0]) if adapt_result is not None else None,
            "adapt_ar": float(adapt_result[3]) if adapt_result is not None else None,
            "llm_num_layers": len(layer_tokens) if llm_result is not None else None,
        })

        print(f"  ADAPT-QAOA (no brute force): {adapt_elapsed:.4f}s")
        print(f"  LLM                     : {llm_elapsed:.4f}s\n")

    summary = {
        "num_graphs": NUM_GRAPHS,
        "plot_mode": PLOT_MODE,
        "adapt_mean_time_seconds_excluding_bruteforce": float(np.mean(adapt_times)),
        "adapt_median_time_seconds_excluding_bruteforce": float(np.median(adapt_times)),
        "llm_mean_time_seconds": float(np.mean(llm_times)),
        "llm_median_time_seconds": float(np.median(llm_times)),
        "mean_speedup_adapt_over_llm": float(np.mean(np.array(adapt_times) / np.array(llm_times))),
    }

    payload = {
        "summary": summary,
        "per_graph_results": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_times(
        graph_ids=graph_ids,
        adapt_times=adapt_times,
        llm_times=llm_times,
        output_path=OUTPUT_PLOT,
    )

    print("Done.")
    print(f"Saved JSON to: {OUTPUT_JSON}")
    print(f"Saved plot to: {OUTPUT_PLOT}")
    print(f"Mean ADAPT time (no brute force): {summary['adapt_mean_time_seconds_excluding_bruteforce']:.4f}s")
    print(f"Mean LLM time               : {summary['llm_mean_time_seconds']:.4f}s")
    print(f"Mean speedup                : {summary['mean_speedup_adapt_over_llm']:.2f}x")


if __name__ == "__main__":
    main()