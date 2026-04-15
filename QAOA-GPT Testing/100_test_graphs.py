import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import quantum_info as qi

########################################
# CONFIG
########################################

INPUT_FILE = "generated_circuits_kingston_100x10.txt"
OUTPUT_JSON = "ideal_simulation_results_100x10.json"
OUTPUT_PLOT_INDEX = "ideal_simulation_best_vs_avg_ar_by_graph.png"
OUTPUT_PLOT_SORTED = "ideal_simulation_best_vs_avg_ar_sorted.png"
OUTPUT_PLOT_SCATTER = "ideal_simulation_best_vs_avg_ar_scatter.png"

CIRCUITS_PER_GRAPH = 10
EXPECTED_NUM_GRAPHS = 100

# Increase these if you want the bottom of the graph lower
INDEX_Y_PAD = 0.3
SCATTER_Y_PAD = 0.3

plt.rcParams["font.family"] = "Times New Roman"


KINGSTON_7Q_COUPLING_MAP = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (3, 6),
]

########################################
# UTILITIES (MATCH TRAINING)
########################################

def maxcut_opt_bruteforce(graph):
    n = graph.number_of_nodes()
    best = 0.0
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        cut = 0.0
        for (u, v) in graph.edges:
            if bits[u] != bits[v]:
                cut += float(graph[u][v]["weight"])
        best = max(best, cut)
    return best


def cost_hamiltonian_Hc(graph):
    n = graph.number_of_nodes()
    paulis, coeffs = [], []

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


def apply_pauli_exp_to_state(psi, P, theta, sign):
    c = np.cos(theta)
    s = np.sin(theta)
    return c * psi + (1j * sign * s) * (P @ psi)

########################################
# OPERATOR POOL (IDENTICAL ORDER TO TRAINING)
########################################

def build_operator_pool(n, coupling_map=None):
    X, Y, Z = [], [], []

    for i in range(n):
        X.append(qi.Pauli(("I" * i + "X" + "I" * (n - i - 1))[::-1]).to_matrix())
        Y.append(qi.Pauli(("I" * i + "Y" + "I" * (n - i - 1))[::-1]).to_matrix())
        Z.append(qi.Pauli(("I" * i + "Z" + "I" * (n - i - 1))[::-1]).to_matrix())

    op_mats = []

    for i in range(n):
        op_mats.append(X[i])
        op_mats.append(Y[i])
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
                op_mats.append(paulis[B][i] @ paulis[C][j])

    return op_mats

########################################
# TOKEN PARSING
########################################

def parse_tokens(tokens):
    if not tokens:
        raise ValueError("Empty line")

    required = [
        "<maxcut_graph>",
        "<end_of_maxcut_graph>",
        "<circuit>",
        "<end_of_circuit>",
    ]
    for tok in required:
        if tok not in tokens:
            raise ValueError(f"Missing {tok}")

    g_start = tokens.index("<maxcut_graph>") + 1
    g_end = tokens.index("<end_of_maxcut_graph>")
    c_start = tokens.index("<circuit>") + 1
    c_end = tokens.index("<end_of_circuit>")

    if g_end <= g_start:
        raise ValueError("Empty graph section")
    if c_end < c_start:
        raise ValueError("Bad circuit section")

    graph_tokens = tokens[g_start:g_end]
    circuit_tokens = tokens[c_start:c_end]

    edges = []
    i = 0
    while i < len(graph_tokens):
        if i + 1 >= len(graph_tokens):
            raise ValueError("Incomplete edge-weight pair in graph section")

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

    ops, gammas, betas = [], [], []
    i = 0
    while i < len(circuit_tokens):
        tok = circuit_tokens[i]

        if not tok.startswith("<new_layer_"):
            raise ValueError(f"Unexpected token in circuit section: {tok}")

        if i + 3 >= len(circuit_tokens):
            raise ValueError(f"Incomplete layer block starting at {tok}")

        ops.append(int(circuit_tokens[i + 1]))
        gammas.append(float(circuit_tokens[i + 2]))
        betas.append(float(circuit_tokens[i + 3]))
        i += 4

    if len(ops) == 0:
        raise ValueError("No circuit layers found")

    return G, ops, gammas, betas

########################################
# CIRCUIT EVALUATION
########################################

def evaluate_circuit(G, ops, gammas, betas, op_mats):
    n = G.number_of_nodes()
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    edge_terms = []
    for (i, j) in G.edges:
        w = float(G[i][j]["weight"])
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        ZZ = qi.Pauli("".join(s)[::-1]).to_matrix()
        edge_terms.append((w, ZZ))

    for k in range(len(ops)):
        for w, ZZ in edge_terms:
            psi = apply_pauli_exp_to_state(psi, ZZ, gammas[k] * w / 2.0, sign=+1)

        op_idx = ops[k]
        if op_idx < 0 or op_idx >= len(op_mats):
            raise ValueError(f"Operator index {op_idx} out of range (pool size {len(op_mats)})")

        psi = apply_pauli_exp_to_state(psi, op_mats[op_idx], betas[k], sign=-1)

    Hc = cost_hamiltonian_Hc(G)
    return np.real(np.vdot(psi, Hc @ psi))

########################################
# PLOTTING HELPERS
########################################

def compute_y_limits(*series, pad=0.01, upper_cap=1.0):
    flat = []
    for s in series:
        flat.extend([float(x) for x in s if x is not None])

    ymin = min(flat) - pad
    ymax = min(upper_cap, max(flat) + 0.005)

    ymin = max(0.0, ymin)
    if ymax <= ymin:
        ymax = ymin + 0.05

    return ymin, ymax


def style_axis(ax, title, ymin, ymax):
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Graph index", fontsize=12)
    ax.set_ylabel("Approximation ratio", fontsize=12)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=9)

########################################
# PLOTTING
########################################

def make_plot_by_graph(best_ars, avg_ars, output_plot):
    x = np.arange(1, len(best_ars) + 1)

    overall_best_mean = float(np.mean(best_ars))
    overall_avg_mean = float(np.mean(avg_ars))
    ymin, ymax = compute_y_limits(
        best_ars, avg_ars, [overall_best_mean, overall_avg_mean], pad=INDEX_Y_PAD
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(x, best_ars, linewidth=2.2, label="Best AR of 10 samples")
    ax.plot(x, avg_ars, linewidth=2.2, label="Average AR of 10 samples")

    ax.axhline(
        overall_best_mean,
        linestyle="--",
        linewidth=1.8,
        label=f"Mean best AR = {overall_best_mean:.3f}",
    )
    ax.axhline(
        overall_avg_mean,
        linestyle=":",
        linewidth=1.8,
        label=f"Mean average AR = {overall_avg_mean:.3f}",
    )

    style_axis(ax, "Ideal-Simulation Approximation Ratio Across Test Graphs", ymin, ymax)
    ax.set_xlim(1, len(best_ars))

    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plot_sorted(best_ars, avg_ars, output_plot):
    best_sorted = np.sort(np.array(best_ars))[::-1]
    avg_sorted = np.sort(np.array(avg_ars))[::-1]
    x = np.arange(1, len(best_sorted) + 1)

    overall_best_mean = float(np.mean(best_sorted))
    overall_avg_mean = float(np.mean(avg_sorted))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(x, best_sorted, linewidth=2.4, label="Best AR of 10 samples (sorted)")
    ax.plot(x, avg_sorted, linewidth=2.4, label="Average AR of 10 samples (sorted)")

    ax.axhline(
        overall_best_mean,
        linestyle="--",
        linewidth=1.8,
        label=f"Mean best AR = {overall_best_mean:.3f}",
    )
    ax.axhline(
        overall_avg_mean,
        linestyle=":",
        linewidth=1.8,
        label=f"Mean average AR = {overall_avg_mean:.3f}",
    )

    style_axis(ax, "Sorted Ideal-Simulation Approximation Ratio Across Test Graphs", 0.0, 1.0)
    ax.set_xlim(1, len(best_sorted))

    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_scatter_plot(best_ars, avg_ars, output_plot):
    x = np.arange(1, len(best_ars) + 1)

    best_mean = float(np.mean(best_ars))
    avg_mean = float(np.mean(avg_ars))
    best_std = float(np.std(best_ars))
    avg_std = float(np.std(avg_ars))

    ymin, ymax = compute_y_limits(
        best_ars, avg_ars, [best_mean, avg_mean], pad=SCATTER_Y_PAD
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for xi, ya, yb in zip(x, avg_ars, best_ars):
        ax.plot([xi, xi], [ya, yb], linewidth=1.0, alpha=0.45)

    ax.scatter(
        x, best_ars,
        s=28, alpha=0.8, marker="o",
        label="Best AR of 10 samples"
    )
    ax.scatter(
        x, avg_ars,
        s=28, alpha=0.8, marker="s",
        label="Average AR of 10 samples"
    )

    ax.axhline(
        best_mean,
        linestyle="--",
        linewidth=1.8,
        label=f"Avg. Best AR: {best_mean:.4f} (±{best_std:.4f})"
    )
    ax.axhline(
        avg_mean,
        linestyle=":",
        linewidth=1.8,
        label=f"Avg. Mean AR: {avg_mean:.4f} (±{avg_std:.4f})"
    )

    style_axis(ax, "Ideal-Simulation Approximation Ratio Across Test Graphs", ymin, ymax)
    ax.set_xlim(1, len(best_ars))
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

########################################
# MAIN TEST
########################################

def test_grouped_circuits(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)
    print(f"Loaded {total_lines} generated circuits.\n")

    if total_lines % CIRCUITS_PER_GRAPH != 0:
        raise ValueError(
            f"Expected number of lines to be divisible by {CIRCUITS_PER_GRAPH}, "
            f"but got {total_lines}"
        )

    num_graphs = total_lines // CIRCUITS_PER_GRAPH
    print(f"Detected {num_graphs} graphs with {CIRCUITS_PER_GRAPH} circuits each.\n")

    if num_graphs != EXPECTED_NUM_GRAPHS:
        print(f"Warning: expected {EXPECTED_NUM_GRAPHS} graphs, but found {num_graphs}.")

    graph_results = []
    failed_total = 0

    for graph_idx in range(num_graphs):
        start = graph_idx * CIRCUITS_PER_GRAPH
        end = start + CIRCUITS_PER_GRAPH
        group_lines = lines[start:end]

        group_ars = []
        group_details = []

        print(f"Evaluating graph {graph_idx + 1}/{num_graphs}")

        reference_graph_edges = None
        reference_opt = None

        for local_idx, line in enumerate(group_lines):
            global_idx = start + local_idx

            try:
                tokens = line.split()
                G, ops, gammas, betas = parse_tokens(tokens)

                if G.number_of_nodes() != 7:
                    raise ValueError(f"Expected 7-qubit graph, got {G.number_of_nodes()} qubits")

                current_edges = sorted(
                    (min(u, v), max(u, v), float(G[u][v]["weight"]))
                    for (u, v) in G.edges
                )

                if reference_graph_edges is None:
                    reference_graph_edges = current_edges
                elif current_edges != reference_graph_edges:
                    raise ValueError("Graph mismatch inside 10-sample group")

                op_mats = build_operator_pool(
                    G.number_of_nodes(),
                    coupling_map=KINGSTON_7Q_COUPLING_MAP
                )

                approx = evaluate_circuit(G, ops, gammas, betas, op_mats)

                if reference_opt is None:
                    reference_opt = maxcut_opt_bruteforce(G)

                ar = approx / reference_opt if reference_opt > 0 else 0.0
                group_ars.append(ar)

                group_details.append({
                    "global_circuit_index": global_idx + 1,
                    "sample_in_graph": local_idx + 1,
                    "approx_value": float(approx),
                    "optimal_value": float(reference_opt),
                    "approx_ratio": float(ar),
                    "num_layers": len(ops),
                    "ops": ops,
                    "gammas": gammas,
                    "betas": betas,
                })

                print(f"  sample {local_idx + 1:2d}: AR = {ar:.4f}")

            except Exception as e:
                failed_total += 1
                group_details.append({
                    "global_circuit_index": global_idx + 1,
                    "sample_in_graph": local_idx + 1,
                    "error": str(e),
                })
                print(f"  sample {local_idx + 1:2d}: FAILED ({e})")

        valid_ars = np.array(group_ars, dtype=float)

        if len(valid_ars) == 0:
            best_ar = None
            avg_ar = None
        else:
            best_ar = float(valid_ars.max())
            avg_ar = float(valid_ars.mean())

        graph_results.append({
            "graph_index": graph_idx + 1,
            "best_ar": best_ar,
            "average_ar": avg_ar,
            "num_valid_samples": int(len(valid_ars)),
            "num_failed_samples": int(CIRCUITS_PER_GRAPH - len(valid_ars)),
            "samples": group_details,
        })

        if best_ar is not None:
            print(f"  -> best AR = {best_ar:.4f}, average AR = {avg_ar:.4f}")
        else:
            print("  -> all 10 samples failed")

        print()

    valid_best = [g["best_ar"] for g in graph_results if g["best_ar"] is not None]
    valid_avg = [g["average_ar"] for g in graph_results if g["average_ar"] is not None]

    summary = {
        "input_file": test_file,
        "circuits_per_graph": CIRCUITS_PER_GRAPH,
        "num_graphs": num_graphs,
        "total_circuits": total_lines,
        "total_failed_circuits": failed_total,
        "mean_best_ar": float(np.mean(valid_best)) if valid_best else None,
        "mean_average_ar": float(np.mean(valid_avg)) if valid_avg else None,
        "std_best_ar": float(np.std(valid_best)) if valid_best else None,
        "std_average_ar": float(np.std(valid_avg)) if valid_avg else None,
        "max_best_ar": float(np.max(valid_best)) if valid_best else None,
        "min_best_ar": float(np.min(valid_best)) if valid_best else None,
        "max_average_ar": float(np.max(valid_avg)) if valid_avg else None,
        "min_average_ar": float(np.min(valid_avg)) if valid_avg else None,
        "plot_by_graph_file": OUTPUT_PLOT_INDEX,
        "plot_sorted_file": OUTPUT_PLOT_SORTED,
        "plot_scatter_file": OUTPUT_PLOT_SCATTER,
    }

    output = {
        "summary": summary,
        "graph_results": graph_results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("======================================")
    print("FINAL SUMMARY")
    print("======================================")
    print(f"Graphs evaluated        : {num_graphs}")
    print(f"Total circuits          : {total_lines}")
    print(f"Failed circuits         : {failed_total}")
    print()
    if valid_best:
        print(f"Mean best AR            : {summary['mean_best_ar']:.4f}")
        print(f"Mean average AR         : {summary['mean_average_ar']:.4f}")
        print(f"Best graph-level AR     : {summary['max_best_ar']:.4f}")
        print(f"Worst graph-level AR    : {summary['min_best_ar']:.4f}")
    else:
        print("No valid graph results.")
    print("======================================\n")

    if valid_best and valid_avg:
        make_plot_by_graph(valid_best, valid_avg, OUTPUT_PLOT_INDEX)
        make_plot_sorted(valid_best, valid_avg, OUTPUT_PLOT_SORTED)
        make_scatter_plot(valid_best, valid_avg, OUTPUT_PLOT_SCATTER)
        print(f"Saved plot to: {OUTPUT_PLOT_INDEX}")
        print(f"Saved plot to: {OUTPUT_PLOT_SORTED}")
        print(f"Saved plot to: {OUTPUT_PLOT_SCATTER}")

    print(f"Saved JSON results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    test_grouped_circuits(INPUT_FILE)