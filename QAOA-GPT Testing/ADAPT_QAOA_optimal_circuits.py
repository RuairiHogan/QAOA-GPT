import math
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import quantum_info as qi
from scipy.optimize import minimize


TARGET_SUCCESSFUL_CIRCUITS = 50
MAX_DEPTH = 9
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.3, 0.9)
WEIGHT_RANGE = (0.01, 1.0)
EARLY_WRITE_AR = 0.98
OPTIMIZER_METHOD = "L-BFGS-B"
GAMMA0_GRID = [0.01, 0.1, 0.5, 1.0]

BASE_DIR = Path(__file__).resolve().parent
CIRCUITS_OUT_FILE = BASE_DIR / "generated_circuits_adapt_optimal_50.txt"
GRAPHS_OUT_FILE = BASE_DIR / "test_graphs_trans_adapt_optimal_50.txt"


def generate_connected_graph(seed):
    random.seed(seed)

    while True:
        p = random.uniform(*EDGE_PROB_RANGE)
        graph = nx.erdos_renyi_graph(NUM_QUBITS, p, seed=seed)
        if nx.is_connected(graph):
            break

    for u, v in graph.edges():
        graph[u][v]["weight"] = round(random.uniform(*WEIGHT_RANGE), 2)

    return graph


def maxcut_opt_bruteforce(graph):
    n = graph.number_of_nodes()
    best = 0.0
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        cut = 0.0
        for u, v in graph.edges:
            if bits[u] != bits[v]:
                cut += graph[u][v]["weight"]
        best = max(best, cut)
    return best


def cost_hamiltonian_Hc(graph):
    n = graph.number_of_nodes()
    paulis = []
    coeffs = []

    for i, j in graph.edges:
        w = float(graph[i][j]["weight"])

        paulis.append(("I" * n)[::-1])
        coeffs.append(0.5 * w)

        chars = ["I"] * n
        chars[i] = "Z"
        chars[j] = "Z"
        paulis.append("".join(chars)[::-1])
        coeffs.append(-0.5 * w)

    return qi.SparsePauliOp(paulis, coeffs).to_matrix()


def build_operator_pool(n):
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
    for i in range(n):
        for j in range(i + 1, n):
            for B in ["X", "Y", "Z"]:
                for C in ["X", "Y", "Z"]:
                    op_mats.append(paulis[B][i] @ paulis[C][j])

    return op_mats


def build_edge_terms(graph):
    n = graph.number_of_nodes()
    edge_terms = []
    for i, j in graph.edges:
        w = float(graph[i][j]["weight"])
        chars = ["I"] * n
        chars[i] = "Z"
        chars[j] = "Z"
        ZZ = qi.Pauli("".join(chars)[::-1]).to_matrix()
        edge_terms.append((w, ZZ))
    return edge_terms


def apply_pauli_exp_to_state(psi, P, theta, sign):
    c = np.cos(theta)
    s = np.sin(theta)
    return c * psi + (1j * sign * s) * (P @ psi)


def evolve_state(edge_terms, op_mats, ops, gammas, betas, n_qubits):
    psi = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)

    for layer_idx, op_index in enumerate(ops):
        gamma = gammas[layer_idx]
        beta = betas[layer_idx]

        for weight, ZZ in edge_terms:
            psi = apply_pauli_exp_to_state(psi, ZZ, gamma * weight / 2, sign=+1)

        psi = apply_pauli_exp_to_state(psi, op_mats[op_index], beta, sign=-1)

    return psi


def circuit_value(Hc, edge_terms, op_mats, ops, gammas, betas, n_qubits):
    psi = evolve_state(edge_terms, op_mats, ops, gammas, betas, n_qubits)
    return float(np.real(np.vdot(psi, Hc @ psi)))


def adapt_operator_score(psi, Hc, op_mat):
    # ADAPT-style proxy gradient: large values indicate the operator can most
    # strongly change the energy from the current state.
    return abs(2.0 * np.imag(np.vdot(psi, op_mat @ (Hc @ psi))))


def optimize_layer_params(Hc, edge_terms, op_mats, ops, n_qubits):
    depth = len(ops)
    best_params = None
    best_value = -math.inf

    def objective(params):
        gammas = params[:depth]
        betas = params[depth:]
        value = circuit_value(Hc, edge_terms, op_mats, ops, gammas, betas, n_qubits)
        return -value

    initial_points = []
    for gamma0 in GAMMA0_GRID:
        gammas0 = np.full(depth, gamma0)
        betas0 = np.full(depth, gamma0)
        initial_points.append(np.concatenate([gammas0, betas0]))

    bounds = [(-math.pi, math.pi)] * (2 * depth)

    for x0 in initial_points:
        print(
            f"    Trying optimizer start gamma/beta="
            f"{x0[0]:.2f} at depth {depth}"
        )
        result = minimize(
            objective,
            x0=x0,
            method=OPTIMIZER_METHOD,
            bounds=bounds,
        )

        if not result.success and result.fun is None:
            continue

        value = -float(result.fun)
        if value > best_value:
            best_value = value
            best_params = result.x

    if best_params is None:
        raise RuntimeError("Parameter optimization failed for all restarts")

    gammas = best_params[:depth].tolist()
    betas = best_params[depth:].tolist()
    return gammas, betas, best_value


def find_optimal_adapt_circuit(graph, op_mats):
    n = graph.number_of_nodes()
    Hc = cost_hamiltonian_Hc(graph)
    edge_terms = build_edge_terms(graph)
    optimum = maxcut_opt_bruteforce(graph)

    ops = []
    gammas = []
    betas = []
    best_solution = None

    for depth in range(1, MAX_DEPTH + 1):
        current_psi = evolve_state(edge_terms, op_mats, ops, gammas, betas, n)

        candidate_scores = [
            (adapt_operator_score(current_psi, Hc, op_mat), idx)
            for idx, op_mat in enumerate(op_mats)
        ]
        candidate_scores.sort(reverse=True)

        selected_op = candidate_scores[0][1]
        selected_score = candidate_scores[0][0]
        ops.append(selected_op)

        print(
            f"  Depth {depth}: selected operator {selected_op} "
            f"(score={selected_score:.6f})"
        )

        gammas, betas, best_value = optimize_layer_params(Hc, edge_terms, op_mats, ops, n)
        ar = best_value / optimum if optimum > 0 else 0.0

        print(
            f"  Depth {depth}: value={best_value:.6f}, opt={optimum:.6f}, "
            f"AR={ar:.6f}"
        )

        best_solution = {
            "ops": ops.copy(),
            "gammas": gammas,
            "betas": betas,
            "value": best_value,
            "optimum": optimum,
            "depth": depth,
            "approx_ratio": ar,
        }

        if ar >= EARLY_WRITE_AR:
            print(
                f"  Depth {depth}: reached early-write threshold "
                f"(AR={ar:.6f} >= {EARLY_WRITE_AR:.2f})"
            )
            return best_solution

    return best_solution


def format_graph_tokens_old(graph):
    tokens = ["<bos>"]
    for u, v in graph.edges():
        tokens.append(f"({u},{v})")
        tokens.append(str(graph[u][v]["weight"]))
    tokens.append("<end_of_graph>")
    return " ".join(tokens)


def format_graph_tokens_trans(graph):
    tokens = ["<score_elite>", "<bos>", "<maxcut_graph>"]
    for u, v in graph.edges():
        tokens.append(f"({u},{v})")
        tokens.append(str(graph[u][v]["weight"]))
    tokens.append("<end_of_maxcut_graph>")
    return " ".join(tokens)


def format_circuit_line(graph, solution):
    tokens = format_graph_tokens_old(graph).split()
    for layer_idx, (op_idx, gamma, beta) in enumerate(
        zip(solution["ops"], solution["gammas"], solution["betas"]),
        start=1,
    ):
        tokens.extend(
            [
                f"<new_layer_{layer_idx}>",
                str(op_idx),
                f"{gamma:.6f}",
                f"{beta:.6f}",
            ]
        )
    tokens.append("<tier_elite>")
    return " ".join(tokens)


def main():
    np.random.seed(int(time.time()))
    op_mats = build_operator_pool(NUM_QUBITS)

    written = 0
    base_seed = int(time.time())

    with open(CIRCUITS_OUT_FILE, "w", encoding="utf-8") as circuits_f, open(
        GRAPHS_OUT_FILE, "w", encoding="utf-8"
    ) as graphs_f:
        for attempt in range(1, TARGET_SUCCESSFUL_CIRCUITS + 1):
            seed = base_seed + attempt - 1

            graph = generate_connected_graph(seed)
            print(
                f"\nAttempt {attempt}: generated graph with {graph.number_of_edges()} edges "
                f"(seed={seed})"
            )
            solution = find_optimal_adapt_circuit(graph, op_mats)

            circuits_f.write(format_circuit_line(graph, solution) + "\n")
            graphs_f.write(format_graph_tokens_trans(graph) + "\n")
            written += 1

            if solution["approx_ratio"] >= EARLY_WRITE_AR:
                status = "early threshold met"
            else:
                status = f"max depth {MAX_DEPTH} reached"

            print(
                f"[{attempt}] Saved circuit {written}/{TARGET_SUCCESSFUL_CIRCUITS} "
                f"({status}, "
                f"(depth={solution['depth']}, value={solution['value']:.6f}, "
                f"opt={solution['optimum']:.6f}, AR={solution['approx_ratio']:.6f})"
            )

    print()
    print(f"Circuits written: {written}")
    print(f"Graphs attempted: {TARGET_SUCCESSFUL_CIRCUITS}")
    print(f"Circuits file: {CIRCUITS_OUT_FILE}")
    print(f"Graphs file: {GRAPHS_OUT_FILE}")


if __name__ == "__main__":
    main()
