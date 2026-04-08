import numpy as np
import networkx as nx
from qiskit import quantum_info as qi
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize
import json
import os
import time

#########################################################
# USER CONTROLS
#########################################################
NUM_GRAPHS = 100
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.3, 0.9)
MAX_DEPTH = 9
GAMMA0_GRID = [0.01, 0.1, 0.5, 1.0]
TARGET_AR = 0.97

# New output filename
DATASET_FILE = "connectivity_scored_dataset.jsonl"

SEED = int(time.time())
BACKEND_NAME = "ibm_kingston"
INITIAL_LAYOUT = [4, 5, 6, 7, 8, 9, 17]

# Hardware-aware score:
# score = AR - lambda_2q * (# two-qubit gates) - lambda_depth * depth
LAMBDA_2Q = 0.003
LAMBDA_DEPTH = 0.0005

# Reject very weak candidates before scoring
MIN_AR_TO_KEEP = 0.90

# Keep only the single best candidate per graph
TOP_K = 1

# "PQAOA" -> HB only
# "PDUAL" -> single + fully connected 2-qubit Pauli strings
OP_POOL_MODE = "PDUAL"

np.random.seed(SEED)

#########################################################
# IBM Kingston execution metadata
#########################################################
# KINGSTON_PHYSICAL_QUBITS = [4, 5, 6, 7, 8, 9, 17]
# LOGICAL_TO_PHYSICAL = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 17}
# PHYSICAL_TO_LOGICAL = {v: k for k, v in LOGICAL_TO_PHYSICAL.items()}

# KINGSTON_7Q_COUPLING_MAP = [
#     (0, 1),
#     (1, 2),
#     (2, 3),
#     (3, 4),
#     (4, 5),
#     (3, 6),
# ]

#########################################################
# Utility: exact OPT(G) for MaxCut
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

#########################################################
# MaxCut value Hamiltonian
#########################################################

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

#########################################################
# Precompute Pauli matrices
#########################################################

def build_single_qubit_paulis(n):
    X = []
    Y = []
    Z = []
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

#########################################################
# Fast application of exp(-i theta P)
#########################################################

def apply_pauli_exp_to_state(psi, P, theta, sign=-1):
    c = np.cos(theta)
    s = np.sin(theta)
    if sign == -1:
        return c * psi - 1j * s * (P @ psi)
    else:
        return c * psi + 1j * s * (P @ psi)

#########################################################
# Cost application
#########################################################

def apply_cost_to_state(psi, edge_terms, gamma):
    for (w, ZZ) in edge_terms:
        theta = gamma * w / 2.0
        psi = apply_pauli_exp_to_state(psi, ZZ, theta, sign=+1)
    return psi

#########################################################
# Full operator pool
#########################################################

def build_operator_list_and_mats(n, I, X, Y, Z, coupling_map=None):
    op_names = []
    op_mats = []

    if OP_POOL_MODE.upper() == "PQAOA":
        op_names.append("HB")
        op_mats.append(None)

    elif OP_POOL_MODE.upper() == "PDUAL":
        for i in range(n):
            op_names.append(f"X_{i}")
            op_mats.append(X[i])
            op_names.append(f"Y_{i}")
            op_mats.append(Y[i])
            op_names.append(f"Z_{i}")
            op_mats.append(Z[i])

        paulis = {"X": X, "Y": Y, "Z": Z}

        # Full operator set: all pairs i < j
        full_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        for (i, j) in full_pairs:
            for B in ["X", "Y", "Z"]:
                for C in ["X", "Y", "Z"]:
                    op_names.append(f"{B}_{i}_{C}_{j}")
                    op_mats.append(paulis[B][i] @ paulis[C][j])

    else:
        raise ValueError("OP_POOL_MODE must be 'PQAOA' or 'PDUAL'")

    return op_names, op_mats

#########################################################
# ADAPT gradient
#########################################################

def adapt_gradient(phi, Hc, Aj):
    comm = Hc @ Aj - Aj @ Hc
    return np.imag(np.vdot(phi, comm @ phi))

#########################################################
# Build/apply the circuit on a statevector
#########################################################

def apply_mixer_to_state(psi, op_index, beta, op_names, op_mats, X):
    name = op_names[op_index]
    if name == "HB":
        for Xi in X:
            psi = apply_pauli_exp_to_state(psi, Xi, beta, sign=-1)
        return psi
    else:
        P = op_mats[op_index]
        return apply_pauli_exp_to_state(psi, P, beta, sign=-1)


def build_state(n, edge_terms, op_names, op_mats, X, op_indices, betas, gammas):
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for k in range(len(op_indices)):
        psi = apply_cost_to_state(psi, edge_terms, gammas[k])
        psi = apply_mixer_to_state(psi, op_indices[k], betas[k], op_names, op_mats, X)
    return psi

#########################################################
# Approximation ratio
#########################################################

def approximation_ratio(psi, Hc, opt_val):
    val = np.real(np.vdot(psi, Hc @ psi))
    return val / opt_val if opt_val > 0 else 0.0

#########################################################
# Numeric handling
#########################################################

def clip_round(x, lo=-10.0, hi=10.0, decimals=2):
    x = float(x)
    if x < lo or x > hi:
        return None
    return round(x, decimals)

#########################################################
# ADAPT-QAOA run
#########################################################

def run_adapt_qaoa_once(graph, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, max_depth):
    n = graph.number_of_nodes()

    op_indices = []
    betas = []
    gammas = []

    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    for layer in range(max_depth):
        phi = apply_cost_to_state(psi, edge_terms, gamma0)

        grads = []
        for j, name in enumerate(op_names):
            if name == "HB":
                Aj = np.zeros((2**n, 2**n), dtype=complex)
                for Xi in X:
                    Aj += Xi
            else:
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
                gammas + [gamma_new]
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

#########################################################
# Tokenizer + writer
#########################################################

def tokenize_graph_and_circuit(graph, op_indices, betas, gammas):
    tokens = ["<bos>"]
    for (u, v) in graph.edges:
        w = round(float(graph[u][v]["weight"]), 2)
        tokens.append(f"({int(u)},{int(v)})")
        tokens.append(w)
    tokens.append("<end_of_graph>")

    for k in range(len(op_indices)):
        beta = clip_round(betas[k])
        gamma = clip_round(gammas[k])
        if beta is None or gamma is None:
            return None

        tokens.append(f"<new_layer_{k+1}>")
        tokens.append(int(op_indices[k]))
        tokens.append(gamma)
        tokens.append(beta)

    return tokens


def ar_tier(ar):
    if ar >= 0.97:
        return "elite"
    elif ar >= 0.94:
        return "good"
    elif ar >= 0.90:
        return "acceptable"
    else:
        return "poor"

#########################################################
# Compilation-aware circuit metrics
#########################################################

def build_cost_sparse_pauli(graph, n_qubits):
    labels = []
    coeffs = []

    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])
        chars = ["I"] * n_qubits
        chars[i] = "Z"
        chars[j] = "Z"
        labels.append("".join(chars)[::-1])
        coeffs.append(-0.5 * w)

    return SparsePauliOp(labels, coeffs=coeffs)


def pauli_label_from_op_name(op_name, n_qubits):
    chars = ["I"] * n_qubits
    parts = op_name.split("_")

    if len(parts) == 2:
        p, i = parts
        chars[int(i)] = p
    elif len(parts) == 4:
        p1, i, p2, j = parts
        chars[int(i)] = p1
        chars[int(j)] = p2
    else:
        raise ValueError(f"Unsupported operator name: {op_name}")

    return "".join(chars)[::-1]


def build_qiskit_circuit_from_layers(graph, op_indices, betas, gammas, op_names, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    cost_op = build_cost_sparse_pauli(graph, n_qubits)

    for op_index, beta, gamma in zip(op_indices, betas, gammas):
        qc.append(PauliEvolutionGate(cost_op, time=gamma), range(n_qubits))

        op_name = op_names[op_index]
        if op_name == "HB":
            raise ValueError("HB mode is not supported in this compilation-aware builder.")
        pauli_label = pauli_label_from_op_name(op_name, n_qubits)
        mixer_op = SparsePauliOp(pauli_label, coeffs=[1.0])
        qc.append(PauliEvolutionGate(mixer_op, time=beta), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def count_two_qubit_gates(qc):
    count = 0
    for instruction in qc.data:
        op = instruction.operation
        if getattr(op, "num_qubits", 0) == 2:
            count += 1
    return count


def compilation_metrics(graph, op_indices, betas, gammas, op_names, backend):
    qc = build_qiskit_circuit_from_layers(
        graph, op_indices, betas, gammas, op_names, graph.number_of_nodes()
    )

    tqc = transpile(
        qc,
        backend=backend,
        initial_layout=INITIAL_LAYOUT,
        optimization_level=1,
        seed_transpiler=1234,
    )

    depth = tqc.depth()
    two_qubit_count = count_two_qubit_gates(tqc)

    return depth, two_qubit_count


def hardware_score(ar, two_qubit_count, depth):
    return ar - LAMBDA_2Q * two_qubit_count - LAMBDA_DEPTH * depth


def format_elapsed_time(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

#########################################################
# Dataset writer
#########################################################

def write_dataset_entry(
    filename,
    graph,
    tokens,
    ar,
    tier,
    op_pool_mode,
    gamma0,
    depth,
    two_qubit_count,
    score,
):
    entry = {
        "num_qubits": graph.number_of_nodes(),
        "edge_prob_model": "erdos_renyi",
        "op_pool": op_pool_mode,
        "seed": SEED,
        "gamma0": gamma0,
        "approx_ratio": round(float(ar), 4),
        "tier": tier,
        "hardware_score": round(float(score), 6),
        "transpiled_depth": int(depth),
        "transpiled_two_qubit_count": int(two_qubit_count),
        "lambda_2q": LAMBDA_2Q,
        "lambda_depth": LAMBDA_DEPTH,
        "hardware_name": "ibm_kingston",
        "tokens": tokens
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

#########################################################
# Main
#########################################################

def main():
    start_time = time.time()

    if not os.path.exists(DATASET_FILE):
        open(DATASET_FILE, "w").close()

    written = 0
    attempts = 0

    print(f"Connecting to backend: {BACKEND_NAME}")
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    while written < NUM_GRAPHS and attempts < NUM_GRAPHS * 10:
        attempts += 1

        s = np.random.uniform(*EDGE_PROB_RANGE)
        G = nx.erdos_renyi_graph(NUM_QUBITS, s)
        if not nx.is_connected(G):
            continue

        for (u, v) in G.edges:
            w = float(np.random.uniform(0.0, 1.0))
            if w == 0.0:
                w = 1e-6
            G[u][v]["weight"] = w

        n = G.number_of_nodes()
        I, X, Y, Z = build_single_qubit_paulis(n)
        edge_terms = precompute_edge_ZZ(G)

        opt_val = maxcut_opt_bruteforce(G)
        Hc = cost_hamiltonian_Hc(G)

        # Full operator set: no coupling-map restriction in ADAPT
        op_names, op_mats = build_operator_list_and_mats(
            n, I, X, Y, Z, coupling_map=None
        )

        candidates = []

        for gamma0 in GAMMA0_GRID:
            op_indices, betas, gammas, ar = run_adapt_qaoa_once(
                G, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, MAX_DEPTH
            )

            if ar < MIN_AR_TO_KEEP:
                print(
                    f"Rejected candidate | n={NUM_QUBITS}, s={s:.2f}, gamma0={gamma0}, "
                    f"AR={ar:.4f} < {MIN_AR_TO_KEEP:.2f}"
                )
                continue

            tokens = tokenize_graph_and_circuit(G, op_indices, betas, gammas)
            if tokens is None:
                continue

            try:
                depth, two_qubit_count = compilation_metrics(
                    G, op_indices, betas, gammas, op_names, backend
                )
            except Exception as e:
                print(f"[WARN] Compilation failed for gamma0={gamma0}: {e}")
                continue

            score = hardware_score(ar, two_qubit_count, depth)
            tier = ar_tier(ar)

            print(
                f"Candidate | s={s:.2f}, gamma0={gamma0}, AR={ar:.4f}, "
                f"2Q={two_qubit_count}, depth={depth}, score={score:.6f}"
            )

            candidates.append({
                "tokens": tokens,
                "ar": ar,
                "tier": tier,
                "gamma0": gamma0,
                "depth": depth,
                "two_qubit_count": two_qubit_count,
                "score": score,
            })

        if len(candidates) == 0:
            continue

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:TOP_K]

        for cand in top_candidates:
            write_dataset_entry(
                DATASET_FILE,
                G,
                cand["tokens"],
                cand["ar"],
                cand["tier"],
                OP_POOL_MODE,
                cand["gamma0"],
                cand["depth"],
                cand["two_qubit_count"],
                cand["score"],
            )

            written += 1
            print(
                f"Wrote circuit {written}/{NUM_GRAPHS} | "
                f"AR={cand['ar']:.4f}, "
                f"2Q={cand['two_qubit_count']}, "
                f"depth={cand['depth']}, "
                f"score={cand['score']:.6f}"
            )

            if written >= NUM_GRAPHS:
                break

    elapsed = time.time() - start_time
    print(f"\nDone. Dataset saved to: {DATASET_FILE}")
    print(f"Total runtime: {format_elapsed_time(elapsed)}")

if __name__ == "__main__":
    main()
