import numpy as np
import networkx as nx
import torch
from scipy.optimize import minimize
import json
import os
import time

#########################################################
# USER CONTROLS
#########################################################
NUM_GRAPHS = 10
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.3, 0.9)
MAX_DEPTH = 9
GAMMA0_GRID = [0.01, 0.1, 0.5, 1.0]
TARGET_AR = 0.97
DATASET_FILE = "qaoa_gpt_dataset_kingston_7q_gpu.jsonl"
SEED = int(time.time())
OP_POOL_MODE = "PDUAL"

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_REAL = torch.float64
DTYPE_COMPLEX = torch.complex128

print(f"Using device: {DEVICE}")

#########################################################
# IBM Kingston 7-qubit subgraph choice
#########################################################
KINGSTON_PHYSICAL_QUBITS = [4, 5, 6, 7, 8, 9, 17]
LOGICAL_TO_PHYSICAL = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 17}
PHYSICAL_TO_LOGICAL = {v: k for k, v in LOGICAL_TO_PHYSICAL.items()}

KINGSTON_7Q_COUPLING_MAP = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (3, 6),
]

#########################################################
# Hardware connectivity config
#########################################################

class HardwareConfig:
    def __init__(self, n_qubits, coupling_map):
        self.n_qubits = n_qubits
        self.coupling_map = sorted({tuple(sorted(edge)) for edge in coupling_map})
        self.coupling_set = set(self.coupling_map)

    def is_edge_allowed(self, i, j):
        return tuple(sorted((i, j))) in self.coupling_set

    def __repr__(self):
        return f"HardwareConfig(n_qubits={self.n_qubits}, coupling_map={self.coupling_map})"


def is_operator_allowed(op_name, hardware):
    if op_name.startswith(("X_", "Y_", "Z_")) and op_name.count("_") == 1:
        return True

    parts = op_name.split("_")
    if len(parts) == 4:
        _, i_str, _, j_str = parts
        try:
            i = int(i_str)
            j = int(j_str)
        except ValueError:
            return False
        return hardware.is_edge_allowed(i, j)

    return False

#########################################################
# Torch Pauli helpers
#########################################################

def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out

def single_qubit_op(n, target, pauli_char):
    I2 = torch.eye(2, dtype=DTYPE_COMPLEX, device=DEVICE)
    X2 = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE_COMPLEX, device=DEVICE)
    Y2 = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE_COMPLEX, device=DEVICE)
    Z2 = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE_COMPLEX, device=DEVICE)

    lookup = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

    ops = []
    for q in range(n):
        ops.append(lookup[pauli_char] if q == target else I2)
    return kron_n(ops)

def two_qubit_pauli_op(n, i, pi, j, pj):
    I2 = torch.eye(2, dtype=DTYPE_COMPLEX, device=DEVICE)
    X2 = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE_COMPLEX, device=DEVICE)
    Y2 = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE_COMPLEX, device=DEVICE)
    Z2 = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE_COMPLEX, device=DEVICE)
    lookup = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

    ops = []
    for q in range(n):
        if q == i:
            ops.append(lookup[pi])
        elif q == j:
            ops.append(lookup[pj])
        else:
            ops.append(I2)
    return kron_n(ops)

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
# MaxCut value Hamiltonian Hc
#########################################################

def cost_hamiltonian_Hc(graph):
    n = graph.number_of_nodes()
    dim = 2**n
    Hc = torch.zeros((dim, dim), dtype=DTYPE_COMPLEX, device=DEVICE)
    I_full = torch.eye(dim, dtype=DTYPE_COMPLEX, device=DEVICE)

    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])
        ZZ = two_qubit_pauli_op(n, i, "Z", j, "Z")
        Hc = Hc + 0.5 * w * (I_full - ZZ)

    return Hc

#########################################################
# Precompute Pauli matrices
#########################################################

def build_single_qubit_paulis(n):
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(single_qubit_op(n, i, "X"))
        Y.append(single_qubit_op(n, i, "Y"))
        Z.append(single_qubit_op(n, i, "Z"))
    I = torch.eye(2**n, dtype=DTYPE_COMPLEX, device=DEVICE)
    return I, X, Y, Z

def precompute_edge_ZZ(graph):
    n = graph.number_of_nodes()
    edge_terms = []
    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])
        ZZ = two_qubit_pauli_op(n, i, "Z", j, "Z")
        edge_terms.append((w, ZZ))
    return edge_terms

#########################################################
# Fast Pauli exponential on statevector
#########################################################

def apply_pauli_exp_to_state(psi, P, theta, sign=-1):
    theta_t = torch.tensor(theta, dtype=DTYPE_REAL, device=DEVICE)
    c = torch.cos(theta_t)
    s = torch.sin(theta_t)
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
# Operator pool
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

        if coupling_map is None:
            coupling_map = [(i, j) for i in range(n) for j in range(i + 1, n)]

        edge_set = sorted({
            tuple(sorted(e))
            for e in coupling_map
            if 0 <= e[0] < n and 0 <= e[1] < n and e[0] != e[1]
        })

        hardware = HardwareConfig(n, edge_set)

        for (i, j) in edge_set:
            for B in ["X", "Y", "Z"]:
                for C in ["X", "Y", "Z"]:
                    op_name = f"{B}_{i}_{C}_{j}"
                    if not is_operator_allowed(op_name, hardware):
                        continue
                    op_names.append(op_name)
                    op_mats.append(paulis[B][i] @ paulis[C][j])
    else:
        raise ValueError("OP_POOL_MODE must be 'PQAOA' or 'PDUAL'")

    return op_names, op_mats

#########################################################
# ADAPT gradient
#########################################################

def adapt_gradient(phi, Hc, Aj):
    comm = Hc @ Aj - Aj @ Hc
    val = torch.vdot(phi, comm @ phi)
    return torch.imag(val).item()

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
    psi = torch.ones(2**n, dtype=DTYPE_COMPLEX, device=DEVICE) / np.sqrt(2**n)
    for k in range(len(op_indices)):
        psi = apply_cost_to_state(psi, edge_terms, gammas[k])
        psi = apply_mixer_to_state(psi, op_indices[k], betas[k], op_names, op_mats, X)
    return psi

#########################################################
# Approximation ratio
#########################################################

def approximation_ratio(psi, Hc, opt_val):
    val = torch.real(torch.vdot(psi, Hc @ psi)).item()
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

    psi = torch.ones(2**n, dtype=DTYPE_COMPLEX, device=DEVICE) / np.sqrt(2**n)

    for layer in range(max_depth):
        phi = apply_cost_to_state(psi, edge_terms, gamma0)

        grads = []
        for j, name in enumerate(op_names):
            if name == "HB":
                Aj = torch.zeros((2**n, 2**n), dtype=DTYPE_COMPLEX, device=DEVICE)
                for Xi in X:
                    Aj = Aj + Xi
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
            val = torch.real(torch.vdot(psi_tmp, Hc @ psi_tmp)).item()
            return -val

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

def write_dataset_entry(filename, graph, tokens, ar, tier, op_pool_mode, gamma0, coupling_map):
    entry = {
        "num_qubits": graph.number_of_nodes(),
        "edge_prob_model": "erdos_renyi",
        "op_pool": op_pool_mode,
        "seed": SEED,
        "gamma0": gamma0,
        "approx_ratio": round(float(ar), 4),
        "tier": tier,
        "hardware_name": "ibm_kingston_7q_subgraph",
        "physical_qubits": KINGSTON_PHYSICAL_QUBITS,
        "logical_to_physical": LOGICAL_TO_PHYSICAL,
        "hardware_coupling_map": coupling_map,
        "device": str(DEVICE),
        "tokens": tokens
    }
    with open(filename, "a") as f:
        f.write(json.dumps(entry) + "\n")

#########################################################
# Test
#########################################################

def test_hardware_constrained_operator_pool():
    n = 7
    coupling_map = KINGSTON_7Q_COUPLING_MAP
    hw = HardwareConfig(n, coupling_map)
    I, X, Y, Z = build_single_qubit_paulis(n)

    op_names, _ = build_operator_list_and_mats(n, I, X, Y, Z, coupling_map=hw.coupling_map)

    expected_singles = (
        {f"X_{i}" for i in range(n)}
        | {f"Y_{i}" for i in range(n)}
        | {f"Z_{i}" for i in range(n)}
    )

    expected_two_qubit = {
        f"{B}_{i}_{C}_{j}"
        for (i, j) in coupling_map
        for B in ["X", "Y", "Z"]
        for C in ["X", "Y", "Z"]
    }

    assert expected_singles.issubset(set(op_names)), "Missing single-qubit operators"
    assert expected_two_qubit.issubset(set(op_names)), "Missing allowed two-qubit operators"

    forbidden = {
        "Z_0_Z_6",
        "Z_1_Z_5",
        "Z_2_Z_4",
        "X_0_Y_4",
    }
    assert forbidden.isdisjoint(set(op_names)), "Forbidden two-qubit operators should be excluded"

    print("✅ Kingston 7-qubit hardware-constrained operator pool test passed")

#########################################################
# Main
#########################################################

if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        open(DATASET_FILE, "w").close()

    written = 0
    attempts = 0

    test_hardware_constrained_operator_pool()

    hardware = HardwareConfig(NUM_QUBITS, coupling_map=KINGSTON_7Q_COUPLING_MAP)

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

        op_names, op_mats = build_operator_list_and_mats(
            n, I, X, Y, Z, coupling_map=hardware.coupling_map
        )

        for gamma0 in GAMMA0_GRID:
            op_indices, betas, gammas, ar = run_adapt_qaoa_once(
                G, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, MAX_DEPTH
            )
            tier = ar_tier(ar)
            if tier == "poor":
                print(
                    f"Rejected circuit | n={NUM_QUBITS}, "
                    f"s={s:.2f}, gamma0={gamma0}, "
                    f"AR={ar:.4f} < 0.90"
                )
                continue

            tokens = tokenize_graph_and_circuit(G, op_indices, betas, gammas)
            if tokens is None:
                continue

            write_dataset_entry(
                DATASET_FILE,
                G,
                tokens,
                ar,
                tier,
                OP_POOL_MODE,
                gamma0,
                hardware.coupling_map
            )
            written += 1
            print(f"Wrote circuit {written}/{NUM_GRAPHS} (AR={ar:.4f}, s={s:.2f}, gamma0={gamma0})")

            if written >= NUM_GRAPHS:
                break

    print(f"\nDone. Dataset saved to: {DATASET_FILE}")