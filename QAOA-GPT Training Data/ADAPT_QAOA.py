import numpy as np
import networkx as nx
from qiskit import quantum_info as qi
from scipy.optimize import minimize
import json
import os
import time

#########################################################
# USER CONTROLS (paper-aligned defaults)
#########################################################
# 7, 9 works with 0.01 and 0.1 = gamma0
NUM_GRAPHS = 10
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.3, 0.9)
MAX_DEPTH = 9
GAMMA0_GRID = [0.01, 0.1, 0.5, 1.0]
TARGET_AR = 0.97
DATASET_FILE = "qaoa_gpt_dataset.jsonl"
SEED = int(time.time())

# "PQAOA" -> HB only
# "PDUAL" -> single + 2-qubit Pauli strings (Eq. 4 style pool)
OP_POOL_MODE = "PDUAL"

np.random.seed(SEED)

#########################################################
# Utility: exact OPT(G) for MaxCut (brute force; ok for small n)
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
# MaxCut "value Hamiltonian" Hc so <psi|Hc|psi> = expected cut value
# Hc = sum_{(i,j)} w_ij * (I - Z_i Z_j)/2
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
# Precompute Pauli matrices you’ll reuse a lot
#########################################################

def build_single_qubit_paulis(n):
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(qi.Pauli(("I"*i + "X" + "I"*(n-i-1))[::-1]).to_matrix())
        Y.append(qi.Pauli(("I"*i + "Y" + "I"*(n-i-1))[::-1]).to_matrix())
        Z.append(qi.Pauli(("I"*i + "Z" + "I"*(n-i-1))[::-1]).to_matrix())
    I = np.eye(2**n, dtype=complex)
    return I, X, Y, Z

def precompute_edge_ZZ(graph):
    """Store ZZ matrices per edge so we don't rebuild them every call."""
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
# Fast application of exp(-i theta P) to a statevector when P^2 = I
# exp(-iθP)|ψ> = cos(θ)|ψ> - i sin(θ) P|ψ>
#########################################################

def apply_pauli_exp_to_state(psi, P, theta, sign=-1):
    # sign = -1 gives exp(-i theta P)
    # sign = +1 gives exp(+i theta P)
    c = np.cos(theta)
    s = np.sin(theta)
    if sign == -1:
        return c * psi - 1j * s * (P @ psi)
    else:
        return c * psi + 1j * s * (P @ psi)

#########################################################
# Cost application: U_C(gamma) = ∏ exp(+i gamma (w/2) Z_i Z_j)
# (identity part of Hc is global phase, safe to ignore)
#########################################################

def apply_cost_to_state(psi, edge_terms, gamma):
    for (w, ZZ) in edge_terms:
        theta = gamma * w / 2.0
        psi = apply_pauli_exp_to_state(psi, ZZ, theta, sign=+1)
    return psi

#########################################################
# Operator pool (paper-compatible)
#########################################################

def build_operator_list_and_mats(n, I, X, Y, Z):
    """
    Returns:
      op_names: list[str]
      op_mats:  list[np.ndarray or None]

    For PQAOA:
      operator 0 is HB = sum_i X_i (handled specially as product of exp(-iβX_i)).

    For PDUAL:
      we store Pauli-string matrices (all square to I), so closed-form works.
    """
    op_names = []
    op_mats = []

    if OP_POOL_MODE.upper() == "PQAOA":
        op_names.append("HB")   # special: not a single Pauli string
        op_mats.append(None)

    elif OP_POOL_MODE.upper() == "PDUAL":
        # Singles
        for i in range(n):
            op_names.append(f"X{i}")
            op_mats.append(X[i])
            op_names.append(f"Y{i}")
            op_mats.append(Y[i])
            op_names.append(f"Z{i}")
            op_mats.append(Z[i])

        # Two-qubit Pauli strings B_i C_j for i<j, B,C in {X,Y,Z}
        paulis = {"X": X, "Y": Y, "Z": Z}
        for i in range(n):
            for j in range(i + 1, n):
                for B in ["X", "Y", "Z"]:
                    for C in ["X", "Y", "Z"]:
                        op_names.append(f"{B}{i}{C}{j}")
                        op_mats.append(paulis[B][i] @ paulis[C][j])

    else:
        raise ValueError("OP_POOL_MODE must be 'PQAOA' or 'PDUAL'")

    return op_names, op_mats

#########################################################
# ADAPT gradient: g_j = Im(<phi|[Hc, A_j]|phi>)
#########################################################

def adapt_gradient(phi, Hc, Aj):
    comm = Hc @ Aj - Aj @ Hc
    return np.imag(np.vdot(phi, comm @ phi))

#########################################################
# Build/apply the circuit on a statevector (NO eigenvalues)
#########################################################

def apply_mixer_to_state(psi, op_index, beta, op_names, op_mats, X):
    name = op_names[op_index]
    if name == "HB":
        # exp(-iβ sum_i X_i) = ∏ exp(-iβ X_i) since X_i commute
        for Xi in X:
            psi = apply_pauli_exp_to_state(psi, Xi, beta, sign=-1)
        return psi
    else:
        # Pauli string (squares to I): exp(-iβP)
        P = op_mats[op_index]
        return apply_pauli_exp_to_state(psi, P, beta, sign=-1)

def build_state(n, edge_terms, op_names, op_mats, X, op_indices, betas, gammas):
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)  # |+>^n
    for k in range(len(op_indices)):
        psi = apply_cost_to_state(psi, edge_terms, gammas[k])
        psi = apply_mixer_to_state(psi, op_indices[k], betas[k], op_names, op_mats, X)
    return psi

#########################################################
# Approximation ratio alpha = <psi|Hc|psi> / OPT(G)
#########################################################

def approximation_ratio(psi, Hc, opt_val):
    val = np.real(np.vdot(psi, Hc @ psi))
    return val / opt_val if opt_val > 0 else 0.0

#########################################################
# Numeric handling (paper): round 2 decimals, clip to [-10,10], discard if out
#########################################################

def clip_round(x, lo=-10.0, hi=10.0, decimals=2):
    x = float(x)
    if x < lo or x > hi:
        return None
    return round(x, decimals)

#########################################################
# ADAPT-QAOA run (one gamma0 initialization)
#########################################################

def run_adapt_qaoa_once(graph, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, max_depth):
    n = graph.number_of_nodes()

    op_indices = []
    betas = []
    gammas = []

    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    for layer in range(max_depth):
        # phi = e^{-i gamma0 Hc}|psi>  (we approximate using the ZZ-only cost unitary)
        phi = apply_cost_to_state(psi, edge_terms, gamma0)

        # Choose best operator by |gradient|
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

        # Optimize new (beta, gamma) by re-evaluating the full circuit
        def objective_neg(x):
            beta_new, gamma_new = x
            psi_tmp = build_state(
                n, edge_terms, op_names, op_mats, X,
                op_indices,
                betas + [beta_new],
                gammas + [gamma_new]
            )
            return -np.real(np.vdot(psi_tmp, Hc @ psi_tmp))

        # res = minimize(
        #     objective_neg,
        #     x0=[0.1, 0.1],
        #     method="COBYLA",
        #     options={
        #         "maxiter": 50,
        #         "rhobeg": 0.5
        #     }
        # )
        res = minimize(objective_neg, x0=[0.1, 0.1], method="Nelder-Mead")


        betas.append(float(res.x[0]))
        gammas.append(float(res.x[1]))

        psi = build_state(n, edge_terms, op_names, op_mats, X, op_indices, betas, gammas)
        ar = approximation_ratio(psi, Hc, opt_val)

        if ar >= TARGET_AR:
            break

    return op_indices, betas, gammas, ar

#########################################################
# Tokenizer + writer (paper Step 3 format)
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
        tokens.append(int(op_indices[k]))  # operator index o_k
        tokens.append(gamma)               # γ then β (paper ordering)
        tokens.append(beta)

    tokens.append("<end_of_circuit>")

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


def write_dataset_entry(filename, graph, tokens, ar, tier, op_pool_mode, gamma0):
    entry = {
        "num_qubits": graph.number_of_nodes(),
        "edge_prob_model": "erdos_renyi",
        "op_pool": op_pool_mode,
        "seed": SEED,
        "gamma0": gamma0,
        "approx_ratio": round(float(ar), 4),
        "tier": tier,
        "tokens": tokens
    }
    with open(filename, "a") as f:
        f.write(json.dumps(entry) + "\n")

#########################################################
# Main
#########################################################

if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        open(DATASET_FILE, "w").close()

    written = 0
    attempts = 0

    while written < NUM_GRAPHS and attempts < NUM_GRAPHS * 10:
        attempts += 1

        s = np.random.uniform(*EDGE_PROB_RANGE)
        G = nx.erdos_renyi_graph(NUM_QUBITS, s)
        if not nx.is_connected(G):
            continue

        # Weights ~ U(0,1]
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

        op_names, op_mats = build_operator_list_and_mats(n, I, X, Y, Z)

        for gamma0 in GAMMA0_GRID:
            op_indices, betas, gammas, ar = run_adapt_qaoa_once(
                G, Hc, opt_val, edge_terms, op_names, op_mats, I, X, gamma0, MAX_DEPTH
            )
            tier = ar_tier(ar)
            if tier == "poor":
                print(
                    f"Rejected circuit | n={NUM_QUBITS}, "
                    f"s={s:.2f}, gamma0={gamma0}, "
                    f"AR={ar:.4f} < {TARGET_AR}"
                )
                continue

            tokens = tokenize_graph_and_circuit(G, op_indices, betas, gammas)
            if tokens is None:
                continue

            write_dataset_entry(DATASET_FILE, G, tokens, ar, tier, OP_POOL_MODE, gamma0)
            written += 1
            print(f"Wrote circuit {written}/{NUM_GRAPHS} (AR={ar:.4f}, s={s:.2f}, gamma0={gamma0})")

            if written >= NUM_GRAPHS:
                break

    print(f"\nDone. Dataset saved to: {DATASET_FILE}")
