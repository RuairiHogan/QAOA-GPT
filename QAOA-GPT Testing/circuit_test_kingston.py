import numpy as np
import networkx as nx
from qiskit import quantum_info as qi

########################################
# FIXED KINGSTON 7-QUBIT COUPLING MAP
########################################

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
    """Exact same Hc used in training."""
    n = graph.number_of_nodes()
    paulis, coeffs = [], []

    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])

        # + w/2 * I
        paulis.append(("I" * n)[::-1])
        coeffs.append(0.5 * w)

        # - w/2 * ZiZj
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        paulis.append("".join(s)[::-1])
        coeffs.append(-0.5 * w)

    return qi.SparsePauliOp(paulis, coeffs).to_matrix()


def apply_pauli_exp_to_state(psi, P, theta, sign):
    """
    sign = +1 -> exp(+i theta P)
    sign = -1 -> exp(-i theta P)
    """
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

    # singles first: X_i, Y_i, Z_i
    for i in range(n):
        op_mats.append(X[i])
        op_mats.append(Y[i])
        op_mats.append(Z[i])

    # then two-qubit strings on allowed edges only
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
# TOKEN PARSING FOR FORMAT:
# <maxcut_graph> ... <end_of_maxcut_graph> <circuit> ... <end_of_circuit>
########################################

def parse_tokens(tokens):
    if not tokens:
        raise ValueError("Empty line")

    if "<maxcut_graph>" not in tokens:
        raise ValueError("Missing <maxcut_graph>")
    if "<end_of_maxcut_graph>" not in tokens:
        raise ValueError("Missing <end_of_maxcut_graph>")
    if "<circuit>" not in tokens:
        raise ValueError("Missing <circuit>")
    if "<end_of_circuit>" not in tokens:
        raise ValueError("Missing <end_of_circuit>")

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

    # Parse graph edge-weight pairs
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

    # Precompute cost ZZ terms
    edge_terms = []
    for (i, j) in G.edges:
        w = float(G[i][j]["weight"])
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        ZZ = qi.Pauli("".join(s)[::-1]).to_matrix()
        edge_terms.append((w, ZZ))

    # Apply each layer: cost then mixer/operator
    for k in range(len(ops)):
        # U_C(gamma_k)
        for w, ZZ in edge_terms:
            psi = apply_pauli_exp_to_state(psi, ZZ, gammas[k] * w / 2.0, sign=+1)

        # U_M(beta_k)
        op_idx = ops[k]
        if op_idx < 0 or op_idx >= len(op_mats):
            raise ValueError(f"Operator index {op_idx} out of range (pool size {len(op_mats)})")

        psi = apply_pauli_exp_to_state(psi, op_mats[op_idx], betas[k], sign=-1)

    Hc = cost_hamiltonian_Hc(G)
    return np.real(np.vdot(psi, Hc @ psi))

########################################
# MAIN TEST
########################################

def test_all_circuits(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    ratios = []
    failed = 0
    perfect = 0

    print(f"Testing {len(lines)} circuits...\n")

    for idx, line in enumerate(lines):
        try:
            tokens = line.split()
            G, ops, gammas, betas = parse_tokens(tokens)

            if G.number_of_nodes() != 7:
                raise ValueError(f"Expected 7-qubit graph, got {G.number_of_nodes()} qubits")

            op_mats = build_operator_pool(
                G.number_of_nodes(),
                coupling_map=KINGSTON_7Q_COUPLING_MAP
            )

            approx = evaluate_circuit(G, ops, gammas, betas, op_mats)
            opt = maxcut_opt_bruteforce(G)
            ar = approx / opt if opt > 0 else 0.0
            ratios.append(ar)

            if np.isclose(ar, 1.0, atol=1e-6):
                perfect += 1

            print(f"Circuit {idx + 1:3d}: AR = {ar:.4f}")

        except Exception as e:
            failed += 1
            print(f"Circuit {idx + 1:3d}: FAILED ({e})")

        print("\n==============================")

    total = len(lines)
    valid = len(ratios)

    if valid > 0:
        ratios_np = np.array(ratios)

        mean_ar = ratios_np.mean()
        std_ar = ratios_np.std()
        best_ar = ratios_np.max()
        worst_ar = ratios_np.min()

        success_rate = 100.0 * valid / total
        coeff_var = std_ar / mean_ar if mean_ar > 0 else 0.0

        print(f"Total circuits evaluated : {total}")
        print(f"Valid circuits           : {valid}")
        print(f"Failed circuits          : {failed}")
        print(f"Success rate             : {success_rate:.2f}%")
        print()
        print(f"Mean approximation ratio : {mean_ar:.4f}")
        print(f"Std. deviation           : {std_ar:.4f}")
        print(f"Coeff. of variation      : {coeff_var:.4f}")
        print()
        print(f"Best-performing circuit  : {best_ar:.4f}")
        print(f"Worst valid circuit      : {worst_ar:.4f}")
        print(f"No. of perfect circuits  : {perfect}")
    else:
        print("No valid circuits evaluated.")

    print("==============================\n")

if __name__ == "__main__":
    test_all_circuits("generated_circuits_kingston.txt")