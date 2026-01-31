import numpy as np
import networkx as nx
from qiskit import quantum_info as qi

########################################
# Utilities (MATCH TRAINING)
########################################

def maxcut_opt_bruteforce(graph):
    n = graph.number_of_nodes()
    best = 0.0
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        cut = 0.0
        for (u, v) in graph.edges:
            if bits[u] != bits[v]:
                cut += graph[u][v]["weight"]
        best = max(best, cut)
    return best


def cost_hamiltonian_Hc(graph):
    """ EXACT same Hc used in training """
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
    c = np.cos(theta)
    s = np.sin(theta)
    return c * psi + (1j * sign * s) * (P @ psi)


########################################
# OPERATOR POOL (IDENTICAL TO TRAINING)
########################################

def build_operator_pool(n):
    X, Y, Z = [], [], []
    for i in range(n):
        X.append(qi.Pauli(("I"*i + "X" + "I"*(n-i-1))[::-1]).to_matrix())
        Y.append(qi.Pauli(("I"*i + "Y" + "I"*(n-i-1))[::-1]).to_matrix())
        Z.append(qi.Pauli(("I"*i + "Z" + "I"*(n-i-1))[::-1]).to_matrix())

    op_mats = []

    # singles (ORDER MATTERS)
    for i in range(n):
        op_mats.append(X[i])
        op_mats.append(Y[i])
        op_mats.append(Z[i])

    # two-qubit Pauli strings (ORDER MATTERS)
    paulis = {"X": X, "Y": Y, "Z": Z}
    for i in range(n):
        for j in range(i + 1, n):
            for B in ["X", "Y", "Z"]:
                for C in ["X", "Y", "Z"]:
                    op_mats.append(paulis[B][i] @ paulis[C][j])

    return op_mats



########################################
# TOKEN PARSING
########################################

def parse_tokens(tokens):
    i = 0
    assert tokens[i] == "<bos>"
    i += 1

    edges = []
    while tokens[i] != "<end_of_graph>":
        u, v = tokens[i].strip("()").split(",")
        w = float(tokens[i + 1])
        edges.append((int(u), int(v), w))
        i += 2
    i += 1

    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    ops, gammas, betas = [], [], []
    while i < len(tokens):
        if tokens[i].startswith("<new_layer_"):
            ops.append(int(tokens[i + 1]))
            gammas.append(float(tokens[i + 2]))
            betas.append(float(tokens[i + 3]))
            i += 4
        else:
            break

    return G, ops, gammas, betas


########################################
# CIRCUIT EVALUATION
########################################

def evaluate_circuit(G, ops, gammas, betas, op_mats):
    n = G.number_of_nodes()
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    # Precompute ZZ edges
    edge_terms = []
    for (i, j) in G.edges:
        w = G[i][j]["weight"]
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        ZZ = qi.Pauli("".join(s)[::-1]).to_matrix()
        edge_terms.append((w, ZZ))

    # Apply circuit
    for k in range(len(ops)):
        for w, ZZ in edge_terms:
            psi = apply_pauli_exp_to_state(psi, ZZ, gammas[k] * w / 2, sign=+1)

        idx = ops[k]
        if idx >= len(op_mats):
            raise ValueError(f"Operator index {idx} out of range (pool size {len(op_mats)})")

        psi = apply_pauli_exp_to_state(psi, op_mats[idx], betas[k], sign=-1)

    Hc = cost_hamiltonian_Hc(G)
    return np.real(np.vdot(psi, Hc @ psi))


########################################
# MAIN TEST (ALL CIRCUITS)
########################################

def test_all_circuits(test_file):
    with open(test_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    ratios = []
    failed = 0

    perfect = 0
    print(f"Testing {len(lines)} circuits...\n")

    for idx, line in enumerate(lines):
        try:
            tokens = line.split()
            G, ops, gammas, betas = parse_tokens(tokens)

            op_mats = build_operator_pool(G.number_of_nodes())

            approx = evaluate_circuit(G, ops, gammas, betas, op_mats)
            opt = maxcut_opt_bruteforce(G)

            ar = approx / opt if opt > 0 else 0.0
            ratios.append(ar)

            if np.isclose(ar, 1.0, atol=1e-6):   # <-- ADDED
                perfect += 1

            print(f"Circuit {idx+1:3d}: AR = {ar:.4f}")



        except Exception as e:
            failed += 1
            print(f"Circuit {idx+1:3d}: FAILED ({e})")

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
        print(f"No. of perfect circuits  : ", perfect)
    else:
        print("No valid circuits evaluated.")

    print("==============================\n")



if __name__ == "__main__":
    test_all_circuits("generated_circuits.txt")

