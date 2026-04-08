from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
import os
import threading
import time

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import quantum_info as qi
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize

#########################################################
# USER CONTROLS
#########################################################
NUM_GRAPHS = 2000
NUM_QUBITS = 7
EDGE_PROB_RANGE = (0.3, 0.9)
MAX_DEPTH = 9
GAMMA0_GRID = [0.01, 0.1, 0.5, 1.0]
TARGET_AR = 0.97

DATASET_FILE = "connectivity_scored_dataset.jsonl"

SEED = int(time.time())
BACKEND_NAME = "ibm_kingston"
INITIAL_LAYOUT = [4, 5, 6, 7, 8, 9, 17]

LAMBDA_2Q = 0.003
LAMBDA_DEPTH = 0.0005
MIN_AR_TO_KEEP = 0.90
TOP_K = 1
OP_POOL_MODE = "PDUAL"

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
    x_ops = []
    y_ops = []
    z_ops = []
    for i in range(n):
        x_ops.append(qi.Pauli(("I" * i + "X" + "I" * (n - i - 1))[::-1]).to_matrix())
        y_ops.append(qi.Pauli(("I" * i + "Y" + "I" * (n - i - 1))[::-1]).to_matrix())
        z_ops.append(qi.Pauli(("I" * i + "Z" + "I" * (n - i - 1))[::-1]).to_matrix())
    identity = np.eye(2**n, dtype=complex)
    return identity, x_ops, y_ops, z_ops


def precompute_edge_ZZ(graph):
    n = graph.number_of_nodes()
    edge_terms = []
    for (i, j) in graph.edges:
        w = float(graph[i][j]["weight"])
        s = ["I"] * n
        s[i] = "Z"
        s[j] = "Z"
        zz = qi.Pauli("".join(s)[::-1]).to_matrix()
        edge_terms.append((w, zz))
    return edge_terms


#########################################################
# Fast application of exp(-i theta P)
#########################################################


def apply_pauli_exp_to_state(psi, pauli_matrix, theta, sign=-1):
    c = np.cos(theta)
    s = np.sin(theta)
    if sign == -1:
        return c * psi - 1j * s * (pauli_matrix @ psi)
    return c * psi + 1j * s * (pauli_matrix @ psi)


#########################################################
# Cost application
#########################################################


def apply_cost_to_state(psi, edge_terms, gamma):
    for (w, zz) in edge_terms:
        theta = gamma * w / 2.0
        psi = apply_pauli_exp_to_state(psi, zz, theta, sign=+1)
    return psi


#########################################################
# Full operator pool
#########################################################


def build_operator_list_and_mats(n, identity, x_ops, y_ops, z_ops, coupling_map=None):
    del identity
    del coupling_map

    op_names = []
    op_mats = []

    if OP_POOL_MODE.upper() == "PQAOA":
        op_names.append("HB")
        op_mats.append(None)
    elif OP_POOL_MODE.upper() == "PDUAL":
        for i in range(n):
            op_names.append(f"X_{i}")
            op_mats.append(x_ops[i])
            op_names.append(f"Y_{i}")
            op_mats.append(y_ops[i])
            op_names.append(f"Z_{i}")
            op_mats.append(z_ops[i])

        paulis = {"X": x_ops, "Y": y_ops, "Z": z_ops}
        full_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        for (i, j) in full_pairs:
            for first in ["X", "Y", "Z"]:
                for second in ["X", "Y", "Z"]:
                    op_names.append(f"{first}_{i}_{second}_{j}")
                    op_mats.append(paulis[first][i] @ paulis[second][j])
    else:
        raise ValueError("OP_POOL_MODE must be 'PQAOA' or 'PDUAL'")

    return op_names, op_mats


#########################################################
# ADAPT gradient
#########################################################


def adapt_gradient(phi, hamiltonian, operator):
    comm = hamiltonian @ operator - operator @ hamiltonian
    return np.imag(np.vdot(phi, comm @ phi))


#########################################################
# Build/apply the circuit on a statevector
#########################################################


def apply_mixer_to_state(psi, op_index, beta, op_names, op_mats, x_ops):
    name = op_names[op_index]
    if name == "HB":
        for x_op in x_ops:
            psi = apply_pauli_exp_to_state(psi, x_op, beta, sign=-1)
        return psi

    pauli_matrix = op_mats[op_index]
    return apply_pauli_exp_to_state(psi, pauli_matrix, beta, sign=-1)


def build_state(n, edge_terms, op_names, op_mats, x_ops, op_indices, betas, gammas):
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for k in range(len(op_indices)):
        psi = apply_cost_to_state(psi, edge_terms, gammas[k])
        psi = apply_mixer_to_state(psi, op_indices[k], betas[k], op_names, op_mats, x_ops)
    return psi


#########################################################
# Approximation ratio
#########################################################


def approximation_ratio(psi, hamiltonian, opt_val):
    val = np.real(np.vdot(psi, hamiltonian @ psi))
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


def run_adapt_qaoa_once(graph, hamiltonian, opt_val, edge_terms, op_names, op_mats, identity, x_ops, gamma0, max_depth):
    del graph
    del identity

    n = NUM_QUBITS
    op_indices = []
    betas = []
    gammas = []

    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

    for _ in range(max_depth):
        phi = apply_cost_to_state(psi, edge_terms, gamma0)

        grads = []
        for j, name in enumerate(op_names):
            if name == "HB":
                operator = np.zeros((2**n, 2**n), dtype=complex)
                for x_op in x_ops:
                    operator += x_op
            else:
                operator = op_mats[j]
            grads.append(abs(adapt_gradient(phi, hamiltonian, operator)))

        best_j = int(np.argmax(grads))
        op_indices.append(best_j)

        def objective_neg(x):
            beta_new, gamma_new = x
            psi_tmp = build_state(
                n,
                edge_terms,
                op_names,
                op_mats,
                x_ops,
                op_indices,
                betas + [beta_new],
                gammas + [gamma_new],
            )
            return -np.real(np.vdot(psi_tmp, hamiltonian @ psi_tmp))

        res = minimize(objective_neg, x0=[0.1, 0.1], method="Nelder-Mead")

        betas.append(float(res.x[0]))
        gammas.append(float(res.x[1]))

        psi = build_state(n, edge_terms, op_names, op_mats, x_ops, op_indices, betas, gammas)
        ar = approximation_ratio(psi, hamiltonian, opt_val)

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
    if ar >= 0.94:
        return "good"
    if ar >= 0.90:
        return "acceptable"
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
        pauli, i = parts
        chars[int(i)] = pauli
    elif len(parts) == 4:
        pauli_1, i, pauli_2, j = parts
        chars[int(i)] = pauli_1
        chars[int(j)] = pauli_2
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
        graph,
        op_indices,
        betas,
        gammas,
        op_names,
        graph.number_of_nodes(),
    )

    tqc = transpile(
        qc,
        backend=backend,
        initial_layout=INITIAL_LAYOUT,
        optimization_level=1,
        seed_transpiler=1234,
    )

    return tqc.depth(), count_two_qubit_gates(tqc)


def hardware_score(ar, two_qubit_count, depth):
    return ar - LAMBDA_2Q * two_qubit_count - LAMBDA_DEPTH * depth


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
    seed,
    depth,
    two_qubit_count,
    score,
):
    entry = {
        "num_qubits": graph.number_of_nodes(),
        "edge_prob_model": "erdos_renyi",
        "op_pool": op_pool_mode,
        "seed": seed,
        "gamma0": gamma0,
        "approx_ratio": round(float(ar), 4),
        "tier": tier,
        "hardware_score": round(float(score), 6),
        "transpiled_depth": int(depth),
        "transpiled_two_qubit_count": int(two_qubit_count),
        "lambda_2q": LAMBDA_2Q,
        "lambda_depth": LAMBDA_DEPTH,
        "hardware_name": "ibm_kingston",
        "tokens": tokens,
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def format_elapsed_time(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


#########################################################
# Multi-process helpers
#########################################################


def split_graph_targets(total_graphs, num_workers):
    base = total_graphs // num_workers
    remainder = total_graphs % num_workers
    return [
        base + (1 if worker_id < remainder else 0)
        for worker_id in range(num_workers)
    ]


def worker_output_filename(base_filename, num_workers, worker_id):
    root, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".jsonl"
    return f"{root}.workers_{num_workers}.part_{worker_id}{ext}"


#########################################################
# Worker entrypoint
#########################################################


def run_generation(num_graphs, dataset_file, seed, worker_id=0, stop_event=None):
    rng = np.random.default_rng(seed)

    if os.path.exists(dataset_file):
        os.remove(dataset_file)
    open(dataset_file, "w", encoding="utf-8").close()

    written = 0
    attempts = 0

    print(f"[worker {worker_id}] Connecting to backend: {BACKEND_NAME}")
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    while written < num_graphs and attempts < num_graphs * 10:
        if stop_event is not None and stop_event.is_set():
            print(f"[worker {worker_id}] Stop requested. Exiting cleanly.")
            break

        attempts += 1

        s = float(rng.uniform(*EDGE_PROB_RANGE))
        graph_seed = int(rng.integers(0, 2**32 - 1))
        graph = nx.erdos_renyi_graph(NUM_QUBITS, s, seed=graph_seed)
        if not nx.is_connected(graph):
            continue

        for (u, v) in graph.edges:
            w = float(rng.uniform(0.0, 1.0))
            if w == 0.0:
                w = 1e-6
            graph[u][v]["weight"] = w

        n = graph.number_of_nodes()
        identity, x_ops, y_ops, z_ops = build_single_qubit_paulis(n)
        edge_terms = precompute_edge_ZZ(graph)

        opt_val = maxcut_opt_bruteforce(graph)
        hamiltonian = cost_hamiltonian_Hc(graph)
        op_names, op_mats = build_operator_list_and_mats(
            n,
            identity,
            x_ops,
            y_ops,
            z_ops,
            coupling_map=None,
        )

        candidates = []

        for gamma0 in GAMMA0_GRID:
            if stop_event is not None and stop_event.is_set():
                print(f"[worker {worker_id}] Stop requested during candidate search.")
                break

            op_indices, betas, gammas, ar = run_adapt_qaoa_once(
                graph,
                hamiltonian,
                opt_val,
                edge_terms,
                op_names,
                op_mats,
                identity,
                x_ops,
                gamma0,
                MAX_DEPTH,
            )

            if ar < MIN_AR_TO_KEEP:
                print(
                    f"[worker {worker_id}] Rejected candidate | "
                    f"n={NUM_QUBITS}, s={s:.2f}, gamma0={gamma0}, "
                    f"AR={ar:.4f} < {MIN_AR_TO_KEEP:.2f}"
                )
                continue

            tokens = tokenize_graph_and_circuit(graph, op_indices, betas, gammas)
            if tokens is None:
                continue

            try:
                depth, two_qubit_count = compilation_metrics(
                    graph,
                    op_indices,
                    betas,
                    gammas,
                    op_names,
                    backend,
                )
            except Exception as exc:
                print(f"[worker {worker_id}] [WARN] Compilation failed for gamma0={gamma0}: {exc}")
                continue

            score = hardware_score(ar, two_qubit_count, depth)
            tier = ar_tier(ar)

            print(
                f"[worker {worker_id}] Candidate | s={s:.2f}, gamma0={gamma0}, "
                f"AR={ar:.4f}, 2Q={two_qubit_count}, depth={depth}, score={score:.6f}"
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

        if stop_event is not None and stop_event.is_set():
            break

        if not candidates:
            continue

        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
        top_candidates = candidates[:TOP_K]

        for candidate in top_candidates:
            write_dataset_entry(
                dataset_file,
                graph,
                candidate["tokens"],
                candidate["ar"],
                candidate["tier"],
                OP_POOL_MODE,
                candidate["gamma0"],
                seed,
                candidate["depth"],
                candidate["two_qubit_count"],
                candidate["score"],
            )

            written += 1
            print(
                f"[worker {worker_id}] Wrote circuit {written}/{num_graphs} | "
                f"AR={candidate['ar']:.4f}, "
                f"2Q={candidate['two_qubit_count']}, "
                f"depth={candidate['depth']}, "
                f"score={candidate['score']:.6f}"
            )

            if written >= num_graphs:
                break

    print(f"[worker {worker_id}] Done. Dataset saved to: {dataset_file}")
    return {
        "worker_id": worker_id,
        "written": written,
        "dataset_file": dataset_file,
        "seed": seed,
    }


def start_quit_listener(stop_event):
    def listen():
        while not stop_event.is_set():
            try:
                user_input = input().strip().lower()
            except EOFError:
                return

            if user_input == "q":
                print("\nQuit requested. Waiting for workers to stop cleanly...")
                stop_event.set()
                return

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    return listener


#########################################################
# CLI
#########################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADAPT-QAOA data generation with one JSONL output per worker."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes to launch.",
    )
    parser.add_argument(
        "--graphs",
        type=int,
        default=NUM_GRAPHS,
        help="Total number of dataset entries to generate across all workers.",
    )
    parser.add_argument(
        "--dataset-file",
        default=DATASET_FILE,
        help="Base output filename used to derive per-worker JSONL filenames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base RNG seed. Each worker uses base_seed + worker_id.",
    )
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.workers == 1:
        stop_event = multiprocessing.Manager().Event()
        print("Type 'q' then press Enter to stop cleanly.")
        start_quit_listener(stop_event)
        run_generation(args.graphs, args.dataset_file, args.seed, worker_id=0, stop_event=stop_event)
        elapsed = time.time() - start_time
        print(f"Total runtime: {format_elapsed_time(elapsed)}")
        return

    graph_targets = split_graph_targets(args.graphs, args.workers)
    jobs = []

    for worker_id, num_graphs in enumerate(graph_targets):
        if num_graphs <= 0:
            continue
        jobs.append({
            "worker_id": worker_id,
            "num_graphs": num_graphs,
            "dataset_file": worker_output_filename(args.dataset_file, args.workers, worker_id),
            "seed": args.seed + worker_id,
        })

    print(f"Launching {len(jobs)} worker process(es)")
    print("Type 'q' then press Enter to stop cleanly.")
    for job in jobs:
        print(
            f"worker {job['worker_id']}: "
            f"{job['num_graphs']} graphs -> {job['dataset_file']} "
            f"(seed={job['seed']})"
        )

    with multiprocessing.Manager() as manager:
        stop_event = manager.Event()
        start_quit_listener(stop_event)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_generation,
                    job["num_graphs"],
                    job["dataset_file"],
                    job["seed"],
                    job["worker_id"],
                    stop_event,
                )
                for job in jobs
            ]
            results = [future.result() for future in futures]

    total_written = sum(result["written"] for result in results)

    print("\nPer-worker outputs:")
    for result in sorted(results, key=lambda item: item["worker_id"]):
        print(
            f"worker {result['worker_id']}: "
            f"{result['written']} entries -> {result['dataset_file']}"
        )

    print(f"\nTotal circuits written: {total_written}/{args.graphs}")
    print("No merge was performed. Each worker wrote to its own JSONL file.")
    elapsed = time.time() - start_time
    print(f"Total runtime: {format_elapsed_time(elapsed)}")


if __name__ == "__main__":
    main()
