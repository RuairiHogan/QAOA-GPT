import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# ============================================================
# USER SETTINGS
# ============================================================

BACKEND_NAME = "ibm_kingston"
N_QUBITS = 7
INITIAL_LAYOUT = [4, 5, 6, 7, 8, 9, 17]
SHOTS = 1024

KINGSTON_FILE = "generated_circuits_kingston_21.txt"
OLD_FILE = "generated_circuits.txt"

# Kingston logical hardware subgraph
KINGSTON_7Q_COUPLING_MAP = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (3, 6),
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ParsedInstance:
    source_file: str
    graph_edges: List[Tuple[int, int, float]]
    layers: List[Tuple[int, float, float]]   # (op_index, gamma, beta)
    format_type: str                         # "kingston" or "old"


# ============================================================
# PARSERS
# ============================================================

def parse_kingston_line(line: str, source_file: str) -> ParsedInstance:
    edge_pattern = re.compile(r"\((\d+),(\d+)\)\s+(-?\d+(?:\.\d+)?)")
    layer_pattern = re.compile(r"<new_layer_\d+>\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

    graph_match = re.search(r"<maxcut_graph>(.*?)<end_of_maxcut_graph>", line)
    circ_match = re.search(r"<circuit>(.*?)<end_of_circuit>", line)

    if graph_match is None or circ_match is None:
        raise ValueError(f"Could not parse Kingston-format line:\n{line}")

    graph_part = graph_match.group(1)
    circuit_part = circ_match.group(1)

    edges = [(int(u), int(v), float(w)) for u, v, w in edge_pattern.findall(graph_part)]
    layers = [(int(op), float(gamma), float(beta)) for op, gamma, beta in layer_pattern.findall(circuit_part)]

    return ParsedInstance(
        source_file=source_file,
        graph_edges=edges,
        layers=layers,
        format_type="kingston",
    )


def parse_old_line(line: str, source_file: str) -> ParsedInstance:
    edge_pattern = re.compile(r"\((\d+),(\d+)\)\s+(-?\d+(?:\.\d+)?)")
    layer_pattern = re.compile(r"<new_layer_\d+>\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

    graph_match = re.search(r"<bos>(.*?)<end_of_graph>", line)
    if graph_match is None:
        raise ValueError(f"Could not parse old-format graph:\n{line}")

    graph_part = graph_match.group(1)
    edges = [(int(u), int(v), float(w)) for u, v, w in edge_pattern.findall(graph_part)]
    layers = [(int(op), float(gamma), float(beta)) for op, gamma, beta in layer_pattern.findall(line)]

    return ParsedInstance(
        source_file=source_file,
        graph_edges=edges,
        layers=layers,
        format_type="old",
    )


def load_instances(filename: str, format_type: str) -> List[ParsedInstance]:
    instances = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                if format_type == "kingston":
                    inst = parse_kingston_line(line, filename)
                elif format_type == "old":
                    inst = parse_old_line(line, filename)
                else:
                    raise ValueError(f"Unknown format_type: {format_type}")
                instances.append(inst)
            except Exception as e:
                print(f"[WARN] Skipping line {line_num} in {filename}: {e}")

    return instances


# ============================================================
# OPERATOR POOLS
# ============================================================

def build_op_names_pdual_hardware(n_qubits: int, coupling_map: List[Tuple[int, int]]) -> List[str]:
    op_names = []

    # singles
    for i in range(n_qubits):
        op_names.append(f"X_{i}")
        op_names.append(f"Y_{i}")
        op_names.append(f"Z_{i}")

    # hardware edges only
    for i, j in sorted({tuple(sorted(e)) for e in coupling_map}):
        for B in ["X", "Y", "Z"]:
            for C in ["X", "Y", "Z"]:
                op_names.append(f"{B}_{i}_{C}_{j}")

    return op_names


def build_op_names_pdual_full(n_qubits: int) -> List[str]:
    op_names = []

    # singles
    for i in range(n_qubits):
        op_names.append(f"X_{i}")
        op_names.append(f"Y_{i}")
        op_names.append(f"Z_{i}")

    # all pairs i < j
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for B in ["X", "Y", "Z"]:
                for C in ["X", "Y", "Z"]:
                    op_names.append(f"{B}_{i}_{C}_{j}")

    return op_names


def pauli_label_from_op_name(op_name: str, n_qubits: int) -> str:
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

    # Qiskit Pauli labels are reversed relative to logical qubit indexing
    return "".join(chars)[::-1]


# ============================================================
# CIRCUIT BUILDING
# ============================================================

def build_cost_operator(edges: List[Tuple[int, int, float]], n_qubits: int) -> SparsePauliOp:
    labels = []
    coeffs = []

    # Use only the ZZ part; identity term is a global phase
    for i, j, w in edges:
        chars = ["I"] * n_qubits
        chars[i] = "Z"
        chars[j] = "Z"
        labels.append("".join(chars)[::-1])
        coeffs.append(-0.5 * w)

    return SparsePauliOp(labels, coeffs=coeffs)


def build_qiskit_circuit(inst: ParsedInstance,
                         op_names_kingston: List[str],
                         op_names_old: List[str],
                         n_qubits: int = 7) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))  # |+>^n

    cost_op = build_cost_operator(inst.graph_edges, n_qubits)

    if inst.format_type == "kingston":
        op_names = op_names_kingston
    elif inst.format_type == "old":
        op_names = op_names_old
    else:
        raise ValueError(f"Unknown instance format: {inst.format_type}")

    for op_index, gamma, beta in inst.layers:
        if op_index < 0 or op_index >= len(op_names):
            raise ValueError(
                f"Operator index {op_index} out of range for {inst.format_type} pool "
                f"(size {len(op_names)})"
            )

        # cost evolution
        qc.append(PauliEvolutionGate(cost_op, time=gamma), range(n_qubits))

        # mixer/operator evolution
        op_name = op_names[op_index]
        pauli_label = pauli_label_from_op_name(op_name, n_qubits)
        mixer_op = SparsePauliOp(pauli_label, coeffs=[1.0])
        qc.append(PauliEvolutionGate(mixer_op, time=beta), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ============================================================
# MAXCUT SCORING
# ============================================================

def cut_value(bitstring: str, edges: List[Tuple[int, int, float]]) -> float:
    bits = [int(b) for b in bitstring[::-1]]  # reverse so index 0 = logical qubit 0
    total = 0.0
    for u, v, w in edges:
        if bits[u] != bits[v]:
            total += w
    return total


def brute_force_maxcut(edges: List[Tuple[int, int, float]], n_qubits: int = 7) -> float:
    best = 0.0
    for z in range(1 << n_qubits):
        bits = [(z >> i) & 1 for i in range(n_qubits)]
        total = 0.0
        for u, v, w in edges:
            if bits[u] != bits[v]:
                total += w
        if total > best:
            best = total
    return best


def expected_cut_from_counts(counts: Dict[str, int], edges: List[Tuple[int, int, float]]) -> float:
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    return sum(cut_value(bitstring, edges) * c for bitstring, c in counts.items()) / total_shots


def best_observed_cut(counts: Dict[str, int], edges: List[Tuple[int, int, float]]) -> Tuple[str, float]:
    best_bitstring = None
    best_value = -1.0
    for bitstring in counts:
        val = cut_value(bitstring, edges)
        if val > best_value:
            best_value = val
            best_bitstring = bitstring
    return best_bitstring, best_value


# ============================================================
# MAIN
# ============================================================

def main():
    kingston_instances = load_instances(KINGSTON_FILE, "kingston")
    old_instances = load_instances(OLD_FILE, "old")

    all_instances = kingston_instances + old_instances

    print(f"Loaded {len(kingston_instances)} circuits from {KINGSTON_FILE}")
    print(f"Loaded {len(old_instances)} circuits from {OLD_FILE}")
    print(f"Total circuits to run: {len(all_instances)}")

    op_names_kingston = build_op_names_pdual_hardware(N_QUBITS, KINGSTON_7Q_COUPLING_MAP)
    op_names_old = build_op_names_pdual_full(N_QUBITS)

    circuits = []
    kept_instances = []

    for idx, inst in enumerate(all_instances):
        try:
            qc = build_qiskit_circuit(inst, op_names_kingston, op_names_old, n_qubits=N_QUBITS)
            circuits.append(qc)
            kept_instances.append(inst)
        except Exception as e:
            print(f"[WARN] Skipping circuit {idx} from {inst.source_file}: {e}")

    print(f"Successfully built {len(circuits)} circuits.")

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    transpiled = transpile(
        circuits,
        backend=backend,
        initial_layout=INITIAL_LAYOUT,
        optimization_level=1,
    )

    print("\nTranspiled summary:")
    for i, tc in enumerate(transpiled[:5]):
        print(f"  Circuit {i}: depth={tc.depth()}, ops={dict(tc.count_ops())}")

    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)

    print(f"\nSubmitted job ID: {job.job_id()}")
    result = job.result()

    print("\nResults:")
    for i, inst in enumerate(kept_instances):
        counts = result[i].data.meas.get_counts()

        expected_cut = expected_cut_from_counts(counts, inst.graph_edges)
        optimal_cut = brute_force_maxcut(inst.graph_edges, n_qubits=N_QUBITS)
        approx_ratio = expected_cut / optimal_cut if optimal_cut > 0 else 0.0

        best_bitstring, best_cut = best_observed_cut(counts, inst.graph_edges)

        print(f"\nCircuit {i} | source={inst.source_file} | format={inst.format_type}")
        print(f"  Expected cut value: {expected_cut:.6f}")
        print(f"  Optimal cut value:  {optimal_cut:.6f}")
        print(f"  Approx ratio:       {approx_ratio:.6f}")
        print(f"  Best observed:      {best_bitstring} -> {best_cut:.6f}")
        print(f"  Top counts:         {sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]}")


if __name__ == "__main__":
    main()