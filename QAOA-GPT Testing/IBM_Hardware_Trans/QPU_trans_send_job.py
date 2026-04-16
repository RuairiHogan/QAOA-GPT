import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

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

BASE_DIR = Path(__file__).resolve().parent
KINGSTON_FILE = BASE_DIR / "generated_circuits_trans_optimal_50.txt"
OLD_FILE = BASE_DIR / "generated_circuits_og_optimal_50.txt"

# how many circuits per submitted job
CHUNK_SIZE = 50

# metadata file to save submitted job info
SUBMISSION_METADATA_FILE = BASE_DIR / "submitted_jobs_trans_optimal_50.json"

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
    source_line: int
    graph_edges: List[Tuple[int, int, float]]
    layers: List[Tuple[int, float, float]]   # (op_index, gamma, beta)
    format_type: str                         # "kingston" or "old"


# ============================================================
# PARSERS
# ============================================================

def parse_kingston_line(line: str, source_file: str, line_num: int) -> ParsedInstance:
    edge_pattern = re.compile(r"\((\d+),(\d+)\)\s+(-?\d+(?:\.\d+)?)")
    layer_pattern = re.compile(r"<new_layer_\d+>\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

    tokens = line.split()
    if len(tokens) < 2:
        raise ValueError("Trans-format line is too short to strip the first two tokens")
    line = " ".join(tokens[2:])

    graph_match = re.search(r"<maxcut_graph>(.*?)<end_of_maxcut_graph>", line)
    circ_match = re.search(r"<circuit>(.*?)<end_of_circuit>", line)

    if graph_match is None or circ_match is None:
        raise ValueError(f"Could not parse trans-format line:\n{line}")

    graph_part = graph_match.group(1)
    circuit_part = circ_match.group(1)

    edges = [(int(u), int(v), float(w)) for u, v, w in edge_pattern.findall(graph_part)]
    layers = [(int(op), float(gamma), float(beta)) for op, gamma, beta in layer_pattern.findall(circuit_part)]

    return ParsedInstance(
        source_file=source_file,
        source_line=line_num,
        graph_edges=edges,
        layers=layers,
        format_type="trans",
    )


def parse_old_line(line: str, source_file: str, line_num: int) -> ParsedInstance:
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
        source_line=line_num,
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
                if format_type == "trans":
                    inst = parse_kingston_line(line, filename, line_num)
                elif format_type == "old":
                    inst = parse_old_line(line, filename, line_num)
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

    for i in range(n_qubits):
        op_names.append(f"X_{i}")
        op_names.append(f"Y_{i}")
        op_names.append(f"Z_{i}")

    for i, j in sorted({tuple(sorted(e)) for e in coupling_map}):
        for B in ["X", "Y", "Z"]:
            for C in ["X", "Y", "Z"]:
                op_names.append(f"{B}_{i}_{C}_{j}")

    return op_names


def build_op_names_pdual_full(n_qubits: int) -> List[str]:
    op_names = []

    for i in range(n_qubits):
        op_names.append(f"X_{i}")
        op_names.append(f"Y_{i}")
        op_names.append(f"Z_{i}")

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

    return "".join(chars)[::-1]


# ============================================================
# CIRCUIT BUILDING
# ============================================================

def build_cost_operator(edges: List[Tuple[int, int, float]], n_qubits: int) -> SparsePauliOp:
    labels = []
    coeffs = []

    for i, j, w in edges:
        chars = ["I"] * n_qubits
        chars[i] = "Z"
        chars[j] = "Z"
        labels.append("".join(chars)[::-1])
        coeffs.append(-0.5 * w)

    return SparsePauliOp(labels, coeffs=coeffs)


def build_qiskit_circuit(
    inst: ParsedInstance,
    op_names_trans: List[str],
    op_names_old: List[str],
    n_qubits: int = 7
) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    cost_op = build_cost_operator(inst.graph_edges, n_qubits)

    if inst.format_type == "trans":
        op_names = op_names_trans
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

        qc.append(PauliEvolutionGate(cost_op, time=gamma), range(n_qubits))

        op_name = op_names[op_index]
        pauli_label = pauli_label_from_op_name(op_name, n_qubits)
        mixer_op = SparsePauliOp(pauli_label, coeffs=[1.0])
        qc.append(PauliEvolutionGate(mixer_op, time=beta), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ============================================================
# HELPERS
# ============================================================

def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield i, items[i:i + chunk_size]


# ============================================================
# MAIN
# ============================================================

def main():
    trans_instances = load_instances(KINGSTON_FILE, "trans")
    old_instances = load_instances(OLD_FILE, "old")
    all_instances = trans_instances + old_instances

    print(f"Loaded {len(trans_instances)} circuits from {KINGSTON_FILE}")
    print(f"Loaded {len(old_instances)} circuits from {OLD_FILE}")
    print(f"Total circuits parsed: {len(all_instances)}")

    op_names_trans = build_op_names_pdual_full(N_QUBITS)
    op_names_old = build_op_names_pdual_full(N_QUBITS)

    built_circuits = []
    kept_instances = []

    for idx, inst in enumerate(all_instances):
        try:
            qc = build_qiskit_circuit(inst, op_names_trans, op_names_old, n_qubits=N_QUBITS)
            built_circuits.append(qc)
            kept_instances.append(inst)
        except Exception as e:
            print(f"[WARN] Skipping circuit {idx} from {inst.source_file}:{inst.source_line} -> {e}")

    print(f"Successfully built {len(built_circuits)} circuits.")

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    submissions = []

    for chunk_start, circuit_chunk in chunk_list(built_circuits, CHUNK_SIZE):
        inst_chunk = kept_instances[chunk_start:chunk_start + len(circuit_chunk)]

        transpiled = transpile(
            circuit_chunk,
            backend=backend,
            initial_layout=INITIAL_LAYOUT,
            optimization_level=1,
        )

        print(f"\nSubmitting circuits {chunk_start} to {chunk_start + len(circuit_chunk) - 1}")
        for local_i, tc in enumerate(transpiled[:3]):
            print(
                f"  Preview circuit {chunk_start + local_i}: "
                f"depth={tc.depth()}, ops={dict(tc.count_ops())}"
            )

        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=SHOTS)
        job_id = job.job_id()

        print(f"  Submitted job ID: {job_id}")

        submission_entry = {
            "job_id": job_id,
            "backend": BACKEND_NAME,
            "shots": SHOTS,
            "initial_layout": INITIAL_LAYOUT,
            "chunk_start_index": chunk_start,
            "chunk_size": len(circuit_chunk),
            "circuits": [
                {
                    "global_index": chunk_start + local_idx,
                    "source_file": str(inst.source_file),
                    "source_line": inst.source_line,
                    "format_type": inst.format_type,
                    "graph_edges": inst.graph_edges,
                    "layers": inst.layers,
                }
                for local_idx, inst in enumerate(inst_chunk)
            ],
        }
        submissions.append(submission_entry)

    with open(SUBMISSION_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2)

    print("\nDone.")
    print("Submitted job IDs:")
    for sub in submissions:
        print(f"  {sub['job_id']}")

    print(f"\nSaved metadata to: {SUBMISSION_METADATA_FILE}")
    print("You can now retrieve these jobs later in a separate script and compute correlations there.")


if __name__ == "__main__":
    main()
