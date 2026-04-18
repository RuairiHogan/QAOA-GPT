import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import quantum_info as qi
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService


BACKEND_NAME = "ibm_kingston"
N_QUBITS = 7
INITIAL_LAYOUT = [4, 5, 6, 7, 8, 9, 17]

LAMBDA_2Q = 0.003
LAMBDA_DEPTH = 0.0005

BASE_DIR = Path(__file__).resolve().parent
TRANS_FILE = BASE_DIR / "generated_circuits_trans_20.txt"
OLD_FILE = BASE_DIR / "generated_circuits_og_20.txt"
OUTPUT_JSON = BASE_DIR / "transpile_score_test_20_summary.json"

OUTPUT_2Q_BAR = BASE_DIR / "transpile_test_20_mean_two_qubit_gates.png"
OUTPUT_DEPTH_BAR = BASE_DIR / "transpile_test_20_mean_depth.png"
OUTPUT_SCORE_BAR = BASE_DIR / "transpile_test_20_mean_score.png"
OUTPUT_TRANS_SCORE_HIST = BASE_DIR / "transpile_test_20_trans_score_histogram.png"
OUTPUT_OLD_SCORE_HIST = BASE_DIR / "transpile_test_20_unconstrained_score_histogram.png"

plt.rcParams["font.family"] = "Times New Roman"

BAR_LABEL_FONT_SIZE = 22
BAR_LEGEND_FONT_SIZE = 20
BAR_TICK_FONT_SIZE = 18
BAR_COLORS = ["#00B8D9", "#d4b595"]

HIST_LABEL_FONT_SIZE = 22
HIST_LEGEND_FONT_SIZE = 22
HIST_TICK_FONT_SIZE = 18
HIST_TRANS_COLOR = "#00B8D9"
HIST_OLD_COLOR = "#d4b595"
MEAN_COLOR = "#6C63FF"
ACCENT_COLOR = "#0D1B52"


@dataclass
class ParsedInstance:
    source_file: str
    source_line: int
    graph_edges: List[Tuple[int, int, float]]
    layers: List[Tuple[int, float, float]]
    format_type: str


def parse_trans_line(line: str, source_file: str, line_num: int) -> ParsedInstance:
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

    edges = [(int(u), int(v), float(w)) for u, v, w in edge_pattern.findall(graph_match.group(1))]
    layers = [(int(op), float(gamma), float(beta)) for op, gamma, beta in layer_pattern.findall(circ_match.group(1))]

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

    edges = [(int(u), int(v), float(w)) for u, v, w in edge_pattern.findall(graph_match.group(1))]
    layers = [(int(op), float(gamma), float(beta)) for op, gamma, beta in layer_pattern.findall(line)]

    return ParsedInstance(
        source_file=source_file,
        source_line=line_num,
        graph_edges=edges,
        layers=layers,
        format_type="old",
    )


def load_instances(path: Path, format_type: str) -> List[ParsedInstance]:
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if format_type == "trans":
                instances.append(parse_trans_line(line, str(path), line_num))
            elif format_type == "old":
                instances.append(parse_old_line(line, str(path), line_num))
            else:
                raise ValueError(f"Unsupported format_type: {format_type}")
    return instances


def maxcut_opt_bruteforce_from_edges(edges: List[Tuple[int, int, float]], n_qubits: int) -> float:
    best = 0.0
    for x in range(1 << n_qubits):
        bits = [(x >> i) & 1 for i in range(n_qubits)]
        cut = 0.0
        for u, v, w in edges:
            if bits[u] != bits[v]:
                cut += float(w)
        best = max(best, cut)
    return best


def build_cost_operator(edges: List[Tuple[int, int, float]], n_qubits: int) -> SparsePauliOp:
    labels = []
    coeffs = []
    for i, j, w in edges:
        chars = ["I"] * n_qubits
        chars[i] = "Z"
        chars[j] = "Z"
        labels.append("".join(chars)[::-1])
        coeffs.append(-0.5 * float(w))
    return SparsePauliOp(labels, coeffs=coeffs)


def build_op_names_pdual_full(n_qubits: int) -> List[str]:
    op_names = []
    for i in range(n_qubits):
        op_names.append(f"X_{i}")
        op_names.append(f"Y_{i}")
        op_names.append(f"Z_{i}")

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for b in ["X", "Y", "Z"]:
                for c in ["X", "Y", "Z"]:
                    op_names.append(f"{b}_{i}_{c}_{j}")

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


def build_qiskit_circuit(inst: ParsedInstance, op_names: List[str], n_qubits: int, measure: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits if measure else 0)
    qc.h(range(n_qubits))

    cost_op = build_cost_operator(inst.graph_edges, n_qubits)

    for op_index, gamma, beta in inst.layers:
        if op_index < 0 or op_index >= len(op_names):
            raise ValueError(f"Operator index {op_index} out of range (pool size {len(op_names)})")

        qc.append(PauliEvolutionGate(cost_op, time=gamma), range(n_qubits))

        op_name = op_names[op_index]
        pauli_label = pauli_label_from_op_name(op_name, n_qubits)
        mixer_op = SparsePauliOp(pauli_label, coeffs=[1.0])
        qc.append(PauliEvolutionGate(mixer_op, time=beta), range(n_qubits))

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc


def compute_approx_ratio(inst: ParsedInstance, op_names: List[str], n_qubits: int) -> float:
    qc = build_qiskit_circuit(inst, op_names, n_qubits=n_qubits, measure=False)
    state = qi.Statevector.from_instruction(qc)
    cost_op = build_cost_operator(inst.graph_edges, n_qubits)
    expected_cut = float(np.real(state.expectation_value(cost_op))) + 0.5 * sum(w for _, _, w in inst.graph_edges)
    optimal_cut = maxcut_opt_bruteforce_from_edges(inst.graph_edges, n_qubits)
    return expected_cut / optimal_cut if optimal_cut > 0 else 0.0


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    return sum(
        1
        for instruction in circuit.data
        if instruction.operation.name != "barrier" and len(instruction.qubits) == 2
    )


def compute_score(approx_ratio: float, two_qubit_gates: int, depth: int) -> float:
    return approx_ratio - LAMBDA_2Q * two_qubit_gates - LAMBDA_DEPTH * depth


def mean_std(values: List[float]) -> Tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def make_metric_bar_plot(trans_values, old_values, ylabel, output_path):
    labels = ["Transpilation-Aware GPT", "Unconstrained GPT"]
    means = [float(np.mean(trans_values)), float(np.mean(old_values))]
    stds = [float(np.std(trans_values)), float(np.std(old_values))]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=BAR_COLORS)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=BAR_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=BAR_TICK_FONT_SIZE)
    ax.legend(bars, labels, frameon=True, fontsize=BAR_LEGEND_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ymin = min(means[i] - stds[i] for i in range(len(means)))
    ymax = max(means[i] + stds[i] for i in range(len(means)))
    pad = 0.08 * max(1.0, ymax - ymin)
    if "score" in ylabel.lower():
        ax.set_ylim(ymin - pad, ymax + 2.0 * pad)
    else:
        ax.set_ylim(max(0.0, ymin - pad), ymax + 2.0 * pad)

    text_offset = 0.06 * max(1.0, ymax - ymin)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + text_offset,
            f"{means[i]:.3f} +/- {stds[i]:.3f}",
            ha="center",
            va="center",
            fontsize=BAR_TICK_FONT_SIZE,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_score_histogram(scores, output_path, distribution_label, color):
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.hist(
        scores,
        bins=10,
        color=color,
        edgecolor=ACCENT_COLOR,
        alpha=0.82,
        label=distribution_label,
    )
    ax.axvline(
        mean_score,
        linestyle="--",
        linewidth=1.8,
        color=MEAN_COLOR,
        label=f"Mean score = {mean_score:.4f}",
    )
    ax.axvline(
        median_score,
        linestyle=":",
        linewidth=1.8,
        color=ACCENT_COLOR,
        label=f"Median score = {median_score:.4f}",
    )

    ax.set_xlabel("Transpilation score", fontsize=HIST_LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel("Number of circuits", fontsize=HIST_LABEL_FONT_SIZE, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=HIST_TICK_FONT_SIZE)
    ax.tick_params(axis="y", pad=8)
    ax.legend(frameon=True, fontsize=HIST_LEGEND_FONT_SIZE)

    fig.subplots_adjust(left=0.12, bottom=0.14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_instances(instances: List[ParsedInstance], backend, op_names: List[str]) -> List[dict]:
    results = []
    for idx, inst in enumerate(instances, start=1):
        approx_ratio = compute_approx_ratio(inst, op_names, N_QUBITS)
        qc = build_qiskit_circuit(inst, op_names, n_qubits=N_QUBITS, measure=True)
        transpiled = transpile(
            qc,
            backend=backend,
            initial_layout=INITIAL_LAYOUT,
            optimization_level=1,
        )
        two_qubit_gates = count_two_qubit_gates(transpiled)
        depth = int(transpiled.depth())
        score = compute_score(approx_ratio, two_qubit_gates, depth)

        results.append(
            {
                "circuit_index": idx,
                "source_file": inst.source_file,
                "source_line": inst.source_line,
                "format_type": inst.format_type,
                "approx_ratio": float(approx_ratio),
                "two_qubit_gates": two_qubit_gates,
                "depth": depth,
                "transpilation_score": float(score),
                "num_layers": len(inst.layers),
            }
        )

        print(
            f"{inst.format_type:>5} circuit {idx:2d}: "
            f"AR={approx_ratio:.4f}, 2Q={two_qubit_gates}, depth={depth}, score={score:.4f}"
        )

    return results


def summarize_results(results: List[dict], label: str) -> dict:
    ars = [r["approx_ratio"] for r in results]
    two_q = [r["two_qubit_gates"] for r in results]
    depths = [r["depth"] for r in results]
    scores = [r["transpilation_score"] for r in results]

    return {
        "label": label,
        "num_circuits": len(results),
        "mean_approx_ratio": float(np.mean(ars)),
        "std_approx_ratio": float(np.std(ars)),
        "mean_two_qubit_gates": float(np.mean(two_q)),
        "std_two_qubit_gates": float(np.std(two_q)),
        "mean_depth": float(np.mean(depths)),
        "std_depth": float(np.std(depths)),
        "mean_transpilation_score": float(np.mean(scores)),
        "std_transpilation_score": float(np.std(scores)),
        "min_transpilation_score": float(np.min(scores)),
        "max_transpilation_score": float(np.max(scores)),
    }


def main():
    trans_instances = load_instances(TRANS_FILE, "trans")
    old_instances = load_instances(OLD_FILE, "old")
    op_names = build_op_names_pdual_full(N_QUBITS)

    print(f"Loaded {len(trans_instances)} transpilation-aware circuits from {TRANS_FILE.name}")
    print(f"Loaded {len(old_instances)} unconstrained circuits from {OLD_FILE.name}")

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    trans_results = process_instances(trans_instances, backend, op_names)
    old_results = process_instances(old_instances, backend, op_names)

    trans_two_q = [r["two_qubit_gates"] for r in trans_results]
    old_two_q = [r["two_qubit_gates"] for r in old_results]
    trans_depth = [r["depth"] for r in trans_results]
    old_depth = [r["depth"] for r in old_results]
    trans_scores = [r["transpilation_score"] for r in trans_results]
    old_scores = [r["transpilation_score"] for r in old_results]

    make_metric_bar_plot(trans_two_q, old_two_q, "Mean Two-Qubit Gate Count", OUTPUT_2Q_BAR)
    make_metric_bar_plot(trans_depth, old_depth, "Mean Circuit Depth", OUTPUT_DEPTH_BAR)
    make_metric_bar_plot(trans_scores, old_scores, "Mean Transpilation Score", OUTPUT_SCORE_BAR)
    make_score_histogram(
        trans_scores,
        OUTPUT_TRANS_SCORE_HIST,
        "Transpilation-aware GPT score distribution",
        HIST_TRANS_COLOR,
    )
    make_score_histogram(
        old_scores,
        OUTPUT_OLD_SCORE_HIST,
        "Unconstrained GPT score distribution",
        HIST_OLD_COLOR,
    )

    output = {
        "backend_name": BACKEND_NAME,
        "initial_layout": INITIAL_LAYOUT,
        "lambda_2q": LAMBDA_2Q,
        "lambda_depth": LAMBDA_DEPTH,
        "summary_trans": summarize_results(trans_results, "trans"),
        "summary_old": summarize_results(old_results, "old"),
        "trans_results": trans_results,
        "old_results": old_results,
        "plots": {
            "mean_two_qubit_gates": OUTPUT_2Q_BAR.name,
            "mean_depth": OUTPUT_DEPTH_BAR.name,
            "mean_transpilation_score": OUTPUT_SCORE_BAR.name,
            "trans_score_histogram": OUTPUT_TRANS_SCORE_HIST.name,
            "old_score_histogram": OUTPUT_OLD_SCORE_HIST.name,
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved summary to: {OUTPUT_JSON.name}")
    print(f"Saved plot to: {OUTPUT_2Q_BAR.name}")
    print(f"Saved plot to: {OUTPUT_DEPTH_BAR.name}")
    print(f"Saved plot to: {OUTPUT_SCORE_BAR.name}")
    print(f"Saved plot to: {OUTPUT_TRANS_SCORE_HIST.name}")
    print(f"Saved plot to: {OUTPUT_OLD_SCORE_HIST.name}")


if __name__ == "__main__":
    main()
