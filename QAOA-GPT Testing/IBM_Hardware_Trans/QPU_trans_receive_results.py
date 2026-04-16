import json
from statistics import mean, pstdev
from typing import Dict, List, Tuple
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService


# ============================================================
# USER SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_METADATA_FILE = BASE_DIR / "submitted_jobs_trans_optimal_50.json"
OUTPUT_RESULTS_FILE = BASE_DIR / "trans_qpu_results_optimal_50.json"
N_QUBITS = 7

# If you want explicit auth instead of saved account, uncomment:
# IBM_TOKEN = "YOUR_IBM_TOKEN"
# USE_EXPLICIT_TOKEN = True
USE_EXPLICIT_TOKEN = False


# ============================================================
# MAXCUT HELPERS
# ============================================================

def cut_value(bitstring: str, edges: List[List[float]]) -> float:
    bits = [int(b) for b in bitstring[::-1]]
    total = 0.0
    for u, v, w in edges:
        if bits[u] != bits[v]:
            total += w
    return total


def brute_force_maxcut(edges: List[List[float]], n_qubits: int = 7) -> float:
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


def expected_cut_from_counts(counts: Dict[str, int], edges: List[List[float]]) -> float:
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    return sum(cut_value(bitstring, edges) * c for bitstring, c in counts.items()) / total_shots


def best_observed_cut(counts: Dict[str, int], edges: List[List[float]]) -> Tuple[str, float]:
    best_bitstring = None
    best_value = -1.0
    for bitstring in counts:
        val = cut_value(bitstring, edges)
        if val > best_value:
            best_value = val
            best_bitstring = bitstring
    return best_bitstring, best_value


def top_counts(counts: Dict[str, int], k: int = 10) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]


# ============================================================
# RESULT EXTRACTION
# ============================================================

def extract_counts(pub_result) -> Dict[str, int]:
    """
    Robustly extract counts from a SamplerV2 pub result.
    Tries common classical register names first, then scans attributes.
    """
    data = pub_result.data

    # Common names
    for name in ["meas", "c", "cr", "bits"]:
        if hasattr(data, name):
            reg = getattr(data, name)
            if hasattr(reg, "get_counts"):
                return reg.get_counts()

    # Fallback: scan all public attributes
    available = []
    for name in dir(data):
        if name.startswith("_"):
            continue
        available.append(name)
        try:
            obj = getattr(data, name)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()
        except Exception:
            pass

    raise AttributeError(
        "Could not find a measurement register with get_counts() in DataBin. "
        f"Available attributes: {available}"
    )


# ============================================================
# SUMMARY HELPERS
# ============================================================

def safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    return pstdev(xs) if len(xs) > 1 else 0.0


def summarize_group(results: List[dict], label: str) -> dict:
    ars = [r["approx_ratio"] for r in results]
    exp_vals = [r["expected_cut"] for r in results]
    best_vals = [r["best_observed_cut"] for r in results]
    opt_vals = [r["optimal_cut"] for r in results]

    perfect_expected = sum(1 for r in results if abs(r["approx_ratio"] - 1.0) < 1e-9)
    perfect_observed = sum(
        1 for r in results if abs(r["best_observed_cut"] - r["optimal_cut"]) < 1e-9
    )

    return {
        "label": label,
        "num_circuits": len(results),
        "mean_approx_ratio": safe_mean(ars),
        "std_approx_ratio": safe_std(ars),
        "min_approx_ratio": min(ars) if ars else 0.0,
        "max_approx_ratio": max(ars) if ars else 0.0,
        "mean_expected_cut": safe_mean(exp_vals),
        "mean_best_observed_cut": safe_mean(best_vals),
        "mean_optimal_cut": safe_mean(opt_vals),
        "perfect_expected_count": perfect_expected,
        "perfect_observed_count": perfect_observed,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    with open(SUBMISSION_METADATA_FILE, "r", encoding="utf-8") as f:
        submissions = json.load(f)

    if USE_EXPLICIT_TOKEN:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
        )
    else:
        service = QiskitRuntimeService()

    all_results = []

    print(f"Loaded {len(submissions)} submitted job entries from {SUBMISSION_METADATA_FILE}")

    for sub_idx, sub in enumerate(submissions):
        job_id = sub["job_id"]
        print(f"\nFetching job {sub_idx + 1}/{len(submissions)}: {job_id}")

        job = service.job(job_id)
        status = str(job.status())
        print(f"  Status: {status}")

        result = job.result()
        circuit_meta = sub["circuits"]

        if len(result) != len(circuit_meta):
            print(
                f"  [WARN] result length {len(result)} != metadata length {len(circuit_meta)} "
                f"for job {job_id}"
            )

        for i, meta in enumerate(circuit_meta):
            counts = extract_counts(result[i])

            edges = meta["graph_edges"]
            expected_cut = expected_cut_from_counts(counts, edges)
            optimal_cut = brute_force_maxcut(edges, n_qubits=N_QUBITS)
            approx_ratio = expected_cut / optimal_cut if optimal_cut > 0 else 0.0

            best_bitstring, best_cut = best_observed_cut(counts, edges)

            entry = {
                "job_id": job_id,
                "global_index": meta["global_index"],
                "source_file": meta["source_file"],
                "source_line": meta["source_line"],
                "format_type": meta["format_type"],
                "graph_edges": edges,
                "layers": meta["layers"],
                "counts": counts,
                "top_counts": top_counts(counts, k=10),
                "expected_cut": expected_cut,
                "optimal_cut": optimal_cut,
                "approx_ratio": approx_ratio,
                "best_observed_bitstring": best_bitstring,
                "best_observed_cut": best_cut,
                "num_shots": sum(counts.values()),
            }

            all_results.append(entry)

    all_results.sort(key=lambda x: x["global_index"])

    print(f"\nRecovered {len(all_results)} total circuit results.")

    trans_half = [r for r in all_results if r["format_type"] == "trans"]
    baseline_half = [r for r in all_results if r["format_type"] == "old"]

    summary_trans_half = summarize_group(trans_half, "trans_half")
    summary_baseline_half = summarize_group(baseline_half, "baseline_half")
    summary_all = summarize_group(all_results, "all_results")

    comparison = {
        "trans_half_mean_ar": summary_trans_half["mean_approx_ratio"],
        "baseline_half_mean_ar": summary_baseline_half["mean_approx_ratio"],
        "difference_mean_ar_trans_minus_baseline": (
            summary_trans_half["mean_approx_ratio"] - summary_baseline_half["mean_approx_ratio"]
        ),
        "trans_half_mean_expected_cut": summary_trans_half["mean_expected_cut"],
        "baseline_half_mean_expected_cut": summary_baseline_half["mean_expected_cut"],
        "difference_mean_expected_cut_trans_minus_baseline": (
            summary_trans_half["mean_expected_cut"] - summary_baseline_half["mean_expected_cut"]
        ),
        "trans_half_perfect_observed_count": summary_trans_half["perfect_observed_count"],
        "baseline_half_perfect_observed_count": summary_baseline_half["perfect_observed_count"],
    }

    output = {
        "metadata_file": str(SUBMISSION_METADATA_FILE),
        "num_jobs": len(submissions),
        "num_results": len(all_results),
        "summary_all": summary_all,
        "summary_trans_half": summary_trans_half,
        "summary_baseline_half": summary_baseline_half,
        "comparison": comparison,
        "all_results": all_results,
    }

    with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved full results to: {OUTPUT_RESULTS_FILE}")

    print("\n=== SUMMARY ===")
    print("Trans circuits:")
    print(f"  mean AR = {summary_trans_half['mean_approx_ratio']:.6f}")
    print(f"  std AR  = {summary_trans_half['std_approx_ratio']:.6f}")
    print(f"  perfect observed = {summary_trans_half['perfect_observed_count']}")

    print("\nOld-format circuits:")
    print(f"  mean AR = {summary_baseline_half['mean_approx_ratio']:.6f}")
    print(f"  std AR  = {summary_baseline_half['std_approx_ratio']:.6f}")
    print(f"  perfect observed = {summary_baseline_half['perfect_observed_count']}")

    print("\nDifference:")
    print(
        f"  mean AR difference (trans - old) = "
        f"{comparison['difference_mean_ar_trans_minus_baseline']:.6f}"
    )


if __name__ == "__main__":
    main()
