import argparse
import json
from pathlib import Path


NEW_LAMBDA_2Q = 0.01


def compute_score(approx_ratio, two_qubit_count, depth, lambda_2q, lambda_depth):
    return approx_ratio - lambda_2q * two_qubit_count - lambda_depth * depth


def rewrite_dataset(input_path: Path, output_path: Path, lambda_2q: float) -> int:
    updated = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            approx_ratio = float(entry["approx_ratio"])
            two_qubit_count = int(entry["transpiled_two_qubit_count"])
            depth = int(entry["transpiled_depth"])
            lambda_depth = float(entry["lambda_depth"])

            entry["lambda_2q"] = lambda_2q
            entry["hardware_score"] = round(
                compute_score(
                    approx_ratio,
                    two_qubit_count,
                    depth,
                    lambda_2q,
                    lambda_depth,
                ),
                6,
            )

            dst.write(json.dumps(entry) + "\n")
            updated += 1

    return updated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a JSONL transpilation dataset with a new lambda_2q and "
            "recomputed hardware_score."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="transpilation_dataset.jsonl",
        help="Input JSONL dataset path.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="transpilation_dataset_lambda2q_0p01.jsonl",
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
        "--lambda-2q",
        type=float,
        default=NEW_LAMBDA_2Q,
        help="New lambda_2q value to apply to every record.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    updated = rewrite_dataset(input_path, output_path, args.lambda_2q)
    print(
        f"Updated {updated} records in '{output_path}' "
        f"with lambda_2q={args.lambda_2q}."
    )


if __name__ == "__main__":
    main()
