# Build a training corpus from the transpilation JSONL dataset.
# Output format per line: <score_{scorebin}> followed by the token sequence.

import json

INPUT = "../QAOA-GPT Training Data/transpilation_dataset_canonical.jsonl"
OUTPUT = "train.txt"

# Set this to False if lower scores should be considered better.
HIGHER_SCORE_IS_BETTER = True

# Adjustable cutoffs for score bins.
# For HIGHER_SCORE_IS_BETTER=True:
# - score >= ELITE_MIN       -> elite
# - score >= GOOD_MIN        -> good
# - score >= ACCEPTABLE_MIN  -> acceptable
# - otherwise                -> poor
#
# Defaults are based on approximate quartiles of the current dataset.
ELITE_MIN = -0.15
GOOD_MIN = -0.591
ACCEPTABLE_MIN = -1.05


def score_to_bin(score: float) -> str:
    if HIGHER_SCORE_IS_BETTER:
        if score >= ELITE_MIN:
            return "elite"
        if score >= GOOD_MIN:
            return "good"
        if score >= ACCEPTABLE_MIN:
            return "acceptable"
        return "poor"

    if score <= ELITE_MIN:
        return "elite"
    if score <= GOOD_MIN:
        return "good"
    if score <= ACCEPTABLE_MIN:
        return "acceptable"
    return "poor"


def strip_transpilation_metrics(tokens: list[str]) -> list[str]:
    cleaned = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "<transpilation_metrics>":
            # Skip until and including the closing transpilation metrics token.
            while i < len(tokens) and tokens[i] != "<end_of_transpilation_metrics>":
                i += 1
            if i < len(tokens):
                i += 1
            continue

        cleaned.append(tok)
        i += 1

    return cleaned


written = 0
skipped = 0

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        entry = json.loads(line)

        tokens = entry.get("tokens")
        score = entry.get("hardware_score")

        if not isinstance(tokens, list) or score is None:
            skipped += 1
            continue

        score_bin = score_to_bin(float(score))
        score_token = f"<score_{score_bin}>"

        filtered_tokens = strip_transpilation_metrics(tokens)
        full_sequence = [score_token] + filtered_tokens
        fout.write(" ".join(full_sequence) + "\n")
        written += 1

print(f"Corpus written to {OUTPUT}. lines_written={written}, lines_skipped={skipped}")
