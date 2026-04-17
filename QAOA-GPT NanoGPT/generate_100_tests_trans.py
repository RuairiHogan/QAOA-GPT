import torch
import pickle
import os
from datetime import datetime
from model import GPTConfig, GPT

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "test_graphs_trans.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "generated_circuits_trans_100x10.txt")

out_dir = "out-qaoa"
checkpoint_file = "second_transpilation.pt"
data_dir = "data/qaoa"

device = "cpu"
max_new_tokens = 50
temperature = 0.8
top_k = 50

NUM_GRAPHS_TO_USE = 100
NUM_SAMPLES_PER_GRAPH = 10
MAX_ATTEMPTS_PER_SAMPLE = 10

END_CIRCUIT_TOKEN = "<end_of_circuit>"

# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------
with open(f"{data_dir}/meta_for_both_trans.pkl", "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]


def encode(tokens):
    return torch.tensor([stoi[t] for t in tokens], dtype=torch.long)


def decode(indices):
    return [itos[i] for i in indices]


def trim_generated_circuit(out_tokens):
    """
    Keep the generated sequence up to the first end-of-circuit marker.
    """
    trimmed = []
    for t in out_tokens:
        trimmed.append(t)
        if t == END_CIRCUIT_TOKEN:
            return trimmed, True
    return trimmed, False


def generate_one_sample(prompt_tokens, model, temperature, top_k):
    """
    Generate one stochastic circuit continuation from a single prompt.
    Retry until a full circuit is produced or the attempt budget is exhausted.
    """
    last_tokens = None
    for _ in range(MAX_ATTEMPTS_PER_SAMPLE):
        x = encode(prompt_tokens)[None, :].to(device)

        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        out_tokens = decode(y[0].tolist())
        trimmed_tokens, is_complete = trim_generated_circuit(out_tokens)
        last_tokens = trimmed_tokens
        if is_complete:
            return trimmed_tokens, True

    return last_tokens, False


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
checkpoint = torch.load(
    os.path.join(out_dir, checkpoint_file),
    map_location=device
)

gptconf = GPTConfig(
    vocab_size=vocab_size,
    block_size=checkpoint["model_args"]["block_size"],
    n_layer=checkpoint["model_args"]["n_layer"],
    n_head=checkpoint["model_args"]["n_head"],
    n_embd=checkpoint["model_args"]["n_embd"],
    bias=checkpoint["model_args"]["bias"],
)

model = GPT(gptconf)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

# ---------------------------------------------------
# RUN GENERATION
# ---------------------------------------------------
start_time = datetime.now()

graphs_used = 0
samples_written = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
    open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_idx, line in enumerate(fin, start=1):
        if graphs_used >= NUM_GRAPHS_TO_USE:
            break

        tokens = [t.strip() for t in line.strip().split() if t.strip() and not t.startswith("<seed=")]

        if not tokens:
            continue

        try:
            # sanity check that all prompt tokens are in vocab
            _ = encode(tokens)
        except KeyError as e:
            print(f"Skipping graph {line_idx}: unknown token {e}")
            continue

        graphs_used += 1
        print(f"\nGraph {graphs_used}/{NUM_GRAPHS_TO_USE} (source line {line_idx})")

        for sample_idx in range(1, NUM_SAMPLES_PER_GRAPH + 1):
            generated_tokens, is_complete = generate_one_sample(
                prompt_tokens=tokens,
                model=model,
                temperature=temperature,
                top_k=top_k,
            )

            if not is_complete:
                raise RuntimeError(
                    f"Failed to generate a complete circuit for source line {line_idx} "
                    f"sample {sample_idx} after {MAX_ATTEMPTS_PER_SAMPLE} attempts"
                )

            final_tokens = generated_tokens

            fout.write(" ".join(final_tokens) + "\n")
            fout.flush()

            samples_written += 1
            print(
                f"  wrote sample {sample_idx}/{NUM_SAMPLES_PER_GRAPH} "
                f"(total {samples_written})"
            )

end_time = datetime.now()
elapsed = end_time - start_time

print(f"\n✅ All generated circuits written to {OUTPUT_FILE}")
print(f"✅ Graphs used: {graphs_used}")
print(f"✅ Total samples written: {samples_written}")
print(f"⏱️ Total generation time: {elapsed}")
