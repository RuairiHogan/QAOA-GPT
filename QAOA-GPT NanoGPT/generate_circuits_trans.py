import os
from datetime import datetime

import pickle
import torch

from model import GPTConfig, GPT

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, ".." , "QAOA-GPT Testing", "test_graphs_trans_optimal_50.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, ".." , "QAOA-GPT Testing", "generated_circuits_trans_optimal_50.txt")

CHECKPOINT_FILE = os.path.join(BASE_DIR, "out-qaoa", "second_transpilation.pt")
META_FILE = os.path.join(BASE_DIR, "data", "qaoa", "meta_for_both_trans.pkl")

device = "cpu"
max_new_tokens = 180
temperature = 0.8
top_k = 50

END_CIRCUIT_TOKEN = "<end_of_circuit>"

# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]


def encode(tokens):
    return torch.tensor([stoi[t] for t in tokens], dtype=torch.long)


def decode(indices):
    return [itos[i] for i in indices]


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)

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
# RUN GENERATION FOR EACH GRAPH
# ---------------------------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

start_time = datetime.now()

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line_idx, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue

        # Tokenize by whitespace because transformed examples include commas in edge tuples.
        tokens = line.split()
        tokens = [t for t in tokens if not t.startswith("<seed=")]

        x = encode(tokens)[None, :].to(device)

        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        out_tokens = decode(y[0].tolist())

        # Keep generated sequence up to the first end-of-circuit marker.
        trimmed = []
        for t in out_tokens:
            trimmed.append(t)
            if t == END_CIRCUIT_TOKEN:
                break

        fout.write(" ".join(trimmed) + "\n")
        print(f"Generated circuit {line_idx + 1}")

elapsed = datetime.now() - start_time
print(f"\nAll circuits written to {OUTPUT_FILE}")
print(f"Total generation time: {elapsed}")
