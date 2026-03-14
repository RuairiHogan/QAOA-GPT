import torch
import pickle
import csv
from model import GPTConfig, GPT
import os
from datetime import datetime 

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "..", ".." , "QAOA-GPT Testing", "test_graphs.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", ".." , "QAOA-GPT Testing", "generated_circuits.txt")

out_dir = "out-qaoa"
data_dir = "data/qaoa"

device = "cpu"
max_new_tokens = 120
temperature = 0.8
top_k = 50

STOP_TOKEN = "<bos>"   # stop if GPT tries to start a new graph

# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------

with open(f"{data_dir}/meta.pkl", "rb") as f:
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

checkpoint = torch.load(f"{out_dir}/ckpt.pt", map_location=device)

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

start_time = datetime.now() 

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    reader = csv.reader(fin)

    for line_idx, row in enumerate(reader):
        # Remove optional seed token
        tokens = [t for t in row if not t.startswith("<seed=")]

        # Encode prompt
        x = encode(tokens)[None, :].to(device)

        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        out_tokens = decode(y[0].tolist())

        # --- Trim output if GPT starts another graph ---
        trimmed = []
        for t in out_tokens:
            if t == STOP_TOKEN and len(trimmed) > 0:
                break
            trimmed.append(t)

        # Write space-separated tokens
        fout.write(" ".join(trimmed) + "\n")

        print(f"Generated circuit {line_idx + 1}")

end_time = datetime.now()   
elapsed = end_time - start_time

print(f"\n✅ All circuits written to {OUTPUT_FILE}")
print(f"⏱️  Total generation time: {elapsed}")
