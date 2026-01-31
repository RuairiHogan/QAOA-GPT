import torch
import pickle
import csv
import os
from datetime import datetime
import numpy as np

from model import QAOAgpt, QAOAConfig

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR, "..", "QAOA-GPT Testing", "test_graphs.txt"
)
GRAPH_EMB_FILE = os.path.join(
    BASE_DIR, "..", "QAOA-GPT Testing", "test_graph_emb.npy"
)
OUTPUT_FILE = os.path.join(
    BASE_DIR, "..", "QAOA-GPT Testing", "generated_circuits.txt"
)

out_dir = "out-qaoa"
data_dir = "data/qaoa"

device = "cpu"

max_new_tokens = 120
temperature = 0.8
top_k = 50

STOP_TOKEN = "<bos>"   # stop if model tries to start a new graph

# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]

def encode(tokens):
    return torch.tensor([stoi[t] for t in tokens], dtype=torch.long)

def decode(indices):
    return [itos[i] for i in indices]

# ---------------------------------------------------
# LOAD GRAPH EMBEDDINGS
# ---------------------------------------------------
graph_embs = np.load(GRAPH_EMB_FILE).astype(np.float32)
graph_embs = torch.from_numpy(graph_embs)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
model_args = checkpoint["model_args"]

conf = QAOAConfig(
    vocab_size=vocab_size,
    block_size=model_args["block_size"],
    n_layer=model_args["n_layer"],
    n_head=model_args["n_head"],
    n_embd=model_args["n_embd"],
    dropout=model_args["dropout"],
    bias=model_args["bias"],
    graph_dim=model_args["graph_dim"],
    eos_token_id=model_args.get("eos_token_id", -1),
)

model = QAOAgpt(conf)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

# ---------------------------------------------------
# RUN GENERATION
# ---------------------------------------------------
start_time = datetime.now()

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for idx, line in enumerate(fin):
        
        tokens = line.strip().split()
        tokens = [t for t in tokens if not t.startswith("<seed=")]


        # Encode prompt
        x = encode(tokens)[None, :].to(device)

        # Fetch corresponding graph embedding
        g = graph_embs[idx].unsqueeze(0).to(device)

        with torch.no_grad():
            y = model.generate(
                x,
                graph_emb=g,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=conf.eos_token_id if conf.eos_token_id >= 0 else None,
            )

        out_tokens = decode(y[0].tolist())

        # Stop if model tries to start a new graph
        trimmed = []
        for t in out_tokens:
            if t == STOP_TOKEN and len(trimmed) > 0:
                break
            trimmed.append(t)

        fout.write(" ".join(trimmed) + "\n")
        print(f"Generated circuit {idx + 1}")

end_time = datetime.now()
elapsed = end_time - start_time

print(f"\n✅ All circuits written to {OUTPUT_FILE}")
print(f"⏱️  Total generation time: {elapsed}")
