# generate_circuits.py
# Works with your generate(self, idx, graph_emb, max_new_tokens, temperature=1.0, top_k=None) method.

import os
import pickle
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------
# PATHS / CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# One prompt per line (tokenized text, space-separated tokens)
INPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "test_graphs.txt")

# One embedding per line/prompt (shape: [N, D])
GRAPH_EMB_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "test_graph_emb.npy")

OUTPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "generated_circuits.txt")

out_dir = "out-qaoa"
data_dir = "data/qaoa"

device = "cuda" if torch.cuda.is_available() else "cpu"

max_new_tokens = 80
temperature = 0.7
top_k = 20

STOP_TOKEN = "<end_of_circuit>"
PAD_TOKEN = "<pad>"

# ---------------------------------------------------
# IMPORT YOUR MODEL
# ---------------------------------------------------
# Adjust these imports to your filenames/classes
from qaoa_model import QAOAgpt, QAOAConfig  # <-- change if your module/class names differ


# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------
meta_path = os.path.join(data_dir, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta.get("vocab_size", len(stoi))

if STOP_TOKEN not in stoi:
    raise ValueError(f"STOP_TOKEN {STOP_TOKEN} not found in tokenizer stoi")

EOS_ID = stoi[STOP_TOKEN]
PAD_ID = stoi.get(PAD_TOKEN, None)

def encode_tokens(tokens):
    ids = []
    for t in tokens:
        if t not in stoi:
            raise ValueError(f"Unknown token in prompt: {t}")
        ids.append(stoi[t])
    return torch.tensor(ids, dtype=torch.long)

def decode_ids(ids):
    return [itos[i] for i in ids]


# ---------------------------------------------------
# LOAD GRAPH EMBEDDINGS
# ---------------------------------------------------
graph_embs = np.load(GRAPH_EMB_FILE).astype(np.float32)
if graph_embs.ndim != 2:
    raise ValueError(f"Expected graph_embs to be 2D [N, D], got shape {graph_embs.shape}")

graph_embs = torch.from_numpy(graph_embs)  # [N, D]


# ---------------------------------------------------
# LOAD MODEL CHECKPOINT
# ---------------------------------------------------
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)

model_args = checkpoint.get("model_args", {})
# Fallbacks if something is missing
block_size = model_args.get("block_size", 128)
n_layer = model_args.get("n_layer", 6)
n_head = model_args.get("n_head", 6)
n_embd = model_args.get("n_embd", 384)
dropout = model_args.get("dropout", 0.0)
bias = model_args.get("bias", False)

# Your model likely needs this to match graph embedding dimension
graph_dim = model_args.get("graph_dim", graph_embs.shape[1])

conf = QAOAConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    graph_dim=graph_dim,
)

model = QAOAgpt(conf)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

# Sanity check graph dim
if graph_dim != graph_embs.shape[1]:
    raise ValueError(
        f"Model graph_dim={graph_dim} but embeddings have dim={graph_embs.shape[1]}. "
        f"Fix by setting graph_dim correctly in the config/checkpoint or regenerating embeddings."
    )


# ---------------------------------------------------
# GENERATE CIRCUITS
# ---------------------------------------------------
start_time = datetime.now()

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    lines = [ln.strip() for ln in fin if ln.strip()]

    if len(lines) != graph_embs.shape[0]:
        raise ValueError(
            f"Line count mismatch: {len(lines)} prompts in {INPUT_FILE} "
            f"but {graph_embs.shape[0]} embeddings in {GRAPH_EMB_FILE}"
        )

    for i, line in enumerate(lines):
        # token prompt
        prompt_tokens = line.split()

        # (optional) remove any <seed=...> tags if you use them
        prompt_tokens = [t for t in prompt_tokens if not t.startswith("<seed=")]

        x = encode_tokens(prompt_tokens)[None, :].to(device)  # [1, T]
        g = graph_embs[i].unsqueeze(0).to(device).float()     # [1, D]

        with torch.no_grad():
            y = model.generate(
                x,
                graph_emb=g,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        out_tokens = decode_ids(y[0].tolist())

        # Hard stop at first EOS token in the decoded output
        trimmed = []
        for t in out_tokens:
            trimmed.append(t)
            if t == STOP_TOKEN:
                break

        fout.write(" ".join(trimmed) + "\n")
        print(f"Generated circuit {i + 1}/{len(lines)}")

end_time = datetime.now()
print(f"\n✅ All circuits written to: {OUTPUT_FILE}")
print(f"⏱️  Total generation time: {end_time - start_time}")