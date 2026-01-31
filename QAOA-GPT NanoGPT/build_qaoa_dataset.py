import os
import numpy as np
import pickle
import random

# -------------------------------------------------
# CONFIG — CHANGE THESE
# -------------------------------------------------
DATASET_DIR = "data/qaoa"
INPUT_TXT   = "../QAOA-GPT Data Preparation/train.txt"          # one circuit per line
GRAPH_EMB   = "../QAOA-GPT Data Preparation/graph_embeddings.npy"  # already exists
TRAIN_FRAC  = 0.9                     # 90/10 split
SEED        = 1337

# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------
meta_path = os.path.join(DATASET_DIR, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]

assert "<end_of_circuit>" in stoi, \
    "Tokenizer must contain <end_of_circuit> token"

EOS_ID = stoi["<end_of_circuit>"]

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
with open(INPUT_TXT, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

graph_emb = np.load(GRAPH_EMB).astype(np.float32)

assert len(lines) == len(graph_emb), \
    "graph_embeddings.npy must align 1:1 with text lines"

print(f"Loaded {len(lines)} circuits")

# -------------------------------------------------
# TOKENIZATION
# -------------------------------------------------
def tokenize_line(line: str) -> np.ndarray:
    tokens = []
    for tok in line.split():
        if tok not in stoi:
            raise ValueError(f"Unknown token: {tok}")
        tokens.append(stoi[tok])

    # enforce EOS at the end
    if tokens[-1] != EOS_ID:
        tokens.append(EOS_ID)

    return np.array(tokens, dtype=np.int64)

all_seqs = [tokenize_line(line) for line in lines]

# -------------------------------------------------
# SANITY CHECKS
# -------------------------------------------------
# each sequence must have exactly one EOS at the end
for i, seq in enumerate(all_seqs):
    if seq[-1] != EOS_ID:
        raise RuntimeError(f"Sequence {i} does not end with EOS")
    if (seq == EOS_ID).sum() != 1:
        raise RuntimeError(f"Sequence {i} has multiple EOS tokens")

print("✓ EOS sanity checks passed")

# -------------------------------------------------
# TRAIN / VAL SPLIT
# -------------------------------------------------
N = len(all_seqs)
indices = list(range(N))

random.seed(SEED)
random.shuffle(indices)

split = int(TRAIN_FRAC * N)
train_idx = indices[:split]
val_idx   = indices[split:]

train_seqs = [all_seqs[i] for i in train_idx]
val_seqs   = [all_seqs[i] for i in val_idx]

train_graph_emb = graph_emb[train_idx]
val_graph_emb   = graph_emb[val_idx]

# -------------------------------------------------
# SAVE OUTPUT FILES
# -------------------------------------------------
os.makedirs(DATASET_DIR, exist_ok=True)

np.save(
    os.path.join(DATASET_DIR, "train_seqs.npy"),
    np.array(train_seqs, dtype=object)
)
np.save(
    os.path.join(DATASET_DIR, "val_seqs.npy"),
    np.array(val_seqs, dtype=object)
)
np.save(
    os.path.join(DATASET_DIR, "train_graph_emb.npy"),
    train_graph_emb
)
np.save(
    os.path.join(DATASET_DIR, "val_graph_emb.npy"),
    val_graph_emb
)

print("✓ Dataset written:")
print(f"  train_seqs.npy        ({len(train_seqs)})")
print(f"  train_graph_emb.npy   ({train_graph_emb.shape})")
print(f"  val_seqs.npy          ({len(val_seqs)})")
print(f"  val_graph_emb.npy     ({val_graph_emb.shape})")
