import os
import numpy as np
import pickle
import random

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_DIR = "data/qaoa"
INPUT_TXT   = "../QAOA-GPT Data Preparation/train.txt"
GRAPH_EMB   = "../QAOA-GPT Data Preparation/graph_embeddings.npy"
TRAIN_FRAC  = 0.9
SEED        = 1337

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<end_of_circuit>"

# IMPORTANT: must match your NanoGPT model block_size
BLOCK_SIZE = 128

# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------
meta_path = os.path.join(DATASET_DIR, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]

# -------------------------------------------------
# ADD PAD TOKEN (ID = 0)
# -------------------------------------------------
if PAD_TOKEN not in stoi:
    print("Adding <pad> token to tokenizer")

    # shift all existing tokens up by +1
    stoi = {tok: idx + 1 for tok, idx in stoi.items()}
    itos = {idx + 1: tok for idx, tok in itos.items()}

    stoi[PAD_TOKEN] = 0
    itos[0] = PAD_TOKEN

assert stoi[PAD_TOKEN] == 0

# -------------------------------------------------
# EOS CHECK
# -------------------------------------------------
assert EOS_TOKEN in stoi, "Tokenizer must contain <end_of_circuit> token"

EOS_ID = stoi[EOS_TOKEN]
PAD_ID = stoi[PAD_TOKEN]

# -------------------------------------------------
# SAVE UPDATED TOKENIZER
# -------------------------------------------------
meta["stoi"] = stoi
meta["itos"] = itos

with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

print(f"Tokenizer vocab size: {len(stoi)}")
print(f"PAD_ID={PAD_ID}, EOS_ID={EOS_ID}, BLOCK_SIZE={BLOCK_SIZE}")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
with open(INPUT_TXT, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

graph_emb = np.load(GRAPH_EMB).astype(np.float32)

assert len(lines) == len(graph_emb), \
    "graph_embeddings.npy must align 1:1 with text lines"

print(f"Loaded {len(lines)} circuits")
print(f"Loaded embeddings: {graph_emb.shape}, dtype={graph_emb.dtype}")

# -------------------------------------------------
# TOKENIZATION
# -------------------------------------------------
def tokenize_line(line: str) -> np.ndarray:
    ids = []
    for tok in line.split():
        if tok not in stoi:
            raise ValueError(f"Unknown token: {tok}")
        ids.append(stoi[tok])

    if len(ids) == 0:
        raise ValueError("Empty token sequence")

    # enforce exactly one EOS at the end
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)

    return np.array(ids, dtype=np.int64)

all_seqs = [tokenize_line(line) for line in lines]

# -------------------------------------------------
# SANITY CHECKS (pre-pad)
# -------------------------------------------------
for i, seq in enumerate(all_seqs):
    if seq[-1] != EOS_ID:
        raise RuntimeError(f"Sequence {i} does not end with EOS")
    if (seq == EOS_ID).sum() != 1:
        raise RuntimeError(f"Sequence {i} has multiple EOS tokens")
    if (seq == PAD_ID).any():
        raise RuntimeError(f"Sequence {i} contains PAD tokens (should not)")

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

assert len(train_seqs) == len(train_graph_emb)
assert len(val_seqs) == len(val_graph_emb)

# -------------------------------------------------
# PAD / TRUNCATE TO FIXED LENGTH (NO OBJECT ARRAYS)
# -------------------------------------------------
def pad_or_trunc(seq: np.ndarray, T: int, pad_id: int) -> np.ndarray:
    """
    Returns a length-T int64 array.
    - If seq is longer than T: truncate to first T tokens.
    - If seq is shorter than T: pad with pad_id at the end.
    """
    if seq.shape[0] >= T:
        return seq[:T].astype(np.int64, copy=False)

    out = np.full((T,), pad_id, dtype=np.int64)
    out[:seq.shape[0]] = seq
    return out

train_tok = np.stack([pad_or_trunc(s, BLOCK_SIZE, PAD_ID) for s in train_seqs], axis=0)
val_tok   = np.stack([pad_or_trunc(s, BLOCK_SIZE, PAD_ID) for s in val_seqs], axis=0)

# -------------------------------------------------
# SANITY CHECKS (post-pad)
# -------------------------------------------------
if train_tok.dtype == object or val_tok.dtype == object:
    raise RuntimeError("Bug: still produced object arrays (should never happen)")

# EOS should appear at least once before padding; after truncation it *might* be lost
# if BLOCK_SIZE is too small. Warn if that happens.
def eos_present_fraction(tok_arr: np.ndarray, eos_id: int) -> float:
    return float(np.mean(np.any(tok_arr == eos_id, axis=1)))

train_eos_frac = eos_present_fraction(train_tok, EOS_ID)
val_eos_frac = eos_present_fraction(val_tok, EOS_ID)

if train_eos_frac < 1.0 or val_eos_frac < 1.0:
    print("WARNING: Some sequences lost EOS due to truncation.")
    print(f"  train EOS-present fraction: {train_eos_frac:.4f}")
    print(f"  val   EOS-present fraction: {val_eos_frac:.4f}")
    print("  Consider increasing BLOCK_SIZE so EOS is always included.")

print("✓ Padding/truncation complete")
print(f"train_tok: {train_tok.shape}, dtype={train_tok.dtype}")
print(f"val_tok:   {val_tok.shape}, dtype={val_tok.dtype}")

# -------------------------------------------------
# SAVE OUTPUT FILES
# -------------------------------------------------
os.makedirs(DATASET_DIR, exist_ok=True)

np.save(os.path.join(DATASET_DIR, "train_seqs.npy"), train_tok)
np.save(os.path.join(DATASET_DIR, "val_seqs.npy"), val_tok)

np.save(os.path.join(DATASET_DIR, "train_graph_emb.npy"), train_graph_emb.astype(np.float32, copy=False))
np.save(os.path.join(DATASET_DIR, "val_graph_emb.npy"), val_graph_emb.astype(np.float32, copy=False))

print("✓ Dataset written:")
print(f"  train_seqs.npy        {train_tok.shape} {train_tok.dtype}")
print(f"  train_graph_emb.npy   {train_graph_emb.shape} {train_graph_emb.dtype}")
print(f"  val_seqs.npy          {val_tok.shape} {val_tok.dtype}")
print(f"  val_graph_emb.npy     {val_graph_emb.shape} {val_graph_emb.dtype}")