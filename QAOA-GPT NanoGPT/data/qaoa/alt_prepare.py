# prepare.py
import os
import pickle
import numpy as np

dataset = "qaoa"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = BASE_DIR 

train_file = os.path.join(data_dir, "train_new_bins.txt")
val_file   = os.path.join(data_dir, "val_new_bins.txt")
vocab_file = os.path.join(data_dir, "vocab_new_bins.txt")

# ----------------------------
# Load vocab
# ----------------------------
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f if line.strip()]

stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

# ----------------------------
# Tokenize helper
# ----------------------------
def tokenize(path):
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    ids = [stoi[t] for t in tokens]
    return np.array(ids, dtype=np.uint16)

# ----------------------------
# Write binaries
# ----------------------------
train_ids = tokenize(train_file)
train_ids.tofile(os.path.join(data_dir, "train_new_bins.bin"))

if os.path.exists(val_file):
    val_ids = tokenize(val_file)
    val_ids.tofile(os.path.join(data_dir, "val_new_bins.bin"))

# ----------------------------
# WRITE META (THIS IS WHAT YOU WERE MISSING)
# ----------------------------
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}

with open(os.path.join(data_dir, "meta_new_bins.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Saved meta_new_bins.pkl")
