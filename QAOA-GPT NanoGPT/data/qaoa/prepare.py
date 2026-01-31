# prepare.py
import os
import pickle
import numpy as np

dataset = "qaoa"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = BASE_DIR 

vocab_file = os.path.join(data_dir, "vocab.txt")

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
# WRITE META (THIS IS WHAT YOU WERE MISSING)
# ----------------------------
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}

with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Saved meta.pkl")
