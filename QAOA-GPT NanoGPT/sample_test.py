import torch
import pickle
import json
from model import GPTConfig, GPT

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

out_dir = "out-qaoa"          # where ckpt.pt lives
data_dir = "data/qaoa"        # where meta.pkl lives
device = "cpu"
dtype = torch.float32

max_new_tokens = 120          # how long a circuit you want
temperature = 0.8
top_k = 100

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

ckpt_path = f"{out_dir}/ckpt.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

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
# BUILD PROMPT (GRAPH ONLY)
# ---------------------------------------------------

# Example: YOU MUST MATCH YOUR TRAINING FORMAT
prompt_tokens = [
    "<bos>",
    "(0,2)", "0.54",
    "(0,4)", "0.19",
    "(0,6)", "0.77",
    "(1,3)", "0.68",
    "(1,5)", "0.31",
    "(1,6)", "0.84",
    "(2,3)", "0.46",
    "(2,5)", "0.92",
    "(3,4)", "0.27",
    "(3,6)", "0.58",
    "(4,5)", "0.14",
    "(5,6)", "0.63",
    "<end_of_graph>"
]


x = encode(prompt_tokens)[None, :].to(device)

# ---------------------------------------------------
# GENERATE
# ---------------------------------------------------

# ---------------------------------------------------
# GENERATE (STOP AFTER ONE CIRCUIT)
# ---------------------------------------------------

model.eval()

generated = x.clone()

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_str = itos[next_token.item()]

        # 🚨 STOP CONDITION: model tries to start a new graph
        if token_str == "<bos>":
            break

        generated = torch.cat([generated, next_token], dim=1)

output_tokens = decode(generated[0].tolist())

print("\n=== GENERATED CIRCUIT (SINGLE GRAPH) ===")
print(" ".join(output_tokens))
