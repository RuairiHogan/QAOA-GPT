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


def get_num_qubits_from_graph_tokens(tokens):
    nodes = set()
    for t in tokens:
        if t.startswith("(") and "," in t:
            u, v = t.strip("()").split(",")
            nodes.add(int(u))
            nodes.add(int(v))
    return max(nodes) + 1 if nodes else 0


def is_next_token_op_index(token):
    return token.startswith("<new_layer_")


def allowed_op_token_ids(n_qubits, coupling_map, stoi):
    op_list = []
    # single qubits
    for i in range(n_qubits):
        op_list.extend([f"X{i}", f"Y{i}", f"Z{i}"])

    # two-qubit only on connectivity edges
    for (i, j) in sorted({tuple(sorted(e)) for e in coupling_map}):
        if i < 0 or j < 0 or i >= n_qubits or j >= n_qubits or i == j:
            continue
        for B in ["X", "Y", "Z"]:
            for C in ["X", "Y", "Z"]:
                op_list.append(f"{B}{i}{C}{j}")

    allowed_ids = []
    for op_index in range(len(op_list)):
        tok = str(op_index)
        if tok in stoi:
            allowed_ids.append(stoi[tok])
    return allowed_ids


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

        n_qubits = get_num_qubits_from_graph_tokens(tokens)
        coupling_map = [(i, i + 1) for i in range(max(0, n_qubits - 1))]
        allowed_op_ids = set(allowed_op_token_ids(n_qubits, coupling_map, stoi))

        # Step-by-step generation with op-mask when the model should predict an operator index.
        generated = tokens.copy()
        x_step = x
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = x_step if x_step.size(1) <= model.config.block_size else x_step[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                if is_next_token_op_index(generated[-1]):
                    mask = torch.full_like(logits, -float('Inf'))
                    if len(allowed_op_ids) > 0:
                        mask[:, list(allowed_op_ids)] = logits[:, list(allowed_op_ids)]
                    logits = mask

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                x_step = torch.cat((x_step, idx_next), dim=1)
                generated.append(itos[idx_next.item()])

                if generated[-1] == STOP_TOKEN and len(generated) > len(tokens):
                    break

        out_tokens = generated[len(tokens):]

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
