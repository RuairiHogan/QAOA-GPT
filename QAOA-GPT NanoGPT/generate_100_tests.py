import torch
import pickle
import os
import csv
from datetime import datetime
from model import GPTConfig, GPT

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "test_graphs_kingston.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "QAOA-GPT Testing", "generated_circuits_kingston_100x10.txt")

out_dir = "out-qaoa"
checkpoint_file = "third_hardware_kingston.pt"
data_dir = "data/qaoa"

device = "cpu"
max_new_tokens = 120
temperature = 0.8
top_k = 50

NUM_GRAPHS_TO_USE = 100
NUM_SAMPLES_PER_GRAPH = 10

STOP_TOKEN = "<bos>"

KINGSTON_7Q_COUPLING_MAP = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (3, 6),
]

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


def get_num_qubits_from_graph_tokens(tokens):
    nodes = set()
    for t in tokens:
        if t.startswith("(") and "," in t and t.endswith(")"):
            try:
                u, v = t.strip("()").split(",")
                nodes.add(int(u))
                nodes.add(int(v))
            except ValueError:
                pass
    return max(nodes) + 1 if nodes else 0


def is_next_token_op_index(token):
    return token.startswith("<new_layer_")


def allowed_op_token_ids(n_qubits, coupling_map, stoi):
    """
    Reconstruct the operator ordering exactly as in training:
      singles: X_i, Y_i, Z_i
      then two-qubit strings on allowed coupling edges:
      B_i_C_j for B,C in {X,Y,Z}
    The model emits operator indices as string tokens: "0", "1", ...
    """
    op_list = []

    # singles first
    for i in range(n_qubits):
        op_list.append(f"X_{i}")
        op_list.append(f"Y_{i}")
        op_list.append(f"Z_{i}")

    # two-qubit strings on allowed edges only
    edge_set = sorted({tuple(sorted(e)) for e in coupling_map})
    for (i, j) in edge_set:
        if i < 0 or j < 0 or i >= n_qubits or j >= n_qubits or i == j:
            continue
        for B in ["X", "Y", "Z"]:
            for C in ["X", "Y", "Z"]:
                op_list.append(f"{B}_{i}_{C}_{j}")

    allowed_ids = []
    for op_index in range(len(op_list)):
        tok = str(op_index)
        if tok in stoi:
            allowed_ids.append(stoi[tok])

    return allowed_ids


def extract_maxcut_graph_body(tokens):
    """
    Extract only the weighted MaxCut graph:
      <maxcut_graph> (u,v) w ... <end_of_maxcut_graph>

    Ignore anything else in that section, such as:
      <IBM_KINGSTON>
      <hardware_graph>
      unweighted hardware edges
      <end_of_hardware_graph>
    """
    if "<maxcut_graph>" not in tokens or "<end_of_maxcut_graph>" not in tokens:
        return []

    start = tokens.index("<maxcut_graph>") + 1
    end = tokens.index("<end_of_maxcut_graph>")
    section = tokens[start:end]

    clean = []
    i = 0
    while i < len(section):
        tok = section[i]

        if tok.startswith("(") and tok.endswith(")") and "," in tok and i + 1 < len(section):
            try:
                float(section[i + 1])
                clean.append(tok)
                clean.append(section[i + 1])
                i += 2
                continue
            except ValueError:
                pass

        i += 1

    return clean


def trim_generated_circuit(out_tokens):
    """
    Keep only circuit layer blocks:
      <new_layer_k> op_idx gamma beta

    Stop before:
      <bos>
      <end_of_circuit>
      <tier_*>

    Skip:
      duplicate <circuit>
      any stray tokens
    """
    cleaned = []
    i = 0

    while i < len(out_tokens):
        t = out_tokens[i]

        if t == STOP_TOKEN or t == "<end_of_circuit>" or t.startswith("<tier_"):
            break

        if t == "<circuit>":
            i += 1
            continue

        if t.startswith("<new_layer_"):
            if i + 3 >= len(out_tokens):
                break
            cleaned.extend([
                out_tokens[i],
                out_tokens[i + 1],
                out_tokens[i + 2],
                out_tokens[i + 3],
            ])
            i += 4
            continue

        i += 1

    return cleaned


def generate_one_sample(prompt_tokens, model, n_qubits, allowed_op_ids, temperature, top_k):
    """
    Generate one stochastic circuit continuation from a single prompt.
    """
    x = encode(prompt_tokens)[None, :].to(device)

    generated = prompt_tokens.copy()
    x_step = x

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = (
                x_step
                if x_step.size(1) <= model.config.block_size
                else x_step[:, -model.config.block_size:]
            )

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # If previous token is <new_layer_k>, next token must be
            # a legal operator-index token on the Kingston coupling map.
            if is_next_token_op_index(generated[-1]):
                mask = torch.full_like(logits, -float("Inf"))
                if allowed_op_ids:
                    allowed_list = list(allowed_op_ids)
                    mask[:, allowed_list] = logits[:, allowed_list]
                logits = mask

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            x_step = torch.cat((x_step, idx_next), dim=1)
            next_token = itos[idx_next.item()]
            generated.append(next_token)

            if next_token == STOP_TOKEN and len(generated) > len(prompt_tokens):
                break

    out_tokens = generated[len(prompt_tokens):]
    return trim_generated_circuit(out_tokens)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
checkpoint = torch.load(
    os.path.join(out_dir, checkpoint_file),
    map_location=device
)

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
# RUN GENERATION
# ---------------------------------------------------
start_time = datetime.now()

graphs_used = 0
samples_written = 0

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    reader = csv.reader(fin)

    for line_idx, row in enumerate(reader, start=1):
        if graphs_used >= NUM_GRAPHS_TO_USE:
            break

        tokens = [t.strip() for t in row if t.strip() and not t.startswith("<seed=")]

        if not tokens:
            continue

        n_qubits = get_num_qubits_from_graph_tokens(tokens)
        if n_qubits != 7:
            print(f"Skipping graph {line_idx}: expected 7 qubits, got {n_qubits}")
            continue

        try:
            # sanity check that all prompt tokens are in vocab
            _ = encode(tokens)
        except KeyError as e:
            print(f"Skipping graph {line_idx}: unknown token {e}")
            continue

        graph_body = extract_maxcut_graph_body(tokens)
        if not graph_body:
            print(f"Skipping graph {line_idx}: could not extract MaxCut graph body")
            continue

        allowed_op_ids = set(
            allowed_op_token_ids(n_qubits, KINGSTON_7Q_COUPLING_MAP, stoi)
        )

        graphs_used += 1
        print(f"\nGraph {graphs_used}/{NUM_GRAPHS_TO_USE} (source line {line_idx})")

        for sample_idx in range(1, NUM_SAMPLES_PER_GRAPH + 1):
            circuit_tokens = generate_one_sample(
                prompt_tokens=tokens,
                model=model,
                n_qubits=n_qubits,
                allowed_op_ids=allowed_op_ids,
                temperature=temperature,
                top_k=top_k,
            )

            final_tokens = (
                ["<maxcut_graph>"]
                + graph_body
                + ["<end_of_maxcut_graph>", "<circuit>"]
                + circuit_tokens
                + ["<end_of_circuit>"]
            )

            fout.write(" ".join(final_tokens) + "\n")
            fout.flush()

            samples_written += 1
            print(
                f"  wrote sample {sample_idx}/{NUM_SAMPLES_PER_GRAPH} "
                f"(total {samples_written})"
            )

end_time = datetime.now()
elapsed = end_time - start_time

print(f"\n✅ All generated circuits written to {OUTPUT_FILE}")
print(f"✅ Graphs used: {graphs_used}")
print(f"✅ Total samples written: {samples_written}")
print(f"⏱️ Total generation time: {elapsed}")