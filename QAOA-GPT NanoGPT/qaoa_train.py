"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import QAOAgpt, QAOAConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

graph_dim = 128          # set this to the FEATHER embedding dimension you saved
eos_token_id = -1        # set >=0 to enable loss masking-after-EOS inside model.forward()

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# ✅ QAOA-GPT style data loader:
# expects per-instance token sequences + per-instance graph embeddings.
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

train_seqs_path = os.path.join(data_dir, "train_seqs.npy")
val_seqs_path   = os.path.join(data_dir, "val_seqs.npy")
train_gemb_path = os.path.join(data_dir, "train_graph_emb.npy")
val_gemb_path   = os.path.join(data_dir, "val_graph_emb.npy")

if master_process:
    print("Loading per-instance sequences + graph embeddings...")
    print("  ", train_seqs_path)
    print("  ", train_gemb_path)
    print("  ", val_seqs_path)
    print("  ", val_gemb_path)

train_seqs = np.load(train_seqs_path, allow_pickle=True)
val_seqs   = np.load(val_seqs_path, allow_pickle=True)
train_gemb = np.load(train_gemb_path).astype(np.float32)
val_gemb   = np.load(val_gemb_path).astype(np.float32)

assert len(train_seqs) == len(train_gemb), "train_seqs and train_graph_emb must be same length"
assert len(val_seqs) == len(val_gemb), "val_seqs and val_graph_emb must be same length"
assert train_gemb.shape[1] == graph_dim, f"graph_dim={graph_dim} but train_graph_emb has dim {train_gemb.shape[1]}"
assert val_gemb.shape[1] == graph_dim, f"graph_dim={graph_dim} but val_graph_emb has dim {val_gemb.shape[1]}"

def get_batch(split):
    seqs = train_seqs if split == "train" else val_seqs
    gemb = train_gemb if split == "train" else val_gemb

    B = batch_size
    T = block_size

    X = torch.empty((B, T), dtype=torch.long)
    Y = torch.empty((B, T), dtype=torch.long)
    G = torch.empty((B, graph_dim), dtype=torch.float32)

    # sample windows INSIDE one instance at a time (no cross-graph bleed)
    for i in range(B):
        j = np.random.randint(0, len(seqs))
        seq = seqs[j]

        if len(seq) < T + 1:
            raise ValueError(f"Sequence {j} length {len(seq)} < block_size+1 ({T+1}). Lower block_size or filter sequences.")

        start = np.random.randint(0, len(seq) - T - 1)
        chunk = seq[start : start + T + 1]

        # ensure integer type
        X[i] = torch.from_numpy(chunk[:-1].astype(np.int64))
        Y[i] = torch.from_numpy(chunk[1:].astype(np.int64))
        G[i] = torch.from_numpy(gemb[j])

    if device_type == 'cuda':
        X = X.pin_memory().to(device, non_blocking=True)
        Y = Y.pin_memory().to(device, non_blocking=True)
        G = G.pin_memory().to(device, non_blocking=True)
    else:
        X, Y, G = X.to(device), Y.to(device), G.to(device)

    return X, Y, G

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    if meta_vocab_size is not None:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# -----------------------------------------------------------------------------
# model init
# -----------------------------------------------------------------------------
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    graph_dim=graph_dim,
    eos_token_id=eos_token_id,
)

if init_from == 'scratch':
    print("Initializing a new QAOAgpt model from scratch")
    if meta_vocab_size is None:
        print("defaulting vocab_size to GPT-2 50304 (50257 rounded up)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    conf = QAOAConfig(**model_args)
    model = QAOAgpt(conf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    checkpoint_model_args = checkpoint['model_args']
    # ensure compatibility on resume
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'graph_dim', 'eos_token_id']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]

    conf = QAOAConfig(**model_args)
    model = QAOAgpt(conf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    raise RuntimeError("This QAOAgpt setup does not support loading OpenAI GPT-2 weights directly. Use init_from='scratch' or 'resume'.")

# crop down the model block size if desired
if block_size < model.config.block_size:
    # If you kept nanoGPT crop_block_size method, you can call it here.
    # Otherwise, just ensure your config.block_size equals block_size.
    try:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size
    except Exception:
        pass

model.to(device)

# initialize GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
# If your QAOAgpt class doesn't include configure_optimizers, adapt this accordingly.
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    # model = torch.compile(model)  # optional

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an accurate loss over either split
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, G = get_batch(split)
            with ctx:
                logits, loss = model(X, graph_emb=G, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, G = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, graph_emb=G, targets=Y)
            loss = loss / gradient_accumulation_steps

        # prefetch next batch
        X, Y, G = get_batch('train')

        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        # estimate_mfu exists on nanoGPT model; your QAOAgpt may not have it. Guard it.
        if local_iter_num >= 5 and hasattr(raw_model, "estimate_mfu"):
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
