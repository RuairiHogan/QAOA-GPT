# config/qaoa_gpt.py
# NanoGPT config tuned for QAOA-GPT

# I/O
dataset = 'qaoa'          # folder name under data/
out_dir = 'out-qaoa'
eval_interval = 200
eval_iters = 50
log_interval = 20

vocab_size = 1061

# Training length
max_iters = 16000         # good starting point (was 20000)
lr_decay_iters = 16000  # was 20000
warmup_iters = 100

init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'


# Batch & sequence
batch_size = 16           # reduce to 32 if GPU memory limited (was 64)
block_size = 128          # more than enough for your circuits (was 256)
gradient_accumulation_steps = 3

# Model size (THIS IS IMPORTANT)
n_layer = 5               # depth was 6
n_head = 5                # attention heads was 6
n_embd = 320               # embedding size was 384

# Optimization
learning_rate = 1e-4 # was 3e-4
min_lr = 3e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Regularization
dropout = 0.1
bias = False

# AdamW
optimizer = 'adamw'

# Hardware
device = 'cpu'           # use 'cpu' if no GPU
dtype = 'float32'         # 'float32' if CPU
compile = False

# Disable wandb
wandb_log = False

# Evaluation
always_save_checkpoint = True
