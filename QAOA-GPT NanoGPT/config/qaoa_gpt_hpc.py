# config/qaoa_gpt.py
# NanoGPT config tuned for QAOA-GPT

# I/O
dataset = 'qaoa'          # folder name under data/
out_dir = 'out-qaoa'
eval_interval = 200
eval_iters = 100
log_interval = 50

vocab_size = 1766

# Training length
max_iters = 40000         # good starting point (was 20000)
lr_decay_iters = 40000  # was 20000
warmup_iters = 1500

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'


# Batch & sequence
batch_size = 32           # reduce to 32 if GPU memory limited (was 64)
block_size = 128          # more than enough for your circuits (was 256)
gradient_accumulation_steps = 16

# Model size (THIS IS IMPORTANT)
n_layer = 7               # depth was 6
n_head = 7                # attention heads was 6
n_embd = 448               # embedding size was 384

# Optimization
learning_rate = 1e-4 # was 3e-4
min_lr = 3e-5
weight_decay = 1.5e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Regularization
dropout = 0.3
bias = False

# AdamW
optimizer = 'adamw'

# Hardware
device = 'cuda'           # use 'cpu' if no GPU
dtype = 'bfloat16'         # 'float32' if CPU
compile = True

# Disable wandb
wandb_log = False

# Evaluation
always_save_checkpoint = True
