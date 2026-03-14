# config/qaoa_gpt_overfit20.py

dataset = 'qaoa'
out_dir = 'out-qaoa-overfit20'

eval_interval = 20
eval_iters = 20
log_interval = 5

# Training length
max_iters = 10000         # good starting point (was 20000)
lr_decay_iters = 10000  # was 20000
warmup_iters = 100

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'


# Batch & sequence
batch_size = 1           # reduce to 32 if GPU memory limited (was 64)
block_size = 128  # was 128 but crashed          # more than enough for your circuits (was 256)
gradient_accumulation_steps = 1


n_layer = 5
n_head = 5
n_embd = 320

learning_rate = 3e-4
min_lr = 3e-4
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dropout = 0.0
bias = False

device = 'cuda'      # or 'cpu' if needed
dtype = 'float16'    # use 'float32' on cpu
compile = False

wandb_log = False
always_save_checkpoint = True
decay_lr = False