import random
import os

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # or point to data/qaoa explicitly

train_path = os.path.join(DATA_DIR, "train.txt")
val_path   = os.path.join(DATA_DIR, "val.txt")

with open(train_path) as f:
    lines = f.readlines()

random.shuffle(lines)

split = int(0.95 * len(lines))
train_lines = lines[:split]
val_lines = lines[split:]

with open(train_path, "w") as f:
    f.writelines(train_lines)

with open(val_path, "w") as f:
    f.writelines(val_lines)
