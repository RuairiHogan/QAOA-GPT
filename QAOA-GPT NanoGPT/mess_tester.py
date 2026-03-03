import numpy as np
a = np.load("data/qaoa/train_seqs.npy", allow_pickle=True)
print(a.dtype, a.shape, type(a[0]))