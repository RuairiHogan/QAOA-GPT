vocab = set()

with open("train_new_bins.txt") as f:
    for line in f:
        for tok in line.strip().split():
            vocab.add(tok)

vocab = sorted(vocab)

with open("vocab_new_bins.txt", "w") as f:
    for tok in vocab:
        f.write(tok + "\n")

print(f"Vocab size: {len(vocab)}")
