vocab = set()

with open("train.txt") as f:
    for line in f:
        for tok in line.strip().split():
            vocab.add(tok)

vocab = sorted(vocab)

with open("vocab.txt", "w") as f:
    for tok in vocab:
        f.write(tok + "\n")

print(f"Vocab size: {len(vocab)}")
