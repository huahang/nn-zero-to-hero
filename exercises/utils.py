import torch
import random


def load_data(path, block_size=3):
    words = open(path, "r").read().splitlines()
    all_letters = []
    for word in words:
        for letter in word:
            all_letters.append(letter)
    all_letters = sorted(list(set(all_letters)))
    stoi = {letter: i + 1 for i, letter in enumerate(all_letters)}
    itos = {i + 1: letter for i, letter in enumerate(all_letters)}
    stoi["."] = 0
    itos[0] = "."

    def build_dataset(words):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # crop and append
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(X.shape, Y.shape)
        return X, Y

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])
    return Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos
