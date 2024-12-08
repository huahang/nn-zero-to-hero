import numpy as np

def load_data(path, block_size=3):
  words = open(path, "r").read().splitlines()
  all_letters = []
  for word in words:
    for letter in word:
      all_letters.append(letter)
  all_letters = sorted(list(set(all_letters)))
  stoi = {letter: i+1 for i, letter in enumerate(all_letters)}
  itos = {i+1: letter for i, letter in enumerate(all_letters)}
  stoi["."] = 0
  itos[0] = "."
  X = []
  Y = []
  for word in words:
    chs = "." * block_size + word + "."
    for i in range(len(chs) - block_size):
      x = chs[i:i+block_size]
      y = chs[i+block_size]
      x = [stoi[ch] for ch in x]
      y = [stoi[y]]
      X.append(x)
      Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  N = len(X)
  N_train = int(0.8 * N)
  N_val = (len(X) - N_train) // 2
  N_test = N - N_train - N_val
  X_train = X[:N_train]
  Y_train = Y[:N_train]
  X_val = X[N_train:N_train+N_val]
  Y_val = Y[N_train:N_train+N_val]
  X_test = X[N_train+N_val:]
  Y_test = Y[N_train+N_val:]
  X = {
    "train": X_train,
    "val": X_val,
    "test": X_test
  }
  Y = {
    "train": Y_train,
    "val": Y_val,
    "test": Y_test
  }
  return X, Y, stoi, itos
