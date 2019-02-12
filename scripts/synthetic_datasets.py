import numpy as np

def generate_gaussian(N, D, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (N, D))
    y = (np.random.normal(0, 1, N) > 0) * 2.0 - 1.0
    return X, y

def generate_gaussian_mixture(N, D, separation=1, offset=0, seed=0):
    assert(N % 2 == 0)
    np.random.seed(seed)
    Xp = np.random.normal(0, 1, (N // 2, D))
    Xp[:, 0] += separation / 2 + offset
    Xn = np.random.normal(0, 1, (N // 2, D))
    Xn[:, 0] += -separation / 2 + offset
    X = np.vstack((Xp, Xn))
    y = np.hstack((np.full(N // 2, 1), np.full(N // 2, -1)))
    return X, y