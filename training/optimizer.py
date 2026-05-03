import numpy as np


def get_batches(X, y, batch_size, shuffle=True):
    """
    Generate mini-batches for training.
    """

    n = X.shape[0]

    if shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]