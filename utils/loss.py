import numpy as np


EPS = 1e-15


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


def binary_cross_entropy_grad(y_true, y_pred):
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return (y_pred - y_true) / y_pred.shape[0]


def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def categorical_cross_entropy_grad(y_true, y_pred):
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return (y_pred - y_true) / y_pred.shape[0]


def get_loss(name):
    losses = {
        "mse": (mse, mse_grad),
        "binary_cross_entropy": (binary_cross_entropy, binary_cross_entropy_grad),
        "categorical_cross_entropy": (categorical_cross_entropy, categorical_cross_entropy_grad),
    }

    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}")

    return losses[name]  # ← this line was missing