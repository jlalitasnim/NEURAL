import numpy as np


def make_xor(n=200, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 2)

    y = (
        ((X[:, 0] > 0) & (X[:, 1] > 0)) |
        ((X[:, 0] < 0) & (X[:, 1] < 0))
    ).astype(float)

    X += np.random.randn(n, 2) * noise

    return X, y.reshape(-1, 1)


def make_moons(n=300, noise=0.2, seed=42):
    np.random.seed(seed)

    half = n // 2

    theta = np.linspace(0, np.pi, half)
    X1 = np.c_[np.cos(theta), np.sin(theta)]

    X2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta) - 0.5]

    X = np.vstack([X1, X2])
    X += np.random.randn(n, 2) * noise

    y = np.hstack([np.zeros(half), np.ones(half)])

    return X, y.reshape(-1, 1)


def make_circles(n=300, noise=0.1, factor=0.5, seed=42):
    np.random.seed(seed)

    half = n // 2

    theta = np.linspace(0, 2 * np.pi, half)

    X1 = np.c_[np.cos(theta), np.sin(theta)]
    X2 = np.c_[factor * np.cos(theta), factor * np.sin(theta)]

    X = np.vstack([X1, X2])
    X += np.random.randn(n, 2) * noise

    y = np.hstack([np.zeros(half), np.ones(half)])

    return X, y.reshape(-1, 1)


def make_blobs(n=300, centers=3, std=1.0, seed=42):
    np.random.seed(seed)

    centers_coords = np.random.uniform(-5, 5, (centers, 2))

    X = []
    y = []

    per_class = n // centers

    for i, center in enumerate(centers_coords):
        points = np.random.randn(per_class, 2) * std + center
        X.append(points)
        y.append(np.full(per_class, i))

    return np.vstack(X), np.hstack(y).reshape(-1, 1)


def make_regression(n=200, noise=0.3, seed=42):
    np.random.seed(seed)

    X = np.random.randn(n, 1)

    y = 3 * X.squeeze() + np.sin(3 * X.squeeze()) + np.random.randn(n) * noise

    return X, y.reshape(-1, 1)


# 🔥 THIS IS WHAT YOUR APP NEEDS
DATASETS = {
    "XOR": make_xor,
    "Moons": make_moons,
    "Circles": make_circles,
    "Blobs": make_blobs,
    "Regression": make_regression,
}