import numpy as np


class HistoricalPerceptron:
    """
    Rosenblatt perceptron (1957).
    Uses a hard step function.
    Only works for linearly separable data.
    """

    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def _step(self, Z):
        return (Z >= 0).astype(float)

    def fit(self, X, y, X_val=None, y_val=None):
        n, f = X.shape
        self.weights = np.zeros(f)
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y.flatten()):
                pred = self._step(xi @ self.weights + self.bias)
                err = yi - pred
                self.weights += self.lr * err * xi
                self.bias += self.lr * err

            train_pred = self.predict(X).flatten()
            self.history["train_loss"].append(np.mean((y.flatten() - train_pred) ** 2))
            self.history["train_acc"].append(np.mean(train_pred == y.flatten()))

            if X_val is not None:
                val_pred = self.predict(X_val).flatten()
                self.history["val_loss"].append(np.mean((y_val.flatten() - val_pred) ** 2))
                self.history["val_acc"].append(np.mean(val_pred == y_val.flatten()))
            else:
                self.history["val_loss"].append(None)
                self.history["val_acc"].append(None)

        return self.history

    def predict(self, X):
        return self._step(X @ self.weights + self.bias).reshape(-1, 1)


class Perceptron:
    """
    Modern perceptron with sigmoid activation.
    """

    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def fit(self, X, y, X_val=None, y_val=None):
        n, f = X.shape
        self.weights = np.random.randn(f) * 0.01
        self.bias = 0.0

        for _ in range(self.epochs):
            Z = X @ self.weights + self.bias
            A = self._sigmoid(Z)

            err = A - y.flatten()

            self.weights -= self.lr * (X.T @ err) / n
            self.bias -= self.lr * np.mean(err)

            train_pred = (A >= 0.5).astype(int)

            self.history["train_loss"].append(
                -np.mean(
                    y.flatten() * np.log(A + 1e-15) +
                    (1 - y.flatten()) * np.log(1 - A + 1e-15)
                )
            )
            self.history["train_acc"].append(
                np.mean(train_pred == y.flatten())
            )

            if X_val is not None:
                val_A = self._sigmoid(X_val @ self.weights + self.bias)
                val_pred = (val_A >= 0.5).astype(int)

                self.history["val_loss"].append(
                    -np.mean(
                        y_val.flatten() * np.log(val_A + 1e-15) +
                        (1 - y_val.flatten()) * np.log(1 - val_A + 1e-15)
                    )
                )
                self.history["val_acc"].append(
                    np.mean(val_pred == y_val.flatten())
                )
            else:
                self.history["val_loss"].append(None)
                self.history["val_acc"].append(None)

        return self.history

    def predict(self, X):
        return self._sigmoid(X @ self.weights + self.bias).reshape(-1, 1)