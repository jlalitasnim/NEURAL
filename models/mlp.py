import numpy as np
from utils.activation import get_activation


class MLP:
    """
    Multi-Layer Perceptron (from scratch)
    Supports:
    - Forward propagation
    - Backpropagation
    - Multiple hidden layers
    """

    def __init__(self, layer_sizes, activations, l2_lambda=0.0, dropout_rate=0.0):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        self.weights = []
        self.biases = []

        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]

            if self.activations[i] == "relu":
                W = np.random.randn(fan_in, self.layer_sizes[i + 1]) * np.sqrt(2 / fan_in)
            else:
                W = np.random.randn(fan_in, self.layer_sizes[i + 1]) * 0.01

            b = np.zeros((1, self.layer_sizes[i + 1]))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X, training=False):
        A = X
        cache = []

        for i in range(len(self.weights)):
            Z = A @ self.weights[i] + self.biases[i]
            act_fn, _ = get_activation(self.activations[i])
            A_next = act_fn(Z)

            cache.append((A, Z))
            A = A_next

        return A, cache

    def backward(self, A_out, y_true, cache, loss_grad_fn):
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        dA = loss_grad_fn(y_true, A_out)

        for i in reversed(range(len(self.weights))):
            A_prev, Z = cache[i]
            _, act_deriv = get_activation(self.activations[i])

            if i == len(self.weights) - 1:
                dZ = dA
            else:
                dZ = dA * act_deriv(Z)

            grads_W[i] = A_prev.T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            dA = dZ @ self.weights[i].T

        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

    def predict(self, X):
        A, _ = self.forward(X)
        return A

    def get_weights_copy(self):
        return (
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases]
        )

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases