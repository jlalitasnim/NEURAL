import numpy as np
from utils.activation import get_activation


class MLP:
    """
    Multi-Layer Perceptron implemented from scratch.

    Supports:
    - Forward propagation
    - Backpropagation
    - Multiple hidden layers
    - L2 regularization
    - Dropout
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
            fan_out = self.layer_sizes[i + 1]

            if self.activations[i] == "relu":
                W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
            else:
                W = np.random.randn(fan_in, fan_out) * 0.01

            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X, training=False):
        A = X
        cache = []

        for i in range(len(self.weights)):
            Z = A @ self.weights[i] + self.biases[i]

            act_fn, _ = get_activation(self.activations[i])
            A_next = act_fn(Z)

            dropout_mask = None
            is_hidden_layer = i < len(self.weights) - 1

            # Dropout is applied only during training and only on hidden layers
            if training and is_hidden_layer and self.dropout_rate > 0:
                dropout_mask = (
                    np.random.rand(*A_next.shape) > self.dropout_rate
                ).astype(float)

                # Inverted dropout
                A_next = (A_next * dropout_mask) / (1 - self.dropout_rate)

            cache.append((A, Z, dropout_mask))

            A = A_next

        return A, cache

    def backward(self, A_out, y_true, cache, loss_grad_fn):
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        dA = loss_grad_fn(y_true, A_out)

        for i in reversed(range(len(self.weights))):
            A_prev, Z, dropout_mask = cache[i]

            _, act_deriv = get_activation(self.activations[i])

            is_output_layer = i == len(self.weights) - 1

            if is_output_layer:
                dZ = dA
            else:
                # Apply dropout mask during backpropagation
                if dropout_mask is not None:
                    dA = (dA * dropout_mask) / (1 - self.dropout_rate)

                dZ = dA * act_deriv(Z)

            grads_W[i] = A_prev.T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            # L2 regularization: add penalty only to weights, not biases
            if self.l2_lambda > 0:
                grads_W[i] += self.l2_lambda * self.weights[i]

            dA = dZ @ self.weights[i].T

        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

    def predict(self, X):
        A, _ = self.forward(X, training=False)
        return A

    def get_weights_copy(self):
        return (
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases]
        )

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases