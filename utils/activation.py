import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))


def sigmoid_deriv(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def relu(Z):
    return np.maximum(0, Z)


def relu_deriv(Z):
    return (Z > 0).astype(float)


def tanh(Z):
    return np.tanh(Z)


def tanh_deriv(Z):
    return 1 - np.tanh(Z) ** 2


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def softmax_deriv(Z):
    return np.ones_like(Z)


def linear(Z):
    return Z


def linear_deriv(Z):
    return np.ones_like(Z)


def get_activation(name):
    activations = {
        "sigmoid": (sigmoid, sigmoid_deriv),
        "relu": (relu, relu_deriv),
        "tanh": (tanh, tanh_deriv),
        "softmax": (softmax, softmax_deriv),
        "linear": (linear, linear_deriv),
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")

    return activations[name]