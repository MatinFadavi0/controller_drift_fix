"""
model.py

A small multilayer perceptron written from scratch: forward pass,
backward pass (manual backprop), and gradient-descent updates, with no
ML framework involved. numpy is used ONLY as an array/linear-algebra
library (matrix multiply, elementwise ops) -- there is no autograd, no
optimizer object, no layer abstraction borrowed from a framework. Every
gradient below is derived and coded by hand.

Architecture: arbitrary number of hidden layers (tanh activation) and a
linear output layer, trained with full-batch gradient descent on MSE.
"""
import json
import numpy as np


def _tanh(x):
    return np.tanh(x)


def _tanh_grad_from_output(t):
    """d/dx tanh(x) expressed in terms of tanh(x) itself: 1 - tanh(x)^2."""
    return 1.0 - t ** 2


class Standardizer:
    """Feature scaling: (x - mean) / std, fit from training data and
    reused unchanged at inference time. Implemented manually -- this is
    just arithmetic, but it has to be saved/loaded consistently with the
    model or predictions will be wrong."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        # avoid divide-by-zero for any constant feature column
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.mean = np.array(d["mean"], dtype=np.float64)
        s.std = np.array(d["std"], dtype=np.float64)
        return s


class MLP:
    """
    Fully-connected feedforward network.

    layer_sizes: e.g. [6, 12, 2] -> 6 inputs, one hidden layer of 12
    tanh units, 2 linear outputs.

    Weights are initialized with a small random uniform range scaled by
    fan-in (a simple, hand-picked scheme -- not borrowed from a library
    default).
    """

    def __init__(self, layer_sizes, seed=42):
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = 1.0 / np.sqrt(fan_in)
            W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.W.append(W)
            self.b.append(b)

    def forward(self, X):
        """
        Returns (y_hat, cache) where cache holds every layer's
        pre-activation and activation, needed for backprop.
        """
        activations = [X]
        pre_activations = []
        a = X
        n_layers = len(self.W)
        for i in range(n_layers):
            z = a @ self.W[i] + self.b[i]
            pre_activations.append(z)
            if i < n_layers - 1:
                a = _tanh(z)
            else:
                a = z  # linear output layer
            activations.append(a)
        return a, {"activations": activations, "pre_activations": pre_activations}

    def predict(self, X):
        y_hat, _ = self.forward(X)
        return y_hat

    def train_step(self, X, Y, lr):
        """
        One full-batch gradient descent step. Returns the MSE loss
        BEFORE the update (so you can track a clean loss curve).
        """
        n = X.shape[0]
        y_hat, cache = self.forward(X)
        activations = cache["activations"]

        loss = np.mean((y_hat - Y) ** 2)

        n_layers = len(self.W)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # loss = mean over ALL elements of (y_hat - Y), i.e. over n * n_outputs
        # entries, so d(loss)/d(y_hat) = 2*(y_hat - Y) / (n * n_outputs), not
        # just n. Missing the n_outputs factor here is an easy off-by-2x bug
        # (caught by the numerical gradient check in test_model.py).
        n_outputs = Y.shape[1]
        dA = 2.0 * (y_hat - Y) / (n * n_outputs)

        for i in reversed(range(n_layers)):
            a_prev = activations[i]  # input to this layer
            if i == n_layers - 1:
                dZ = dA  # linear layer, no activation derivative
            else:
                a_curr = activations[i + 1]
                dZ = dA * _tanh_grad_from_output(a_curr)

            grads_W[i] = a_prev.T @ dZ
            grads_b[i] = dZ.sum(axis=0, keepdims=True)

            if i > 0:
                dA = dZ @ self.W[i].T

        for i in range(n_layers):
            self.W[i] -= lr * grads_W[i]
            self.b[i] -= lr * grads_b[i]

        return loss

    def to_dict(self):
        return {
            "layer_sizes": self.layer_sizes,
            "W": [w.tolist() for w in self.W],
            "b": [bb.tolist() for bb in self.b],
        }

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        m.layer_sizes = d["layer_sizes"]
        m.W = [np.array(w, dtype=np.float64) for w in d["W"]]
        m.b = [np.array(bb, dtype=np.float64) for bb in d["b"]]
        return m


def save_model(path, mlp, standardizer, meta=None):
    payload = {
        "mlp": mlp.to_dict(),
        "standardizer": standardizer.to_dict(),
        "meta": meta or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_model(path):
    with open(path) as f:
        payload = json.load(f)
    mlp = MLP.from_dict(payload["mlp"])
    standardizer = Standardizer.from_dict(payload["standardizer"])
    return mlp, standardizer, payload.get("meta", {})
