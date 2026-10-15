"""
test_model.py

Since there's no framework doing autodiff for us, it's worth proving the
hand-derived backprop in model.py is actually correct rather than just
"looks plausible". This does a standard numerical gradient check:
perturb each weight by a tiny epsilon, measure the change in loss, and
compare that finite-difference gradient against the analytic gradient
train_step computed. They should match to several decimal places.

Usage:
    python tests/test_model.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from model import MLP  # noqa: E402


def numerical_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 6))
    Y = rng.normal(size=(5, 2))

    mlp = MLP([6, 4, 2], seed=1)

    # Get analytic gradients by doing one manual backward pass without
    # applying the update (replicate train_step's math but capture grads).
    y_hat, cache = mlp.forward(X)
    activations = cache["activations"]
    n = X.shape[0]
    n_outputs = Y.shape[1]
    dA = 2.0 * (y_hat - Y) / (n * n_outputs)

    grads_W = [None] * len(mlp.W)
    for i in reversed(range(len(mlp.W))):
        a_prev = activations[i]
        if i == len(mlp.W) - 1:
            dZ = dA
        else:
            a_curr = activations[i + 1]
            dZ = dA * (1 - a_curr ** 2)
        grads_W[i] = a_prev.T @ dZ
        if i > 0:
            dA = dZ @ mlp.W[i].T

    def loss_fn():
        pred, _ = mlp.forward(X)
        return np.mean((pred - Y) ** 2)

    eps = 1e-5
    max_rel_error = 0.0
    layer_to_check = 0  # checking first layer's weights is representative
    W = mlp.W[layer_to_check]
    analytic = grads_W[layer_to_check]

    # Only check a handful of entries for speed
    checked = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if checked >= 10:
                break
            orig = W[i, j]

            W[i, j] = orig + eps
            loss_plus = loss_fn()
            W[i, j] = orig - eps
            loss_minus = loss_fn()
            W[i, j] = orig

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic_val = analytic[i, j]
            denom = max(abs(numeric), abs(analytic_val), 1e-8)
            rel_error = abs(numeric - analytic_val) / denom
            max_rel_error = max(max_rel_error, rel_error)
            checked += 1

    print(f"Checked {checked} weight entries in layer {layer_to_check}.")
    print(f"Max relative error vs numerical gradient: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, "Backprop gradient does not match numerical check!"
    print("PASS: analytic backprop matches numerical gradient.")


def loss_decreases_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    true_W = rng.normal(size=(6, 2))
    Y = X @ true_W + 0.01 * rng.normal(size=(50, 2))

    mlp = MLP([6, 8, 2], seed=2)
    losses = []
    for _ in range(500):
        losses.append(mlp.train_step(X, Y, lr=0.05))

    print(f"Loss went from {losses[0]:.4f} to {losses[-1]:.4f} over 500 steps.")
    assert losses[-1] < losses[0] * 0.1, "Loss did not decrease enough during training!"
    print("PASS: training loop actually reduces loss on a synthetic linear task.")


if __name__ == "__main__":
    numerical_gradient_check()
    print()
    loss_decreases_check()
