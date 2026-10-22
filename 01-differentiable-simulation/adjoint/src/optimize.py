"""Gradient descent on the design parameters using adjoint gradients."""

from __future__ import annotations

import numpy as np

from .adjoint import discrete_adjoint_gradient
from .ball import Problem, objective, simulate


def gradient_descent(v0, U, prob: Problem, n_iters: int = 60, lr: float = 0.05, optimize_thrust: bool = True):
    """Plain gradient descent, keeping every iterate for plotting.

    Returns:
        List of dicts ``{"v0", "U", "J", "X"}``, one per iteration
        (index 0 = the initial guess).
    """
    v0, U = np.array(v0, float), np.array(U, float)
    history = []
    for _ in range(n_iters + 1):
        X, _ = simulate(v0, U, prob)
        history.append({"v0": v0.copy(), "U": U.copy(), "J": objective(v0, U, prob), "X": X})
        gv, gU = discrete_adjoint_gradient(v0, U, prob)
        v0 -= lr * gv
        if optimize_thrust:
            U -= lr * gU
    return history
