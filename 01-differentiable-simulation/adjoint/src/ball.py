"""Forward simulator: a ball thrown through air with drag and an optional thrust.

State and dynamics
------------------
The state is ``s = (x, v)`` with position ``x`` and velocity ``v`` in 2D:

    dx/dt = v
    dv/dt = g + u(t) - c |v| v

* ``g = (0, -9.81)`` is gravity.
* ``c`` is a quadratic drag coefficient. With ``c = 0`` this is the textbook
  projectile, whose step Jacobian is a constant matrix; with ``c > 0`` the
  Jacobian depends on the velocity, which is what forces an adjoint to store
  (or recompute) the forward trajectory.
* ``u(t)`` is a piecewise-constant thrust with ``K`` segments. It exists to
  give the problem *many* design parameters, so the cost of finite
  differences (one simulation per parameter) can be compared with the cost of
  the adjoint (one backward pass for all of them).

Time stepping is explicit Euler with ``N`` steps of size ``dt``:

    x_{k+1} = x_k + dt v_k
    v_{k+1} = v_k + dt (g + u_k - c |v_k| v_k)

Design parameters and objective
-------------------------------
``theta = (v0, U)``: the launch velocity ``v0`` (2 numbers) and the thrust
table ``U`` of shape ``(K, 2)``. The ball starts at the origin. The objective
is to land on a target point at the final time:

    J(theta) = |x_N - x_target|^2 + reg * mean |U|^2

Everything here is plain NumPy so that the adjoint in :mod:`src.adjoint` is
visibly hand-written; :mod:`src.adjoint` checks it against ``jax.grad``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

GRAVITY = np.array([0.0, -9.81])


class Problem(NamedTuple):
    """Fixed (non-design) settings.

    Attributes:
        n_steps: Number of time steps ``N``.
        dt: Time step.
        drag: Quadratic drag coefficient ``c``.
        n_segments: Number of thrust segments ``K`` (``N`` must divide evenly).
        target: Target landing point, shape ``(2,)``.
        reg: Weight of the thrust penalty in ``J``.
    """

    n_steps: int = 200
    dt: float = 0.01
    drag: float = 0.1
    n_segments: int = 10
    target: np.ndarray = np.array([8.0, 0.0])
    reg: float = 0.0


def thrust_at(U: np.ndarray, k: int, prob: Problem) -> np.ndarray:
    """Thrust applied at step ``k`` (piecewise constant)."""
    return U[k * prob.n_segments // prob.n_steps]


def step(x: np.ndarray, v: np.ndarray, u: np.ndarray, prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    """One explicit Euler step."""
    speed = np.linalg.norm(v)
    return x + prob.dt * v, v + prob.dt * (GRAVITY + u - prob.drag * speed * v)


def simulate(v0: np.ndarray, U: np.ndarray, prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Run the forward model and keep every state.

    Returns:
        ``(X, V)``, each of shape ``(N + 1, 2)``; row ``k`` is the state at step ``k``.
    """
    X = np.zeros((prob.n_steps + 1, 2))
    V = np.zeros((prob.n_steps + 1, 2))
    V[0] = v0
    for k in range(prob.n_steps):
        X[k + 1], V[k + 1] = step(X[k], V[k], thrust_at(U, k, prob), prob)
    return X, V


def objective(v0: np.ndarray, U: np.ndarray, prob: Problem) -> float:
    """``J = |x_N - target|^2 + reg * mean |U|^2``."""
    X, _ = simulate(v0, U, prob)
    return float(np.sum((X[-1] - prob.target) ** 2) + prob.reg * np.mean(U**2))
