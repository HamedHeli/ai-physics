"""Method 2 (topic 02): a physics-informed neural network [Raissi+ 2019].

What is learned: **one solution** ``u(x, t)`` for one initial condition. The
network is the solution itself, a continuous function of ``(x, t)``; no
solver data is used, only the PDE.

Two design choices remove the boundary and initial terms from the loss, so the
only thing left to fit is the PDE residual:

* **Periodicity by construction.** The network sees ``(sin x, cos x, t)``
  instead of ``x``, so its output is ``2 pi``-periodic in ``x`` whatever the
  weights are.
* **Initial condition by construction (hard constraint).**
  ``u(x, t) = u0(x) + t * N(x, t)``. At ``t = 0`` the network term vanishes.
  ``u0`` is known on the grid; :func:`fourier_interpolant` turns it into a
  differentiable function of ``x`` (exact for our band-limited ICs).

Loss: ``mean over random (x, t) of (u_t + u u_x - nu u_xx)^2``. The
derivatives come from forward-mode autodiff through the network
(``jax.jvp``), not from a grid.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .nn import MLPParams, init_mlp, mlp


def fourier_interpolant(u_grid: Array) -> Callable[[Array], Array]:
    """The trigonometric interpolant of grid values as a function of ``x``."""
    n = u_grid.shape[0]
    c = jnp.fft.rfft(u_grid) / n
    k = jnp.arange(c.shape[0])
    w = jnp.where((k == 0) | (k == n // 2), 1.0, 2.0)

    def f(x):
        return jnp.sum(w * (c.real * jnp.cos(k * x) - c.imag * jnp.sin(k * x)))

    return f


def init_pinn(key: Array, hidden: int = 48, depth: int = 4) -> MLPParams:
    return init_mlp(key, [3] + [hidden] * depth + [1])


def make_solution(params: MLPParams, u0_fn: Callable[[Array], Array]) -> Callable[[Array, Array], Array]:
    """``u(x, t)`` for scalar ``x`` and ``t``."""

    def u(x, t):
        z = jnp.stack([jnp.sin(x), jnp.cos(x), t])
        return u0_fn(x) + t * mlp(params, z, act=jnp.tanh)[0]

    return u


def residual(params: MLPParams, u0_fn, nu: float, x: Array, t: Array) -> Array:
    """``u_t + u u_x - nu u_xx`` at scalar ``(x, t)``."""
    u = make_solution(params, u0_fn)
    u_x_fn = lambda x_: jax.jvp(lambda a: u(a, t), (x_,), (1.0,))[1]
    val, u_t = jax.jvp(lambda b: u(x, b), (t,), (1.0,))
    u_x, u_xx = jax.jvp(u_x_fn, (x,), (1.0,))
    return u_t + val * u_x - nu * u_xx


def make_loss(u0_grid: Array, nu: float, t_final: float, n_colloc: int = 1024):
    """Loss over ``n_colloc`` fresh random collocation points per step."""
    u0_fn = fourier_interpolant(u0_grid)

    def loss(params, key):
        kx, kt = jax.random.split(key)
        x = jax.random.uniform(kx, (n_colloc,), maxval=2 * jnp.pi)
        t = jax.random.uniform(kt, (n_colloc,), maxval=t_final)
        r = jax.vmap(residual, in_axes=(None, None, None, 0, 0))(params, u0_fn, nu, x, t)
        return jnp.mean(r**2)

    return loss


def evaluate(params: MLPParams, u0_grid: Array, x: Array, t: Array) -> Array:
    """The PINN on a ``(len(t), len(x))`` space-time grid."""
    u = make_solution(params, fourier_interpolant(u0_grid))
    return jax.vmap(lambda tt: jax.vmap(lambda xx: u(xx, tt))(x))(t)
