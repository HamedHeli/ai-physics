"""Method 4 (topic 04): a latent neural ODE [Chen+ 2018; Wiewel+ 2019].

What is learned: **the time derivative** in a reduced space,
``dz/dt = f_theta(z)``. The space comes from :mod:`.rom` (``z`` = POD
coefficients, scaled to unit variance); the dynamics no longer come from the
PDE but from data. This is the "latent-space physics" recipe of Wiewel+ 2019
with a POD encoder in place of their autoencoder.

Training integrates the ODE with RK4 through all 20 snapshot intervals and
compares every saved state with the data. ``jax.grad`` backpropagates through
the unrolled solver: the discrete adjoint, the same thing
``01-differentiable-simulation/adjoint`` derives by hand. Chen+ 2018 instead
solve the continuous adjoint ODE backwards to save memory; at this size
storing 100 small states is free.

Compared with POD-Galerkin at the same ``r``, the network can learn a
*closure*: the effect of the discarded modes on the kept ones, which the
projected equations miss.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .nn import MLPParams, init_mlp, mlp


class LatentODE(NamedTuple):
    phi: Array  # (n, r) POD basis (fixed)
    scale: Array  # (r,) one shared scale, so the loss weighs modes by their energy


def make_latent(phi: Array, train_u: Array) -> LatentODE:
    z = train_u @ phi
    return LatentODE(phi, jnp.full(phi.shape[1], z.std()))


def init_params(key: Array, r: int, hidden: int = 128) -> MLPParams:
    return init_mlp(key, [r, hidden, hidden, r])


def odeint(params: MLPParams, z0: Array, dt: float, substeps: int, n_save: int) -> Array:
    """RK4 for ``dz/dt = mlp(z)``; returns ``(n_save + 1, r)`` including ``z0``."""
    f = lambda z: mlp(params, z, act=jnp.tanh)

    def step(z, _):
        k1 = f(z)
        k2 = f(z + dt / 2 * k1)
        k3 = f(z + dt / 2 * k2)
        k4 = f(z + dt * k3)
        return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

    def block(z, _):
        z, _ = jax.lax.scan(step, z, None, length=substeps)
        return z, z

    _, zs = jax.lax.scan(block, z0, None, length=n_save)
    return jnp.concatenate([z0[None], zs])


def predict(params: MLPParams, lat: LatentODE, u0: Array, t_final: float, n_save: int, substeps: int = 5) -> Array:
    """Encode ``u0``, integrate, decode: ``(n_save + 1, n)`` states."""
    z0 = (u0 @ lat.phi) / lat.scale
    zs = odeint(params, z0, t_final / n_save / substeps, substeps, n_save)
    return (zs * lat.scale) @ lat.phi.T


def make_loss(
    lat: LatentODE, train_u: Array, t_final: float, window: int = 5, batch: int = 64, substeps: int = 5
):
    """Mean squared error of rollouts over random windows of ``window``
    snapshot intervals, in scaled latent space.

    Short windows (multiple shooting) keep the gradients well-behaved early
    in training; :func:`predict` still rolls out the full trajectory.
    """
    z = (train_u @ lat.phi) / lat.scale  # (samples, T+1, r)
    n_save = z.shape[1] - 1
    dt = t_final / n_save / substeps
    starts = n_save - window + 1

    def loss(params, key):
        ki, ks = jax.random.split(key)
        idx = jax.random.randint(ki, (batch,), 0, z.shape[0])
        t0 = jax.random.randint(ks, (batch,), 0, starts)
        zb = jax.vmap(lambda i, s: jax.lax.dynamic_slice_in_dim(z[i], s, window + 1))(idx, t0)
        pred = jax.vmap(lambda z0: odeint(params, z0, dt, substeps, window))(zb[:, 0])
        return jnp.mean((pred - zb) ** 2)

    return loss
