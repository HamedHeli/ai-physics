"""Ground truth for the 2D FNO notebook: 2D viscous Burgers' equation.

A 2D velocity field ``(u, v)`` on the periodic unit square, pushed around by
itself and smoothed by viscosity:

    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    v_t + u v_x + v v_y = nu (v_xx + v_yy),      (x, y) in [0, 1)^2 periodic

It is the 2D version of the 1D benchmark in ``physics.py``: the simplest flow
that has two velocity components at every point. (The FNO paper's 2D flow is
Navier-Stokes in vorticity form, which has one scalar per point; Burgers keeps
the "2 velocities + 2 positions per point" picture.)

The operator to learn maps the initial velocity field to the field at ``t_final``:

    G: (u0, v0)(x, y)  ->  (u, v)(x, y, t_final)

Initial conditions: each component is a Gaussian random field with Fourier
coefficients of standard deviation ``(|k|^2 + tau^2)^(-alpha)``, zero mean,
rescaled so the field has a fixed RMS. Solver: pseudo-spectral (``rfft2``
derivatives, 2/3-rule dealiasing) with integrating-factor RK4, like ``physics.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class Burgers2DParams(NamedTuple):
    """Physical and numerical constants.

    Attributes:
        nu: Viscosity.
        t_final: Time at which the solution is read off.
        n: Grid points per side.
        dt: Time step.
        rms: RMS speed of the initial field.
    """

    nu: float = 0.02
    t_final: float = 0.5
    n: int = 64
    dt: float = 2e-3
    rms: float = 0.5


def wavenumbers(n: int) -> tuple[Array, Array]:
    """``(kx, ky)`` grids matching ``rfft2`` of an ``(n, n)`` array indexed ``[x, y]``."""
    kx = jnp.fft.fftfreq(n, 1.0 / n)
    ky = jnp.arange(n // 2 + 1)
    return jnp.meshgrid(kx, ky, indexing="ij")


def sample_initial_condition(key: Array, params: Burgers2DParams = Burgers2DParams(),
                             tau: float = 3.0, alpha: float = 2.0) -> Array:
    """One random initial velocity field, ``(n, n, 2)`` indexed ``[x, y, component]``."""
    n = params.n
    KX, KY = wavenumbers(n)
    std = (KX**2 + KY**2 + tau**2) ** (-alpha)
    std = std.at[0, 0].set(0.0)
    keys = jax.random.split(key, 4)
    comps = []
    for kr, ki in (keys[:2], keys[2:]):
        c = std * (jax.random.normal(kr, std.shape) + 1j * jax.random.normal(ki, std.shape))
        f = jnp.fft.irfft2(c, s=(n, n))
        comps.append(f / jnp.sqrt(jnp.mean(f**2)) * params.rms)
    return jnp.stack(comps, axis=-1)


def solve(w0: Array, params: Burgers2DParams = Burgers2DParams()) -> Array:
    """Integrate 2D Burgers from ``w0 = (u0, v0)`` of shape ``(n, n, 2)`` to ``t_final``."""
    n = w0.shape[0]
    KX, KY = wavenumbers(n)
    ikx, iky = (2j * jnp.pi * KX)[..., None], (2j * jnp.pi * KY)[..., None]
    dealias = ((jnp.abs(KX) < n / 3) & (KY < n / 3))[..., None]
    lap = -(2 * jnp.pi) ** 2 * (KX**2 + KY**2)[..., None]
    decay = jnp.exp(params.nu * lap * params.dt / 2)
    fwd = lambda f: jnp.fft.rfft2(f, axes=(0, 1))
    inv = lambda f: jnp.fft.irfft2(f, s=(n, n), axes=(0, 1))

    def advection(w_hat):
        w = inv(w_hat)
        wx, wy = inv(ikx * w_hat), inv(iky * w_hat)
        return -fwd(w[..., :1] * wx + w[..., 1:] * wy) * dealias

    def rk4_step(w_hat, _):
        a = advection(w_hat)
        b = advection(decay * (w_hat + 0.5 * params.dt * a))
        c = advection(decay * w_hat + 0.5 * params.dt * b)
        d = advection(decay**2 * w_hat + params.dt * decay * c)
        return decay**2 * w_hat + params.dt / 6 * (decay**2 * a + 2 * decay * (b + c) + d), None

    n_steps = int(round(params.t_final / params.dt))
    w_hat, _ = jax.lax.scan(rk4_step, fwd(w0), None, length=n_steps)
    return inv(w_hat)


def make_dataset(key: Array, n_samples: int, params: Burgers2DParams = Burgers2DParams()) -> dict[str, Array]:
    """Pairs ``{"a": (S, n, n, 2), "u": (S, n, n, 2)}`` of initial and final velocity fields."""
    keys = jax.random.split(key, n_samples)
    a = jax.vmap(lambda k: sample_initial_condition(k, params))(keys)
    u = jax.lax.map(jax.jit(lambda w: solve(w, params)), a, batch_size=50)
    return {"a": a, "u": u}
