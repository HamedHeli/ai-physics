"""Ground truth for the FNO notebook: the 1D viscous Burgers' equation.

This is the 1D benchmark of the FNO paper (Li et al., 2021, Sec. 5.1):

    u_t + (u^2 / 2)_x = nu * u_xx,      x in [0, 1) periodic,  t in (0, 1]

The operator to learn maps the initial condition to the solution at ``t = 1``:

    G: u0(x)  ->  u(x, t=1)

Initial conditions are drawn from the Gaussian random field used in the paper,
``mu = N(0, 625 (-Laplacian + 25 I)^-2)``. In Fourier space each coefficient of
``u0`` is independent with standard deviation ``25 / (4 pi^2 k^2 + 25)``, so the
field is smooth: most of its energy sits in the first handful of modes.

Solver
------
Pseudo-spectral in space (FFT derivatives, 2/3-rule dealiasing of the
nonlinear term) and integrating-factor RK4 in time: the diffusion term
``-nu (2 pi k)^2`` is integrated exactly, only the advection term is stepped.
Everything is pure JAX, so the solver itself is differentiable (Section 9 of the
notebook uses that to compare with gradients through the FNO).

The paper uses ``nu = 0.1`` and a reference resolution of 8192 points. Here
``nu = 0.02``: with ``nu = 0.1`` the solution at ``t = 1`` has decayed to about 2%
of its initial size and is ~93% explained by linear diffusion alone, which makes
a dull demo. At ``nu = 0.02`` the advection term matters (the answer differs from
pure diffusion by ~70%) while the solution stays smooth enough for 32 grid
points. The reference resolution is 256 and the FNO is trained on every 8th
point (32).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class BurgersParams(NamedTuple):
    """Physical and numerical constants.

    Attributes:
        nu: Viscosity.
        t_final: Time at which the solution is read off.
        n_fine: Grid points of the reference solver.
        dt: Time step of the reference solver.
    """

    nu: float = 0.02
    t_final: float = 1.0
    n_fine: int = 256
    dt: float = 1e-3


def grid(n: int) -> Array:
    """``n`` uniformly spaced points ``x_j = j / n`` on the periodic unit interval."""
    return jnp.arange(n) / n


def sample_initial_condition(key: Array, n: int) -> Array:
    """One draw of ``u0 ~ N(0, 625 (-Laplacian + 25 I)^-2)`` on ``n`` points.

    The field is built directly from its Fourier coefficients
    ``c_k ~ (25 / (4 pi^2 k^2 + 25)) * complex normal`` for ``k = 1 .. n/2``,
    with ``c_0 = 0`` (zero mean).
    """
    k = jnp.arange(n // 2 + 1)
    std = 25.0 / (4 * jnp.pi**2 * k**2 + 25.0)
    kr, ki = jax.random.split(key)
    c = std * (jax.random.normal(kr, k.shape) + 1j * jax.random.normal(ki, k.shape)) / jnp.sqrt(2)
    c = c.at[0].set(0.0)
    # norm="forward": c_k are the actual amplitudes, u(x) = sum_k c_k e^{2 pi i k x} + c.c.
    return jnp.fft.irfft(c, n=n, norm="forward")


def solve(u0: Array, params: BurgersParams = BurgersParams()) -> Array:
    """Integrate Burgers from ``u0`` to ``t_final`` and return ``u(., t_final)``.

    Args:
        u0: Initial condition on a uniform periodic grid, shape ``(n,)``.

    Returns:
        Solution at ``t_final`` on the same grid, shape ``(n,)``.
    """
    n = u0.shape[0]
    k = jnp.arange(n // 2 + 1)
    ik = 2j * jnp.pi * k
    dealias = k < (2 * n) // 6                      # 2/3 rule on the quadratic term
    decay = jnp.exp(-params.nu * (2 * jnp.pi * k) ** 2 * params.dt / 2)   # exp(L dt/2)

    def advection(u_hat):
        u = jnp.fft.irfft(u_hat, n=n)
        return -0.5 * ik * jnp.fft.rfft(u * u) * dealias

    def rk4_step(u_hat, _):
        a = advection(u_hat)
        b = advection(decay * (u_hat + 0.5 * params.dt * a))
        c = advection(decay * u_hat + 0.5 * params.dt * b)
        d = advection(decay**2 * u_hat + params.dt * decay * c)
        u_hat = decay**2 * u_hat + params.dt / 6 * (decay**2 * a + 2 * decay * (b + c) + d)
        return u_hat, None

    n_steps = int(round(params.t_final / params.dt))
    u_hat, _ = jax.lax.scan(rk4_step, jnp.fft.rfft(u0), None, length=n_steps)
    return jnp.fft.irfft(u_hat, n=n)


def make_dataset(key: Array, n_samples: int, params: BurgersParams = BurgersParams()) -> dict[str, Array]:
    """Input/output pairs ``(u0, u(t=1))`` at the reference resolution.

    Returns:
        ``{"a": (n_samples, n_fine), "u": (n_samples, n_fine)}``. Subsample with
        ``[:, ::s]`` to get coarser grids of the very same functions.
    """
    keys = jax.random.split(key, n_samples)
    a = jax.vmap(lambda k: sample_initial_condition(k, params.n_fine))(keys)
    u = jax.jit(jax.vmap(lambda u0: solve(u0, params)))(a)
    return {"a": a, "u": u}
