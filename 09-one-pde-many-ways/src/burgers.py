"""The PDE used by every method: 1D viscous Burgers on a periodic domain.

The equation
------------
::

    u_t + u u_x = nu u_xx,        x in [0, 2 pi) periodic,   t in [0, T]

``u(x, t)`` is a velocity. The two terms on the right of ``u_t`` are the two
things every fluid does:

* ``u u_x`` (advection): each point of the fluid carries its own velocity
  along with it. Fast parts catch up with slow parts, so a smooth bump
  steepens into a front, the 1D cousin of a shock wave (and of a traffic jam
  forming behind slow cars).
* ``nu u_xx`` (viscosity): momentum diffuses, which smooths the front. With
  ``nu = 0.1`` the front stays a few grid cells wide on a 64-point grid.

It is the simplest PDE that is nonlinear, has a physical meaning (it is the
momentum equation of Navier-Stokes with the pressure removed) and is cheap
enough to solve thousands of times on a laptop. Its linear cousin, the heat
equation, would make several of the methods trivially exact.

The classical solver
--------------------
Pseudo-spectral in space, integrating-factor RK4 in time (IFRK4):

* Space: ``u`` is stored on ``N`` grid points and moved to Fourier space with
  the FFT. Derivatives become multiplications, ``d/dx -> i k``. The
  nonlinear term is written in conservative form ``u u_x = (u^2 / 2)_x`` and
  the top third of the modes is zeroed (the 2/3 rule) so the product does not
  alias.
* Time: viscosity is stiff (``nu k^2`` is large for high ``k``), so it is
  solved exactly with the integrating factor ``exp(-nu k^2 t)`` and only the
  advection term goes through classical RK4.

Everything is written in ``jax.numpy`` with ``lax.scan``, so the whole solver
is differentiable with respect to ``nu`` and to the initial condition. That is
all :mod:`.inverse` needs to run differentiable-simulation inverse problems.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class BurgersConfig(NamedTuple):
    """The physical problem shared by every method.

    Attributes:
        n: Grid points used by the learned models.
        nu: Viscosity.
        t_final: The end time ``T``; operator models learn ``u(., 0) -> u(., T)``.
        n_snapshots: Number of saved states after ``t = 0``, evenly spaced up to
            ``T`` (trajectory methods train on these).
        dt: Internal time step of the reference solver.
        n_modes_ic: Initial conditions use Fourier modes ``1..n_modes_ic``.
        amplitude: Scale of the initial conditions.
        n_ref: Grid of the reference solver. The "ground truth" is solved on
            this finer grid and subsampled to ``n`` points, so even the
            classical solver on ``n`` points has a small error to report.
    """

    n: int = 64
    nu: float = 0.1
    t_final: float = 1.0
    n_snapshots: int = 20
    dt: float = 1e-3
    n_modes_ic: int = 4
    amplitude: float = 0.75
    n_ref: int = 256


def grid(n: int) -> Array:
    """``n`` equispaced points on ``[0, 2 pi)``."""
    return jnp.arange(n) * (2 * jnp.pi / n)


def _wavenumbers(n: int) -> tuple[Array, Array]:
    k = jnp.fft.rfftfreq(n, d=1.0 / n)  # 0, 1, ..., n/2
    dealias = (k < n / 3).astype(jnp.float32)  # 2/3 rule
    return k, dealias


def _advection_hat(u_hat: Array, k: Array, dealias: Array, n: int) -> Array:
    """Fourier transform of ``-(u^2/2)_x``, dealiased."""
    u = jnp.fft.irfft(u_hat * dealias, n=n)
    return -0.5j * k * jnp.fft.rfft(u * u) * dealias


@partial(jax.jit, static_argnames=("n_steps", "save_every"))
def solve(u0: Array, nu: float | Array, dt: float, n_steps: int, save_every: int) -> Array:
    """Integrate Burgers from ``u0`` with IFRK4.

    Args:
        u0: Initial condition, shape ``(n,)``.
        nu: Viscosity (a JAX scalar is fine; the solver is differentiable in it).
        dt: Time step.
        n_steps: Total number of steps.
        save_every: Save the state every this many steps.

    Returns:
        States of shape ``(n_steps // save_every + 1, n)``, starting with ``u0``.
    """
    n = u0.shape[-1]
    k, dealias = _wavenumbers(n)
    e = jnp.exp(-nu * k**2 * dt)
    e2 = jnp.exp(-nu * k**2 * dt / 2)
    nl = partial(_advection_hat, k=k, dealias=dealias, n=n)

    def step(v, _):
        k1 = dt * nl(v)
        k2 = dt * nl(e2 * (v + k1 / 2))
        k3 = dt * nl(e2 * v + k2 / 2)
        k4 = dt * nl(e * v + e2 * k3)
        return e * v + (e * k1 + 2 * e2 * (k2 + k3) + k4) / 6, None

    def save_block(v, _):
        v, _ = jax.lax.scan(step, v, None, length=save_every)
        return v, jnp.fft.irfft(v, n=n)

    v0 = jnp.fft.rfft(u0)
    _, saved = jax.lax.scan(save_block, v0, None, length=n_steps // save_every)
    return jnp.concatenate([u0[None], saved], axis=0)


def solve_config(u0: Array, cfg: BurgersConfig, nu: float | Array | None = None) -> Array:
    """:func:`solve` with the time stepping taken from ``cfg``.

    Returns ``(n_snapshots + 1, n)``: ``u0`` and the ``n_snapshots`` saved states.
    """
    n_steps = int(round(cfg.t_final / cfg.dt))
    save_every = n_steps // cfg.n_snapshots
    return solve(u0, cfg.nu if nu is None else nu, cfg.dt, n_steps, save_every)


def random_initial_conditions(key: Array, n_samples: int, cfg: BurgersConfig, n: int | None = None) -> Array:
    """Smooth random periodic initial conditions.

    ``u0(x) = A sum_{k=1..K} (a_k sin kx + b_k cos kx) / k`` with ``a_k, b_k ~ N(0, 1)``,
    so long waves dominate. ``u0`` has zero mean (no net drift) and amplitude
    of order one, enough for the fronts to steepen before ``T = 1``.
    """
    n = cfg.n if n is None else n
    x = grid(n)
    kk = jnp.arange(1, cfg.n_modes_ic + 1)
    ka, kb = jax.random.split(key)
    a = cfg.amplitude * jax.random.normal(ka, (n_samples, cfg.n_modes_ic)) / kk
    b = cfg.amplitude * jax.random.normal(kb, (n_samples, cfg.n_modes_ic)) / kk
    phase = kk[None, :] * x[:, None]  # (n, K)
    return a @ jnp.sin(phase).T + b @ jnp.cos(phase).T


class Dataset(NamedTuple):
    """Trajectories from the reference solver.

    Attributes:
        u: States, shape ``(n_samples, n_snapshots + 1, n)``.
        t: Snapshot times, shape ``(n_snapshots + 1,)``.
        x: Grid, shape ``(n,)``.
    """

    u: Array
    t: Array
    x: Array


def make_dataset(key: Array, n_samples: int, cfg: BurgersConfig, n: int | None = None) -> Dataset:
    """Solve Burgers on ``cfg.n_ref`` points from ``n_samples`` random initial
    conditions and subsample the result to ``n`` points (``n`` must divide
    ``cfg.n_ref``)."""
    n = cfg.n if n is None else n
    u0 = random_initial_conditions(key, n_samples, cfg, cfg.n_ref)
    u = jax.lax.map(lambda a: solve_config(a, cfg), u0, batch_size=64)[..., :: cfg.n_ref // n]
    t = jnp.linspace(0.0, cfg.t_final, cfg.n_snapshots + 1)
    return Dataset(u=u, t=t, x=grid(n))


def relative_l2(pred: Array, true: Array) -> Array:
    """Mean over samples of ``||pred - true|| / ||true||`` (last axis is space)."""
    num = jnp.linalg.norm(pred - true, axis=-1)
    den = jnp.linalg.norm(true, axis=-1)
    return jnp.mean(num / den)


def finite_difference_solve(u0: np.ndarray, nu: float, t_final: float, dt: float) -> np.ndarray:
    """A textbook comparison point: second-order central differences + RK2.

    Not used for the data; it shows what a cheaper classical scheme on the
    same 64-point grid gets wrong (dispersion near the fronts).
    """
    n = u0.shape[-1]
    dx = 2 * np.pi / n

    def rhs(u):
        up, um = np.roll(u, -1, -1), np.roll(u, 1, -1)
        return -(up**2 - um**2) / (4 * dx) + nu * (up - 2 * u + um) / dx**2

    u = np.array(u0, dtype=np.float64)
    for _ in range(int(round(t_final / dt))):
        u_half = u + 0.5 * dt * rhs(u)
        u = u + dt * rhs(u_half)
    return u
