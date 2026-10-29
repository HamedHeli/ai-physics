"""Method 3 (topic 03): a reduced-order model by POD-Galerkin projection.

What is learned: **a low-dimensional linear subspace** of solutions. The
dynamics are not learned; they are the exact PDE, projected.

1. **Basis (offline, from data).** Stack every training snapshot as a column of
   ``S`` (``n x n_snapshots``) and take its SVD. The first ``r`` left singular
   vectors ``Phi = [phi_1 .. phi_r]`` (the POD modes) are the best
   ``r``-dimensional linear subspace for those snapshots in the least-squares
   sense. CROM [Chen+ 2023] replaces this linear ``Phi a`` with a neural-field
   decoder ``g(a, x)``; Sharp+ 2023 learn the map without data.

2. **Dynamics (Galerkin projection).** Write ``u ~ Phi a`` and project the PDE
   onto the modes:

   ::

       da/dt = Phi^T f(Phi a),    f(u) = -u u_x + nu u_xx

   Burgers is quadratic in ``u``, so this collapses to a closed system in ``r``
   numbers with tensors precomputed once:

   ::

       da_i/dt = sum_j L_ij a_j - sum_jk Q_ijk a_j a_k
       L = nu Phi^T D2 Phi,      Q_ijk = phi_i . (phi_j * D phi_k)

   Online cost is ``O(r^3)`` per step, independent of the grid size ``n``.

The modes were fit to *training* snapshots, so on a test IC the error has two
parts: the projection error (the test solution is not exactly in the
subspace) and the Galerkin error (the truncated dynamics drift, because the
discarded modes are what dissipate energy in Burgers).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class PODGalerkin(NamedTuple):
    phi: Array  # (n, r) modes
    sing: Array  # all singular values (for the spectrum plot)
    lin: Array  # (r, r)
    quad: Array  # (r, r, r)


def spectral_derivative(u: Array, order: int) -> Array:
    """``d^order u / dx^order`` on a periodic grid, along the last axis."""
    n = u.shape[-1]
    k = jnp.fft.rfftfreq(n, d=1.0 / n)
    return jnp.fft.irfft((1j * k) ** order * jnp.fft.rfft(u, axis=-1), n=n, axis=-1)


def fit(snapshots: Array, r: int, nu: float) -> PODGalerkin:
    """Build the POD basis and the Galerkin tensors.

    Args:
        snapshots: Training states, any shape ``(..., n)``.
        r: Number of modes.
        nu: Viscosity.
    """
    s = snapshots.reshape(-1, snapshots.shape[-1]).T  # (n, m)
    u_svd, sing, _ = jnp.linalg.svd(s, full_matrices=False)
    phi = u_svd[:, :r]
    dphi = spectral_derivative(phi.T, 1).T
    d2phi = spectral_derivative(phi.T, 2).T
    lin = nu * phi.T @ d2phi
    quad = jnp.einsum("xi,xj,xk->ijk", phi, phi, dphi)
    return PODGalerkin(phi, sing, lin, quad)


def encode(rom: PODGalerkin, u: Array) -> Array:
    return u @ rom.phi


def decode(rom: PODGalerkin, a: Array) -> Array:
    return a @ rom.phi.T


def rhs(rom: PODGalerkin, a: Array) -> Array:
    return rom.lin @ a - jnp.einsum("ijk,j,k->i", rom.quad, a, a)


def rollout(rom: PODGalerkin, u0: Array, dt: float, n_steps: int, save_every: int) -> Array:
    """Integrate the reduced system with RK4 from ``encode(u0)``; return
    decoded states ``(n_steps // save_every + 1, n)``."""

    def step(a, _):
        k1 = rhs(rom, a)
        k2 = rhs(rom, a + dt / 2 * k1)
        k3 = rhs(rom, a + dt / 2 * k2)
        k4 = rhs(rom, a + dt * k3)
        return a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

    def block(a, _):
        a, _ = jax.lax.scan(step, a, None, length=save_every)
        return a, a

    a0 = encode(rom, u0)
    _, saved = jax.lax.scan(block, a0, None, length=n_steps // save_every)
    return decode(rom, jnp.concatenate([a0[None], saved]))
