"""Method 1 (topic 01): differentiable simulation for inverse problems.

Nothing is learned here. The classical solver of :mod:`.burgers` is written in
JAX, so ``jax.grad`` runs the discrete adjoint through all 1000 RK4 steps
(DiffTaichi [Hu+ 2019], Warp [Macklin 2024], and the hand-written adjoint in
``01-differentiable-simulation/adjoint``). That turns two inverse problems into
plain gradient descent:

1. **System identification.** Observe ``u(x, T)`` and the initial condition,
   recover the viscosity ``nu`` (one unknown).
2. **Initial-state recovery.** Observe only ``u(x, T)``, recover ``u(x, 0)``
   (64 unknowns). Viscosity destroys high frequencies, so this is ill-posed:
   a small smoothness penalty picks the smooth answer.

The adjoint costs one backward pass whatever the number of unknowns, which is
why problem 2 is no more expensive per iteration than problem 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jax import Array

from .burgers import BurgersConfig, solve_config


def final_state(u0: Array, nu: Array, cfg: BurgersConfig) -> Array:
    return solve_config(u0, cfg, nu=nu)[-1]


def identify_viscosity(
    u0: Array, u_obs: Array, cfg: BurgersConfig, nu_init: float = 0.02, steps: int = 100, lr: float = 0.1
) -> tuple[list[float], list[float]]:
    """Recover ``nu`` from ``(u0, u(T))`` by Adam on ``log nu``.

    Optimizing ``log nu`` keeps ``nu`` positive and makes steps scale-free.

    Returns:
        ``(nus, losses)`` at every iteration.
    """

    def loss(log_nu):
        return jnp.mean((final_state(u0, jnp.exp(log_nu), cfg) - u_obs) ** 2)

    vg = jax.jit(jax.value_and_grad(loss))
    opt = optax.adam(lr)
    p = jnp.log(nu_init)
    state = opt.init(p)
    nus, losses = [], []
    for _ in range(steps):
        val, g = vg(p)
        nus.append(float(jnp.exp(p)))
        losses.append(float(val))
        upd, state = opt.update(g, state)
        p = optax.apply_updates(p, upd)
    return nus, losses


def viscosity_loss_curve(u0: Array, u_obs: Array, cfg: BurgersConfig, nus: Array) -> Array:
    """The misfit as a function of ``nu`` (for plotting the landscape)."""
    return jax.vmap(lambda nu: jnp.mean((final_state(u0, nu, cfg) - u_obs) ** 2))(nus)


def recover_initial_condition(
    u_obs: Array, cfg: BurgersConfig, steps: int = 400, lr: float = 0.1, smooth: float = 1e-5
) -> tuple[Array, list[float]]:
    """Recover ``u0`` from ``u(T)`` alone.

    Loss: ``mean (u(T; u0) - u_obs)^2 + smooth * mean (D u0)^2`` with ``D`` the
    forward difference. Starts from ``u0 = u_obs`` (the obvious first guess).

    Returns:
        ``(u0_estimate, losses)``.
    """

    def loss(u0):
        misfit = jnp.mean((final_state(u0, cfg.nu, cfg) - u_obs) ** 2)
        return misfit + smooth * jnp.mean((jnp.roll(u0, -1) - u0) ** 2) * cfg.n**2

    vg = jax.jit(jax.value_and_grad(loss))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.05))
    u0 = u_obs
    state = opt.init(u0)
    losses = []
    for _ in range(steps):
        val, g = vg(u0)
        losses.append(float(val))
        upd, state = opt.update(g, state)
        u0 = optax.apply_updates(u0, upd)
    return u0, losses
