"""Ground-truth physics: a handful of colliding balls in 2D.

This is the "real simulator" that the GNN learns to imitate, playing the role
Taichi-MPM / SPH play in Sanchez-Gonzalez et al. (2020). It is deliberately
tiny (three balls by default) so that every matrix inside the GNN stays small
enough to print and look at.

Model
-----
Every ball has mass ``m`` and radius ``r``. Two balls ``i`` and ``j`` interact
only while they overlap, through a spring-dashpot contact force (the standard
discrete-element model):

    d_ij     = p_i - p_j,         dist = |d_ij|,       n_ij = d_ij / dist
    overlap  = max(0, 2r - dist)
    F_ij     = (k * overlap - c * (v_i - v_j) . n_ij) * n_ij    (if overlap > 0)

so ``F_ij = -F_ji`` (Newton's third law) and the force acts along the line of
centres. The acceleration of ball ``i`` is ``sum_j F_ij / m``. There is no
gravity and no walls: trajectories are short and centred on one collision
event.

Time integration is semi-implicit Euler, the same update GNS uses on top of
its predicted acceleration:

    v_{t+1} = v_t + dt * a_t
    p_{t+1} = p_t + dt * v_{t+1}
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class BallParams(NamedTuple):
    """Physical constants of the ground-truth simulator.

    Attributes:
        radius: Ball radius ``r``; balls touch when ``dist < 2r``.
        stiffness: Contact spring constant ``k``.
        damping: Contact dashpot constant ``c`` (makes collisions slightly
            inelastic, so the learned force has a velocity-dependent part).
        mass: Mass ``m`` of every ball.
        dt: Time step.
    """

    radius: float = 0.5
    stiffness: float = 200.0
    damping: float = 2.0
    mass: float = 1.0
    dt: float = 0.01


def contact_acceleration(pos: Array, vel: Array, params: BallParams) -> Array:
    """True per-ball acceleration from pairwise contact forces.

    Args:
        pos: Positions, shape ``(N, 2)``.
        vel: Velocities, shape ``(N, 2)``.
        params: Physical constants.

    Returns:
        Accelerations, shape ``(N, 2)``.
    """
    d = pos[:, None, :] - pos[None, :, :]  # (N, N, 2), d[i, j] = p_i - p_j
    dv = vel[:, None, :] - vel[None, :, :]
    n_balls = pos.shape[0]
    # Avoid 0/0 on the diagonal; those entries are masked out below anyway.
    dist = jnp.sqrt(jnp.sum(d**2, axis=-1) + jnp.eye(n_balls))
    normal = d / dist[..., None]
    overlap = jnp.maximum(0.0, 2.0 * params.radius - dist) * (1.0 - jnp.eye(n_balls))
    in_contact = (overlap > 0.0).astype(pos.dtype)
    normal_speed = jnp.sum(dv * normal, axis=-1)
    magnitude = params.stiffness * overlap - params.damping * normal_speed * in_contact
    force = jnp.sum(magnitude[..., None] * normal, axis=1)
    return force / params.mass


def step(pos: Array, vel: Array, params: BallParams) -> tuple[Array, Array, Array]:
    """Advance one semi-implicit Euler step.

    Returns:
        ``(pos_next, vel_next, acc)`` where ``acc`` is the acceleration that
        was applied, i.e. the GNN's training target for the input state.
    """
    acc = contact_acceleration(pos, vel, params)
    vel_next = vel + params.dt * acc
    pos_next = pos + params.dt * vel_next
    return pos_next, vel_next, acc


def rollout(pos0: Array, vel0: Array, n_steps: int, params: BallParams) -> dict[str, Array]:
    """Simulate ``n_steps`` steps from an initial state.

    Returns:
        Dict with ``pos``, ``vel`` and ``acc``, each of shape
        ``(n_steps, N, 2)``. ``acc[t]`` maps state ``t`` to state ``t + 1``.
    """

    def body(carry, _):
        pos, vel = carry
        pos_next, vel_next, acc = step(pos, vel, params)
        return (pos_next, vel_next), (pos, vel, acc)

    _, (pos, vel, acc) = jax.lax.scan(body, (pos0, vel0), None, length=n_steps)
    return {"pos": pos, "vel": vel, "acc": acc}


def random_initial_state(key: Array, n_balls: int, params: BallParams) -> tuple[Array, Array]:
    """A random "Newton's cradle"-like setup that is guaranteed to collide.

    Balls start roughly on a line along x, spaced a bit more than one diameter
    apart, with small random vertical offsets. The leftmost ball is launched to
    the right; the others start nearly at rest. This produces chains of
    collisions, so the graph's edges switch on and off over a trajectory.

    Returns:
        ``(pos0, vel0)``, each of shape ``(n_balls, 2)``.
    """
    k_gap, k_y, k_speed, k_vel = jax.random.split(key, 4)
    diameter = 2.0 * params.radius
    gaps = diameter + jax.random.uniform(k_gap, (n_balls,), minval=0.05, maxval=0.4)
    x = jnp.cumsum(gaps) - gaps[0]
    y = jax.random.uniform(k_y, (n_balls,), minval=-0.3, maxval=0.3) * params.radius
    pos0 = jnp.stack([x, y], axis=-1)

    vel0 = 0.1 * jax.random.normal(k_vel, (n_balls, 2))
    launch = jax.random.uniform(k_speed, (), minval=1.0, maxval=2.5)
    vel0 = vel0.at[0, 0].add(launch)
    return pos0, vel0


def make_dataset(
    key: Array, n_trajectories: int, n_steps: int, n_balls: int, params: BallParams
) -> dict[str, Array]:
    """Simulate a batch of random trajectories.

    Returns:
        Dict with ``pos``, ``vel``, ``acc`` of shape
        ``(n_trajectories, n_steps, n_balls, 2)``.
    """
    keys = jax.random.split(key, n_trajectories)

    def one(k):
        pos0, vel0 = random_initial_state(k, n_balls, params)
        return rollout(pos0, vel0, n_steps, params)

    return jax.vmap(one)(keys)
