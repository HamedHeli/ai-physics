"""Training the GNN on one-step acceleration targets, GNS-style.

Recipe (Sanchez-Gonzalez et al., 2020, Sec. 3 and Sec. 4):

* **Target:** the ground-truth acceleration of every particle, normalised to
  roughly unit variance.
* **Loss:** mean squared error over particles.
* **Noise injection:** Gaussian noise is added to the input velocities, and
  the target is corrected so the model learns to *undo* the noise:

      v_noisy  = v + eps
      a_target = (v_next - v_noisy) / dt = a - eps / dt

  The model therefore sees slightly wrong states during training, like the
  ones its own rollouts produce, which keeps long rollouts stable.
* **Optimiser:** Adam.

:func:`train` also stores a copy of the parameters every ``snapshot_every``
steps. The notebook uses these snapshots to show how the matrices inside the
network change during learning.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .graph import build_graph
from .model import apply
from .physics import BallParams


class Normalizer(NamedTuple):
    """Scales that bring inputs and targets to roughly unit size."""

    vel_scale: Array
    acc_scale: Array

    @staticmethod
    def fit(data: dict[str, Array]) -> "Normalizer":
        return Normalizer(
            vel_scale=jnp.std(data["vel"]) + 1e-8,
            acc_scale=jnp.std(data["acc"]) + 1e-8,
        )


def flatten_dataset(data: dict[str, Array]) -> dict[str, Array]:
    """``(traj, time, N, 2)`` arrays -> ``(samples, N, 2)``."""
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in data.items()}


def make_loss_fn(conn_radius: float, norm: Normalizer, phys: BallParams, noise_std: float):
    """Batch loss ``(params, key, pos, vel, acc) -> scalar``."""

    def single(params, key, pos, vel, acc):
        eps = noise_std * jax.random.normal(key, vel.shape)
        graph = build_graph(pos, vel + eps, conn_radius, norm.vel_scale)
        target = (acc - eps / phys.dt) / norm.acc_scale
        pred = apply(params, graph)
        return jnp.mean((pred - target) ** 2)

    def loss(params, key, pos, vel, acc):
        keys = jax.random.split(key, pos.shape[0])
        return jnp.mean(jax.vmap(single, in_axes=(None, 0, 0, 0, 0))(params, keys, pos, vel, acc))

    return loss


def train(
    params: dict,
    samples: dict[str, Array],
    conn_radius: float,
    norm: Normalizer,
    phys: BallParams,
    *,
    n_steps: int = 3000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    noise_std: float = 3e-3,
    snapshot_every: int = 100,
    seed: int = 0,
    log_every: int = 500,
) -> tuple[dict, list[dict], list[int], list[float]]:
    """Fit the GNN with Adam.

    Args:
        params: Initial parameters from :func:`src.model.init_params`.
        samples: Flattened dataset from :func:`flatten_dataset`.
        conn_radius: Connectivity radius ``R`` used to build graphs.
        norm: Input/target scales.
        phys: Ground-truth constants (``dt`` is needed for the noise correction).
        n_steps: Number of Adam steps.
        batch_size: Samples per step.
        learning_rate: Adam learning rate.
        noise_std: Std of the velocity noise (set to 0 to disable).
        snapshot_every: Store a parameter copy every this many steps.
        seed: PRNG seed for batching and noise.
        log_every: Print the loss every this many steps (0 = silent).

    Returns:
        ``(params, snapshots, snapshot_steps, losses)``. ``snapshots[0]`` is
        the untrained network; ``losses`` has one entry per step.
    """
    loss_fn = make_loss_fn(conn_radius, norm, phys, noise_std)
    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    # Oversample states where balls touch: most frames have zero force, and the
    # interesting physics (and the interesting messages) are in the collisions.
    touching = np.asarray(jnp.any(jnp.abs(samples["acc"]) > 1e-6, axis=(1, 2)))
    weights = np.where(touching, 1.0, 0.25)
    weights = weights / weights.sum()

    @jax.jit
    def update(params, opt_state, key, pos, vel, acc):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, pos, vel, acc)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    snapshots, snapshot_steps, losses = [params], [0], []
    n = samples["pos"].shape[0]
    for it in range(1, n_steps + 1):
        idx = rng.choice(n, size=batch_size, p=weights)
        key, sub = jax.random.split(key)
        params, opt_state, loss = update(
            params, opt_state, sub, samples["pos"][idx], samples["vel"][idx], samples["acc"][idx]
        )
        losses.append(float(loss))
        if it % snapshot_every == 0:
            snapshots.append(params)
            snapshot_steps.append(it)
        if log_every and it % log_every == 0:
            print(f"step {it:5d}  loss {np.mean(losses[-log_every:]):.4f}")
    return params, snapshots, snapshot_steps, losses


def learned_rollout(
    params: dict, pos0: Array, vel0: Array, n_steps: int, conn_radius: float,
    norm: Normalizer, phys: BallParams,
) -> dict[str, Array]:
    """Run the trained GNN autoregressively, exactly like the true simulator.

    Each step: build the graph, predict acceleration, un-normalise, and apply
    the same semi-implicit Euler update as :func:`src.physics.step`.
    """

    def body(carry, _):
        pos, vel = carry
        acc = apply(params, build_graph(pos, vel, conn_radius, norm.vel_scale)) * norm.acc_scale
        vel_next = vel + phys.dt * acc
        pos_next = pos + phys.dt * vel_next
        return (pos_next, vel_next), (pos, vel, acc)

    _, (pos, vel, acc) = jax.lax.scan(body, (pos0, vel0), None, length=n_steps)
    return {"pos": pos, "vel": vel, "acc": acc}
