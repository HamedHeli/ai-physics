"""Training Transolver on (geometry, temperature) pairs (Wu et al., 2024, Sec. 4).

Recipe
------
* **Data:** for each plate, a point cloud with features ``(x, y, d)`` and the
  solver's temperature at those points.
* **Loss:** relative L2 error per sample, averaged over the batch, the metric
  the paper reports for every benchmark (and the FNO notebook uses too):

      loss = mean_j  ||G_theta(a_j) - u_j||_2 / ||u_j||_2

  It does not depend on how many points the cloud has, so the same number is
  meaningful for 64 and for 2000 points.
* **Optimiser:** AdamW with a one-cycle learning rate, as in the paper code
  (``OneCycleLR``), via ``optax.cosine_onecycle_schedule``.

There is no PDE residual in the loss: the physics enters through the solver's
data and through the geometry feature, as in the paper. :func:`train` stores a
copy of the parameters every ``snapshot_every`` steps for the notebook's explorer.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .model import TransolverConfig, apply


def relative_l2(pred: Array, target: Array) -> Array:
    """``||pred - target|| / ||target||`` along the last axis."""
    return jnp.linalg.norm(pred - target, axis=-1) / jnp.linalg.norm(target, axis=-1)


@partial(jax.jit, static_argnums=3)
def batch_loss(params: dict, feats: Array, u: Array, cfg: TransolverConfig) -> Array:
    """Mean relative L2 error over a batch, ``feats: (B, N, in_dim)``, ``u: (B, N)``."""
    pred = jax.vmap(apply, in_axes=(None, 0, None))(params, feats, cfg)
    return jnp.mean(relative_l2(pred, u))


def train(params: dict, feats: Array, u: Array, cfg: TransolverConfig, n_steps: int = 3000,
          batch_size: int = 16, lr: float = 3e-3, weight_decay: float = 1e-5, seed: int = 0,
          snapshot_every: int = 100, log_every: int = 500):
    """AdamW on random mini-batches of whole point clouds.

    Args:
        feats, u: Training features and targets, ``(n_train, N, in_dim)`` and ``(n_train, N)``.

    Returns:
        ``(params, snapshots, snapshot_steps, losses)``: final parameters, a list
        of parameter pytrees, the step of each snapshot, and the loss per step.
    """
    opt = optax.adamw(optax.cosine_onecycle_schedule(n_steps, lr), weight_decay=weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, f_b, u_b):
        loss, grads = jax.value_and_grad(batch_loss)(params, f_b, u_b, cfg)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    snapshots, snap_steps, losses = [params], [0], []
    for i in range(1, n_steps + 1):
        idx = rng.choice(feats.shape[0], batch_size, replace=False)
        params, opt_state, loss = step(params, opt_state, feats[idx], u[idx])
        losses.append(float(loss))
        if i % snapshot_every == 0:
            snapshots.append(params)
            snap_steps.append(i)
        if log_every and i % log_every == 0:
            print(f"step {i:5d}   relative L2 (train batch) = {loss:.4f}")
    return params, snapshots, snap_steps, np.array(losses)


def evaluate(params: dict, feats: Array, u: Array, cfg: TransolverConfig) -> float:
    """Mean relative L2 error on a set of point clouds, of whatever size they are."""
    return float(batch_loss(params, feats, u, cfg))
