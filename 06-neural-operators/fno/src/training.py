"""Training the FNO on input/output function pairs (Li et al., 2021, Sec. 5).

Recipe
------
* **Data:** pairs ``(a_j, u_j)`` with ``u_j = G(a_j)`` from the reference solver,
  subsampled to the training grid.
* **Loss:** the paper's relative L2 error, averaged over the batch:

      loss = mean_j  ||G_theta(a_j) - u_j||_2 / ||u_j||_2

  The ratio does not depend on the grid size or on how large ``u_j`` is, so the
  same number is meaningful at 32 and at 256 points.
* **Optimiser:** Adam with a cosine-decayed learning rate (the paper uses Adam
  with step decay).

:func:`train` stores a copy of the parameters every ``snapshot_every`` steps.
The notebook uses the snapshots to show how the tensors inside the network
change during learning. Nothing here solves an adjoint equation: the gradient
of the loss comes from ordinary reverse-mode autodiff through the network.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .model import apply


def relative_l2(pred: Array, target: Array) -> Array:
    """``||pred - target|| / ||target||`` along the last axis."""
    return jnp.linalg.norm(pred - target, axis=-1) / jnp.linalg.norm(target, axis=-1)


def batch_loss(params: dict, a: Array, u: Array, apply_fn=apply) -> Array:
    """Mean relative L2 error over a batch, ``a: (B, ...)``, ``u: (B, ...)``.

    Each sample's prediction and target are flattened first, so the same loss
    serves the 1D model (``u: (B, N)``) and the 2D one (``u: (B, n, n, 2)``).
    """
    pred = jax.vmap(apply_fn, in_axes=(None, 0))(params, a)
    b = u.shape[0]
    return jnp.mean(relative_l2(pred.reshape(b, -1), u.reshape(b, -1)))


def train(params: dict, a: Array, u: Array, n_steps: int = 3000, batch_size: int = 32,
          lr: float = 5e-3, seed: int = 0, snapshot_every: int = 100, log_every: int = 500,
          apply_fn=apply):
    """Adam on random mini-batches.

    Args:
        a, u: Training inputs and targets on the training grid, ``(n_train, ...)``.
        apply_fn: Forward pass ``(params, a) -> u`` for one sample
            (:func:`src.model.apply` or :func:`src.model2d.apply`).

    Returns:
        ``(params, snapshots, snapshot_steps, losses)``: final parameters, a list
        of parameter pytrees, the step of each snapshot, and the loss per step.
    """
    opt = optax.adam(optax.cosine_decay_schedule(lr, n_steps))
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, a_b, u_b):
        loss, grads = jax.value_and_grad(batch_loss)(params, a_b, u_b, apply_fn)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    snapshots, snap_steps, losses = [params], [0], []
    for i in range(1, n_steps + 1):
        idx = rng.choice(a.shape[0], batch_size, replace=False)
        params, opt_state, loss = step(params, opt_state, a[idx], u[idx])
        losses.append(float(loss))
        if i % snapshot_every == 0:
            snapshots.append(params)
            snap_steps.append(i)
        if log_every and i % log_every == 0:
            print(f"step {i:5d}   relative L2 (train batch) = {loss:.4f}")
    return params, snapshots, snap_steps, np.array(losses)


_batch_loss_jit = jax.jit(batch_loss, static_argnames="apply_fn")


def evaluate(params: dict, a: Array, u: Array, apply_fn=apply) -> float:
    """Mean relative L2 error on a set of pairs, at whatever resolution they are given."""
    return float(_batch_loss_jit(params, a, u, apply_fn=apply_fn))
