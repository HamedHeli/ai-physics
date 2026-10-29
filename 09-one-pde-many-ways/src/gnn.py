"""Method 5 (topic 05): a GNS-style learned simulator [Sanchez-Gonzalez+ 2020].

What is learned: **the time step** ``u_{t+dt} = u_t + GNN(u_t)`` with
``dt = 0.05`` (50 solver steps), applied autoregressively 20 times to reach
``T = 1``.

The graph
---------
Nodes are the 64 grid points, on a ring (periodic). Each node is connected to
its neighbours at offsets ``s in {-3, .., -1, 1, .., 3}``, so a node sees a
stencil of 7 points, like a wide finite-difference scheme whose coefficients
are learned and depend on the local state.

* Node input: ``u_i``.
* Edge input (sender ``j = i + s`` to receiver ``i``): ``(s dx, u_j - u_i)``.
  Only *relative* positions enter, so the model is translation-equivariant
  by construction, just like the PDE.

Encode-process-decode, as in ``05-graph-neural-networks/gnn`` (where the
graph has 3 balls instead of 64 grid points):

::

    h_i  = MLP_node_enc(u_i)           e_s,i = MLP_edge_enc(s dx, u_{i+s} - u_i)
    repeat M times:
        m_s,i = MLP_edge(e_s,i, h_i, h_{i+s})       message along each edge
        e_s,i += m_s,i
        h_i  += MLP_node(h_i, sum_s m_s,i)          aggregate, update
    du_i = MLP_dec(h_i)

With ``M = 3`` and radius 3 a node's update depends on 19 neighbours. On a
regular grid, ``jnp.roll`` by ``s`` *is* the edge gather, so there is no
explicit edge list.

Training uses one-step pairs with Gaussian noise added to the inputs (the GNS
trick so the model learns to correct its own rollout errors).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .nn import init_mlp, mlp

OFFSETS = (-3, -2, -1, 1, 2, 3)


def init_params(key: Array, latent: int = 32, hidden: int = 64, steps: int = 3) -> dict:
    keys = jax.random.split(key, 3 + 2 * steps)
    return {
        "node_enc": init_mlp(keys[0], [1, hidden, latent]),
        "edge_enc": init_mlp(keys[1], [2, hidden, latent]),
        "edge": [init_mlp(keys[3 + 2 * m], [3 * latent, hidden, latent]) for m in range(steps)],
        "node": [init_mlp(keys[4 + 2 * m], [2 * latent, hidden, latent]) for m in range(steps)],
        "dec": init_mlp(keys[2], [latent, hidden, 1]),
    }


def apply(params: dict, u: Array, u_scale: float, du_scale: float) -> Array:
    """Predicted increment ``u_{t+dt} - u_t`` for a state ``u`` of shape ``(n,)``."""
    n = u.shape[0]
    dx = 2 * jnp.pi / n
    un = u / u_scale
    s = jnp.array(OFFSETS)
    rel_u = jnp.stack([jnp.roll(un, -o) - un for o in OFFSETS])  # (S, n)
    rel_x = jnp.broadcast_to((s * dx)[:, None], rel_u.shape)
    h = mlp(params["node_enc"], un[:, None])  # (n, L)
    e = mlp(params["edge_enc"], jnp.stack([rel_x, rel_u], -1))  # (S, n, L)
    for edge_mlp, node_mlp in zip(params["edge"], params["node"]):
        h_send = jnp.stack([jnp.roll(h, -o, axis=0) for o in OFFSETS])  # (S, n, L)
        h_recv = jnp.broadcast_to(h, h_send.shape)
        m = mlp(edge_mlp, jnp.concatenate([e, h_recv, h_send], -1))
        e = e + m
        h = h + mlp(node_mlp, jnp.concatenate([h, m.sum(0)], -1))
    return mlp(params["dec"], h)[:, 0] * du_scale


def rollout(params: dict, u0: Array, n_steps: int, u_scale: float, du_scale: float) -> Array:
    """Autoregressive rollout; returns ``(n_steps + 1, n)`` including ``u0``."""

    def step(u, _):
        u = u + apply(params, u, u_scale, du_scale)
        return u, u

    _, us = jax.lax.scan(step, u0, None, length=n_steps)
    return jnp.concatenate([u0[None], us])


def make_loss(train_u: Array, batch: int = 64, noise: float = 0.01):
    """One-step loss on random ``(u_t, u_{t+1})`` pairs with input noise.

    Returns ``(loss, u_scale, du_scale)``; the scales normalise inputs and
    outputs.
    """
    x = train_u[:, :-1].reshape(-1, train_u.shape[-1])
    y = (train_u[:, 1:] - train_u[:, :-1]).reshape(-1, train_u.shape[-1])
    u_scale = float(x.std())
    du_scale = float(y.std())

    def loss(params, key):
        ki, kn = jax.random.split(key)
        idx = jax.random.randint(ki, (batch,), 0, x.shape[0])
        xb = x[idx] + noise * u_scale * jax.random.normal(kn, (batch, x.shape[1]))
        # target corrected for the noise, so the model also learns to undo it
        yb = x[idx] + y[idx] - xb
        pred = jax.vmap(lambda a: apply(params, a, u_scale, du_scale))(xb)
        return jnp.mean(((pred - yb) / du_scale) ** 2)

    return loss, u_scale, du_scale
