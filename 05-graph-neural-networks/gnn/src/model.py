"""Encode-Process-Decode graph network (GNS, Sanchez-Gonzalez et al., 2020).

Architecture
------------
::

    Encoder     node latent  h_i  = MLP_node_enc(x_i)            (N, L)
                edge latent  e_ij = MLP_edge_enc(r_ij)           (E, L)

    Processor   repeat M times (each step has its own weights):
                message      m_ij = MLP_edge(e_ij, h_sender, h_receiver)   (E, L)
                             m_ij *= mask_ij          (absent edges send 0)
                aggregate    a_i  = sum of m_ij over edges arriving at i   (N, L)
                node update  h_i' = h_i + MLP_node(h_i, a_i)               (N, L)
                edge update  e_ij' = e_ij + m_ij                            (E, L)

    Decoder     acceleration y_i = MLP_dec(h_i)                    (N, 2)

The *edge function* produces the message, the plain sum collects the messages
at each receiver, and the *node function* reads that sum. With ``M`` steps,
information travels ``M`` hops through the graph.

GNS uses ``L = 128``, ``M = 10`` and LayerNorm. Here ``L = 8`` and ``M = 2`` by
default and LayerNorm is left out, so every intermediate matrix is small
enough to display and still learns three-ball collisions well.

Design notes
------------
* Pure JAX, no NN framework, matching the rest of this repository:
  parameters are a pytree of ``(W, b)`` lists.
* :func:`apply_with_trace` returns every intermediate matrix (latents,
  messages, aggregated messages per processor step). That trace is what the
  notebook visualises.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .graph import Graph

Layer = tuple[Array, Array]
MLPParams = list[Layer]


class GNNConfig(NamedTuple):
    """Static hyperparameters.

    Attributes:
        node_in: Number of input node features.
        edge_in: Number of input edge features.
        out: Output size per node (2 for a 2D acceleration).
        latent: Latent size ``L`` of node and edge vectors.
        hidden: Hidden width of every MLP.
        steps: Number of message-passing steps ``M``.
    """

    node_in: int = 2
    edge_in: int = 3
    out: int = 2
    latent: int = 8
    hidden: int = 32
    steps: int = 2


def init_mlp(key: Array, sizes: list[int]) -> MLPParams:
    """``W ~ N(0, 1/fan_in)``, ``b = 0``."""
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (i, o)) / jnp.sqrt(i), jnp.zeros((o,)))
        for k, i, o in zip(keys, sizes[:-1], sizes[1:])
    ]


def mlp(params: MLPParams, x: Array) -> Array:
    """ReLU hidden layers, linear output layer."""
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def init_params(key: Array, cfg: GNNConfig) -> dict:
    """Initialise all encoder, processor and decoder MLPs.

    Returns:
        ``{"node_enc", "edge_enc", "processor": [{"edge", "node"}, ...], "decoder"}``
    """
    L, H = cfg.latent, cfg.hidden
    k_ne, k_ee, k_dec, k_proc = jax.random.split(key, 4)
    proc_keys = jax.random.split(k_proc, 2 * cfg.steps)
    return {
        "node_enc": init_mlp(k_ne, [cfg.node_in, H, L]),
        "edge_enc": init_mlp(k_ee, [cfg.edge_in, H, L]),
        "processor": [
            {
                "edge": init_mlp(proc_keys[2 * m], [3 * L, H, L]),
                "node": init_mlp(proc_keys[2 * m + 1], [2 * L, H, L]),
            }
            for m in range(cfg.steps)
        ],
        "decoder": init_mlp(k_dec, [L, H, cfg.out]),
    }


class Trace(NamedTuple):
    """Every intermediate matrix of one forward pass.

    Per-step entries are stacked on a leading axis of length ``M``.

    Attributes:
        node_latent0: Encoded nodes ``h^0``, ``(N, L)``.
        edge_latent0: Encoded edges ``e^0``, ``(E, L)``.
        messages: Edge-function outputs (masked), ``(M, E, L)``.
        aggregated: Summed incoming messages per node, ``(M, N, L)``.
        node_latent: Node latents after each step, ``(M, N, L)``.
        edge_latent: Edge latents after each step, ``(M, E, L)``.
        output: Decoded (normalised) acceleration, ``(N, out)``.
    """

    node_latent0: Array
    edge_latent0: Array
    messages: Array
    aggregated: Array
    node_latent: Array
    edge_latent: Array
    output: Array


def apply_with_trace(params: dict, graph: Graph) -> Trace:
    """Run encode-process-decode and keep every intermediate matrix."""
    n_nodes = graph.nodes.shape[0]
    mask = graph.mask[:, None]
    h = mlp(params["node_enc"], graph.nodes)
    e = mlp(params["edge_enc"], graph.edges) * mask
    h0, e0 = h, e

    msgs, aggs, hs, es = [], [], [], []
    for block in params["processor"]:
        # Edge function: what does the sender do to the receiver?
        edge_in = jnp.concatenate([e, h[graph.senders], h[graph.receivers]], axis=-1)
        m = mlp(block["edge"], edge_in) * mask
        # Aggregate: sum messages arriving at each receiver (no parameters).
        agg = jax.ops.segment_sum(m, graph.receivers, num_segments=n_nodes)
        # Node function: update each node from its own state + the summed messages.
        h = h + mlp(block["node"], jnp.concatenate([h, agg], axis=-1))
        e = e + m
        msgs.append(m), aggs.append(agg), hs.append(h), es.append(e)

    out = mlp(params["decoder"], h)
    return Trace(h0, e0, jnp.stack(msgs), jnp.stack(aggs), jnp.stack(hs), jnp.stack(es), out)


def apply(params: dict, graph: Graph) -> Array:
    """Predicted (normalised) per-node acceleration, ``(N, out)``."""
    return apply_with_trace(params, graph).output
