"""Turning a set of particles into a graph (the GNS "encoder" input).

GNS (Sanchez-Gonzalez et al., 2020, Sec. 2) connects every pair of particles
closer than a connectivity radius ``R`` and recomputes those edges at every
time step, because particles move.

For a handful of particles it is simplest (and ``jit``-friendly, since array
shapes never change) to keep *all* ``N (N - 1)`` directed edges and attach a
0/1 mask that says which of them currently exist. A masked-out edge sends a
zero message, which is exactly the same as the edge not being there.

Conventions
-----------
* Edge ``e`` goes from ``senders[e]`` to ``receivers[e]``. The message on edge
  ``e`` is "what the sender does to the receiver", and it is summed at the
  receiver.
* Edge features (the "relative" encoder of GNS) are

      [ (p_receiver - p_sender) / R ,  |p_receiver - p_sender| / R ]

  i.e. the displacement pointing from sender to receiver (the direction a
  contact push acts on the receiver) and its length, both in units of ``R``.
  Absolute positions are never given to the network, which builds in
  translation invariance.
* Node features are the particle's current velocity. (GNS uses the last
  ``C = 5`` velocities plus a material type; one velocity is enough here,
  because the ground-truth force depends only on the current state.)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class Graph(NamedTuple):
    """Input graph for one time step.

    Attributes:
        nodes: Node features, shape ``(N, F_node)``.
        edges: Edge features, shape ``(E, F_edge)`` with ``E = N (N - 1)``.
        mask: 1.0 where the edge exists (``dist < R``), else 0.0; shape ``(E,)``.
        senders: Sender index of each edge, shape ``(E,)``.
        receivers: Receiver index of each edge, shape ``(E,)``.
    """

    nodes: Array
    edges: Array
    mask: Array
    senders: Array
    receivers: Array


def all_pairs(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered pair ``(s, r)`` with ``s != r``, as two index arrays.

    For three nodes this is 0->1, 0->2, 1->0, 1->2, 2->0, 2->1.
    """
    senders, receivers = np.nonzero(1 - np.eye(n_nodes, dtype=int))
    return senders, receivers


def edge_names(n_nodes: int, labels: str = "ABCDEFGHIJ") -> list[str]:
    """Human-readable edge names such as ``"A→B"``, in edge order."""
    senders, receivers = all_pairs(n_nodes)
    return [f"{labels[s]}→{labels[r]}" for s, r in zip(senders, receivers)]


def build_graph(pos: Array, vel: Array, radius: float, vel_scale: Array | float = 1.0) -> Graph:
    """Build the graph for one state.

    Args:
        pos: Positions, shape ``(N, 2)``.
        vel: Velocities, shape ``(N, 2)``.
        radius: Connectivity radius ``R``.
        vel_scale: Normalisation divisor for velocities (GNS normalises every
            input to roughly unit variance).

    Returns:
        A :class:`Graph`.
    """
    senders, receivers = all_pairs(pos.shape[0])
    disp = (pos[receivers] - pos[senders]) / radius
    dist = jnp.linalg.norm(disp, axis=-1, keepdims=True)
    mask = (dist[:, 0] < 1.0).astype(pos.dtype)
    edges = jnp.concatenate([disp, dist], axis=-1)
    nodes = vel / vel_scale
    return Graph(nodes, edges, mask, jnp.asarray(senders), jnp.asarray(receivers))


def adjacency_matrix(graph: Graph, n_nodes: int) -> Array:
    """Dense ``(N, N)`` adjacency matrix, ``A[s, r] = 1`` if edge s->r exists."""
    A = jnp.zeros((n_nodes, n_nodes))
    return A.at[graph.senders, graph.receivers].set(graph.mask)
