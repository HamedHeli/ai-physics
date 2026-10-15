"""Plotting helpers for the GNN notebook: balls, graphs and annotated matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .graph import Graph, all_pairs

BALL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
LABELS = "ABCDEFGHIJ"


def matrix(ax, M, row_labels=None, col_labels=None, title="", vmax=None, fmt="{:.2f}", cmap="RdBu_r"):
    """Heatmap of a small matrix with every value written in its cell."""
    M = np.asarray(M)
    vmax = vmax if vmax is not None else max(float(np.abs(M).max()), 1e-6)
    ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 * vmax else "black")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels if row_labels is not None else range(M.shape[0]), fontsize=8)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels if col_labels is not None else range(M.shape[1]), fontsize=7)
    ax.set_title(title, fontsize=9)


def balls(ax, pos, radius, graph: Graph | None = None, edge_strength=None, acc_true=None,
          acc_pred=None, acc_scale=0.02, xlim=None, ylim=(-1.2, 1.2)):
    """Draw balls, the existing graph edges, and optional acceleration arrows.

    Args:
        edge_strength: Optional per-edge value (e.g. message norm); edge lines
            get thicker with it.
        acc_true / acc_pred: Optional ``(N, 2)`` accelerations, drawn as black
            and red arrows.
    """
    pos = np.asarray(pos)
    n = pos.shape[0]
    for i in range(n):
        ax.add_patch(Circle(pos[i], radius, color=BALL_COLORS[i % 6], alpha=0.35))
        ax.text(*pos[i], LABELS[i], ha="center", va="center", fontsize=11, weight="bold")
    if graph is not None:
        mask = np.asarray(graph.mask)
        s, r = all_pairs(n)
        strength = np.ones_like(mask) if edge_strength is None else np.asarray(edge_strength)
        top = max(float(strength.max()), 1e-6)
        for e in range(len(s)):
            if mask[e] > 0 and s[e] < r[e]:
                w = 1 + 6 * max(strength[e], strength[_reverse(e, s, r)]) / top
                ax.plot(*pos[[s[e], r[e]]].T, color="0.3", lw=w, zorder=0)
    for acc, color in ((acc_true, "black"), (acc_pred, "red")):
        if acc is not None:
            acc = np.asarray(acc)
            for i in range(n):
                ax.annotate("", pos[i] + acc_scale * acc[i], pos[i],
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _reverse(e, s, r):
    return int(np.nonzero((s == r[e]) & (r == s[e]))[0][0])
