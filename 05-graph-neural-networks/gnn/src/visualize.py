"""Plotting helpers for the GNN notebook: balls, graphs and annotated matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .graph import Graph, all_pairs

BALL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
LABELS = "ABCDEFGHIJ"


def matrix(ax, M, row_labels=None, col_labels=None, title="", vmax=None, fmt="{:.2f}", cmap="RdBu_r",
           annotate=True):
    """Heatmap of a small matrix with every value written in its cell (unless ``annotate=False``)."""
    M = np.asarray(M)
    vmax = vmax if vmax is not None else max(float(np.abs(M).max()), 1e-6)
    ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(M if annotate else np.empty((0, 0))):
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


def mlp_diagram(ax, sizes, in_labels=None, out_labels=None, title="", hidden_label="ReLU"):
    """Neuron-and-weight drawing of a small MLP (one column of circles per layer).

    Args:
        sizes: Layer widths, e.g. ``[2, 32, 8]``.
        in_labels / out_labels: Names written next to the input and output neurons.
    """
    xs = np.linspace(0, 1, len(sizes))
    ys = [np.linspace(0.95, 0.05, n) if n > 1 else np.array([0.5]) for n in sizes]
    for k in range(len(sizes) - 1):
        for y0 in ys[k]:
            for y1 in ys[k + 1]:
                ax.plot([xs[k], xs[k + 1]], [y0, y1], color="0.6", lw=0.3, alpha=0.5, zorder=0)
    colors = ["#4C72B0"] + ["#8172B3"] * (len(sizes) - 2) + ["#C44E52"]
    for k, (x, y) in enumerate(zip(xs, ys)):
        s = 60 if len(y) <= 8 else 18
        ax.scatter(np.full_like(y, x), y, s=s, color=colors[k], zorder=2, edgecolor="white", lw=0.5)
    for labels, k, dx, ha in ((in_labels, 0, -0.03, "right"), (out_labels, -1, 0.03, "left")):
        if labels is not None:
            for y, lab in zip(ys[k], labels):
                ax.text(xs[k] + dx, y, lab, ha=ha, va="center", fontsize=8)
    names = ["input"] + [f"hidden\n{n} units, {hidden_label}" for n in sizes[1:-1]] + ["output\n(linear)"]
    for x, n, name in zip(xs, sizes, names):
        ax.text(x, -0.06, f"{name}\n[{n}]" if "hidden" not in name else name, ha="center", va="top", fontsize=8)
    for k in range(len(sizes) - 1):
        ax.text((xs[k] + xs[k + 1]) / 2, 1.0, f"W{k + 1}: {sizes[k]}×{sizes[k + 1]}", ha="center",
                va="bottom", fontsize=8, color="0.3")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.3, 1.08)
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def mlp_flow(layers, X, row_labels, in_labels=None, out_labels=None, title="", mask=None):
    """Push a matrix through a one-hidden-layer MLP and show every stage.

    Draws ``X -> relu(X W1 + b1) -> (hidden) W2 + b2 -> output`` as three
    heatmaps. Every row of ``X`` goes through the same weights, so the row
    labels stay the same from left to right. Matrices wider than 16 columns are
    drawn as colours only (white = 0).

    Args:
        layers: MLP parameters ``[(W1, b1), (W2, b2)]``.
        X: Input matrix, shape ``(rows, in)``.
        mask: Optional per-row 0/1 multiplier applied to the output (the edge mask).

    Returns:
        ``(fig, output)``.
    """
    (W1, b1), (W2, b2) = [(np.asarray(W), np.asarray(b)) for W, b in layers]
    X = np.asarray(X)
    H = np.maximum(0.0, X @ W1 + b1)
    Y = H @ W2 + b2
    if mask is not None:
        Y = Y * np.asarray(mask)[:, None]
    mats = [X, H, Y]
    cell_w = [0.42 if M.shape[1] <= 16 else 0.2 for M in mats]
    widths = [M.shape[1] * w for M, w in zip(mats, cell_w)]
    left, gap, cell_h = 0.7, 2.0, 0.32
    height = len(X) * cell_h
    fig_w, fig_h = left + sum(widths) + 2 * gap + 0.2, height + 1.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    out_title = f"output = hidden W2 + b2  ({len(X)}×{Y.shape[1]})" + ("  × mask" if mask is not None else "")
    titles = [f"input X  ({len(X)}×{X.shape[1]})", f"hidden = ReLU(X W1 + b1)  ({len(X)}×{H.shape[1]})", out_title]
    cols = [in_labels, [str(j) for j in range(H.shape[1])], out_labels]
    x = left
    for k, (M, w, t, c) in enumerate(zip(mats, widths, titles, cols)):
        ax = fig.add_axes([x / fig_w, 0.55 / fig_h, w / fig_w, height / fig_h])
        matrix(ax, M, row_labels, c, t, fmt="{:.2f}" if k != 1 else "{:.1f}", annotate=M.shape[1] <= 16)
        if k < 2:
            sizes = (X.shape[1], H.shape[1]) if k == 0 else (H.shape[1], Y.shape[1])
            label = f"× W{k + 1} ({sizes[0]}×{sizes[1]}) + b{k + 1}" + ("\nthen ReLU" if k == 0 else "")
            fig.text((x + w + (gap - 0.55) / 2) / fig_w, (0.55 + height / 2) / fig_h, label + "\n⟶", ha="center",
                     va="center", fontsize=9, color="0.25")
        x += w + gap
    fig.text(0.1 / fig_w, 1 - 0.15 / fig_h, title, fontsize=11, ha="left", va="top", weight="bold")
    return fig, Y
