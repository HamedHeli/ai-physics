"""Plotting helpers for the Transolver notebook: annotated heatmaps and fields on the plate."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

SLICE_COLORS = plt.get_cmap("tab10").colors


def matrix(ax, M, row_labels=None, col_labels=None, title="", vmax=None, fmt=None, cmap="RdBu_r",
           annotate=True, fontsize=6):
    """Heatmap of a small real matrix with every value written in its cell (unless ``annotate=False``).

    ``fmt`` defaults to 2 decimals, or 3 when every entry is below 0.1.
    """
    M = np.asarray(M)
    vmax = vmax if vmax is not None else max(float(np.abs(M).max()), 1e-6)
    fmt = fmt or ("{:.3f}" if vmax < 0.1 else "{:.2f}")
    vmin = 0 if cmap in ("magma", "viridis", "Greys", "Blues") else -vmax
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if annotate:
        for (i, j), v in np.ndenumerate(M):
            dark = abs(v) > 0.6 * vmax if vmin < 0 else (v > 0.6 * vmax if cmap == "Blues" else v < 0.5 * vmax)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "black")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels if row_labels is not None else range(M.shape[0]), fontsize=6)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels if col_labels is not None else range(M.shape[1]), fontsize=7)
    ax.set_title(title, fontsize=9)


def point_labels(n):
    return [f"p{i}" for i in range(n)]


def channel_labels(c, prefix="c"):
    return [f"{prefix}{i}" for i in range(c)]


def slice_labels(m):
    return [f"s{i}" for i in range(m)]


def plate(ax, hole, lw=1.2):
    """Draw the L-shaped outline and the hole, and set up the axes."""
    ax.add_patch(Polygon([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)], closed=True, fill=False,
                         ec="k", lw=lw))
    ax.add_patch(Circle(hole[:2], hole[2], fill=False, ec="k", lw=lw))
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def field(ax, points, values, hole, title="", vmin=None, vmax=None, cmap="inferno", s=12, colorbar=False):
    """Scatter of a value per point on the plate."""
    sc = ax.scatter(points[:, 0], points[:, 1], c=values, s=s, cmap=cmap, vmin=vmin, vmax=vmax, lw=0)
    plate(ax, hole)
    ax.set_title(title, fontsize=9)
    if colorbar:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    return sc


def grid_field(ax, u_grid, hole, title="", vmax=None, cmap="inferno"):
    """The solver's temperature on its grid, with the outside of the plate blanked out."""
    n = u_grid.shape[0]
    s = np.linspace(0, 1, n)
    X, Y = np.meshgrid(s, s)
    # Every node of the closed plate, boundary included, so the image fills the outline.
    mask = (~((X > 0.5) & (Y > 0.5))) & (np.hypot(X - hole[0], Y - hole[1]) >= hole[2])
    h = 0.5 / (n - 1)
    im = ax.imshow(np.where(mask, u_grid, np.nan), origin="lower", extent=(-h, 1 + h, -h, 1 + h), cmap=cmap,
                   vmin=0, vmax=vmax)
    plate(ax, hole)
    ax.set_title(title, fontsize=9)
    return im


def slice_map(ax, points, w, hole, title="", s=12, label_points=False):
    """Colour every point by its most likely slice; transparency = how sure (``max_m w_im``)."""
    w = np.asarray(w)
    m = w.argmax(1)
    colors = np.array([SLICE_COLORS[k % 10] for k in m])
    alpha = np.clip(w.max(1), 0.15, 1.0)
    ax.scatter(points[:, 0], points[:, 1], c=np.c_[colors, alpha], s=s, lw=0)
    if label_points:
        for i, (px, py) in enumerate(points):
            ax.text(px + 0.012, py + 0.012, str(i), fontsize=5)
    plate(ax, hole)
    ax.set_title(title, fontsize=9)


def slice_legend(ax, m):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", color=SLICE_COLORS[k % 10], label=f"s{k}") for k in range(m)]
    ax.legend(handles=handles, fontsize=6, ncol=m, loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False,
              handletextpad=0.1, columnspacing=0.6)


def physics_attention_flow(block: dict, head: int, points, hole, title=""):
    """One Physics-Attention head as a row of heatmaps:

    ``f`` (N×d) and ``w`` (N×M) -> tokens ``z`` (M×d) -> attention (M×M) ->
    ``z'`` (M×d) -> deslice ``f'`` (N×d), plus the slice map on the plate.
    """
    h = head
    f, w, z, attn = (np.asarray(block[k][h]) for k in ("f", "w", "z", "attn"))
    z_out, f_out = np.asarray(block["z_out"][h]), np.asarray(block["f_out"][h])
    n, m = w.shape
    d = f.shape[1]
    fig, axes = plt.subplots(1, 7, figsize=(19, 6.4), gridspec_kw=dict(width_ratios=[d, m, d, m, d, d, 9]))
    matrix(axes[0], f, point_labels(n), channel_labels(d, "f"), f"f: point features ({n}×{d})")
    matrix(axes[1], w, point_labels(n), slice_labels(m), f"w: slice weights ({n}×{m})", vmax=1, cmap="Blues")
    matrix(axes[2], z, slice_labels(m), channel_labels(d, "f"), f"z: slice tokens ({m}×{d})")
    matrix(axes[3], attn, slice_labels(m), slice_labels(m), f"attention ({m}×{m})", vmax=1, cmap="Blues")
    matrix(axes[4], z_out, slice_labels(m), channel_labels(d, "f"), f"z' = attn·v ({m}×{d})")
    matrix(axes[5], f_out, point_labels(n), channel_labels(d, "f"), f"f' = w·z' ({n}×{d})")
    slice_map(axes[6], np.asarray(points), w, hole, "most likely slice per point", s=40, label_points=True)
    slice_legend(axes[6], m)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


# --- Architecture diagram -----------------------------------------------------------------------------

_POINTWISE, _GATHER, _ATTEND = "#4C78A8", "#F58518", "#54A24B"   # per point · sum over points · among slices


def _tensor(ax, x, y, w, h, label, shape, color="0.85", fs=8):
    """A tensor drawn as a box: height ~ rows (N tall, M short), width ~ channels. ``(x, y)`` is the centre."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=color, ec="k", lw=1))
    ax.text(x, y + h / 2 + 0.12, label, ha="center", va="bottom", fontsize=fs, weight="bold")
    ax.text(x, y - h / 2 - 0.12, shape, ha="center", va="top", fontsize=fs - 1, family="monospace")


def _arrow(ax, x0, y0, x1, y1, text="", color="k", fs=7.5, dy=0.12, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.6,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=color)


def architecture_diagram(cfg, n_label="N"):
    """Transolver drawn as tensors and the operations between them.

    Top: the whole model (embed -> L blocks -> decode). Bottom: one Transolver
    block opened up, head by head. Box height shows the number of rows: ``N``
    (points) is tall, ``M`` (slices) is short, so the squeeze N -> M -> N of
    Physics-Attention is visible. Arrow colours: blue = the same small matrix at
    every row, orange = a weighted sum over points (slice / deslice), green =
    attention among the M slice tokens.
    """
    C, H, d, M, L = cfg.width, cfg.heads, cfg.head_dim, cfg.slices, cfg.layers
    N = n_label
    fig, ax = plt.subplots(figsize=(17, 10.5))
    ax.set_xlim(0, 17); ax.set_ylim(0, 10.5); ax.axis("off")
    tall, short = 1.9, 0.55

    # ---- top: the whole model ---------------------------------------------------------------------
    ax.text(0.2, 10.2, "Transolver, whole model (B plates per batch, N points per plate)",
            fontsize=12, weight="bold", va="top")
    yt = 8.3
    _tensor(ax, 1.2, yt, 0.45, tall, "input", f"(B, {N}, {cfg.in_dim})\nrows: points\ncols: x, y, d")
    _arrow(ax, 1.55, yt, 3.45, yt, f"embed MLP\n{cfg.in_dim}→{2 * C}→{C}, GELU", _POINTWISE)
    _tensor(ax, 4.0, yt, 0.9, tall, "x", f"(B, {N}, {C})\none row per point")
    _arrow(ax, 4.55, yt, 6.35, yt, "", "k")
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((6.4, yt - 1.15), 4.2, 2.3, boxstyle="round,pad=0.05", fc="#F2F2F2", ec="k", lw=1.2))
    ax.text(8.5, yt + 0.55, f"Transolver block × {L}", ha="center", fontsize=11, weight="bold")
    ax.text(8.5, yt - 0.05, "x ← x + PhysicsAttention(LN(x))\nx ← x + MLP(LN(x))", ha="center", va="center",
            fontsize=9, family="monospace")
    ax.text(8.5, yt - 0.85, f"keeps the shape (B, {N}, {C})", ha="center", fontsize=8, style="italic")
    _arrow(ax, 10.65, yt, 11.95, yt, "", "k")
    _tensor(ax, 12.5, yt, 0.9, tall, "x", f"(B, {N}, {C})")
    _arrow(ax, 13.05, yt, 15.0, yt, f"LN, then Linear\n{C}→1", _POINTWISE)
    _tensor(ax, 15.5, yt, 0.2, tall, "u", f"(B, {N})\none temperature\nper point")
    ax.plot([8.5, 8.5], [yt - 1.2, 6.25], color="0.5", ls=":", lw=1.2)
    ax.text(8.6, 6.55, "open one block", fontsize=8, color="0.4")

    # ---- bottom: one block, opened up ---------------------------------------------------------------
    ax.text(0.2, 6.15, f"One Transolver block, opened up ({H} heads of d = {d} channels, M = {M} slices per head; "
            "shapes for one plate; every tensor below exists once per head)", fontsize=12, weight="bold", va="top")
    yb, yg = 3.75, 1.0          # main row (point features f) and lower row (slice-choice features g)
    _tensor(ax, 0.75, yb, 0.9, tall, "LN(x)", f"({N}, {C})")
    _arrow(ax, 1.25, yb, 2.45, yb, "W_fx, split\ninto heads", _POINTWISE)
    _tensor(ax, 2.85, yb, 0.5, tall, "f", f"({N}, {d})")
    _arrow(ax, 1.0, yb - 1.05, 2.45, yg, "", _POINTWISE)
    ax.text(1.45, 1.55, "W_x, split\ninto heads", ha="right", fontsize=7.5, color=_POINTWISE)
    _tensor(ax, 2.85, yg, 0.5, tall * 0.55, "g", f"({N}, {d})", fs=7.5)
    _arrow(ax, 3.15, yg, 4.35, yg, "W_s, ÷τ,\nsoftmax over M", _POINTWISE, dy=-0.62)
    _tensor(ax, 4.85, yg, 0.5, tall * 0.55, "w", f"({N}, {M})\neach row sums to 1", fs=7.5)

    # slice: tokens = weighted averages of f with the columns of w
    _arrow(ax, 3.15, yb, 5.8, yb, "slice: z_m = Σ_i w_im f_i / Σ_i w_im", _GATHER)
    _arrow(ax, 5.12, yg + 0.3, 4.5, yb - 0.08, "", _GATHER)
    _tensor(ax, 6.1, yb, 0.5, short, "z", f"({M}, {d})\nslice tokens")
    _arrow(ax, 6.4, yb, 7.5, yb, "q, k, v = z W_q,\nz W_k, z W_v", _ATTEND)
    _tensor(ax, 7.95, yb, 0.55, short, "A", f"({M}, {M})\nsoftmax(q kᵀ/√d)")
    _arrow(ax, 8.3, yb, 9.35, yb, "z' = A v", _ATTEND)
    _tensor(ax, 9.7, yb, 0.5, short, "z'", f"({M}, {d})\nupdated tokens")

    # deslice: back to the points with the same w
    _arrow(ax, 10.0, yb, 11.95, yb, "deslice:\nf'_i = Σ_m w_im z'_m", _GATHER)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["FancyArrowPatch"]).FancyArrowPatch(
        (5.15, yg - 0.1), (12.0, yb - 0.75), arrowstyle="-|>", mutation_scale=12, color=_GATHER, lw=1.6, ls="--",
        connectionstyle="arc3,rad=0.12"))
    ax.text(8.6, 0.75, "the same w again", fontsize=7.5, color=_GATHER)
    _tensor(ax, 12.3, yb, 0.5, tall, "f'", f"({N}, {d})")
    _arrow(ax, 12.6, yb, 13.6, yb, "concat heads,\nW_o", _POINTWISE)
    _tensor(ax, 14.05, yb, 0.9, tall, "out", f"({N}, {C})")
    _arrow(ax, 14.55, yb, 15.55, yb, "+ x, then\n+ MLP(LN(·))", _POINTWISE)
    _tensor(ax, 16.15, yb, 0.9, tall, "x", f"({N}, {C})")

    # legend
    for k, (c, t) in enumerate([(_POINTWISE, "same small matrix at every row (no point sees another)"),
                                (_GATHER, "weighted sum over the N points (slice, deslice): the only N ↔ M step"),
                                (_ATTEND, "attention among the M slice tokens: the only all-to-all step, M × M")]):
        ax.plot([12.0, 12.5], [1.75 - 0.42 * k] * 2, color=c, lw=3)
        ax.text(12.6, 1.75 - 0.42 * k, t, fontsize=8, va="center")
    ax.text(12.0, 0.25, f"Box height = number of rows: {N} points (tall) vs M = {M} slices (short).", fontsize=8,
            style="italic")
    return fig


# --- What a hidden layer does: lines, ReLU, sums -------------------------------------------------------

def hidden_layer_intuition(hole=(0.3, 0.2, 0.09), margin=0.1, n_grid=161):
    """Hand-built neurons on the L-plate, with ``(x, y)`` as the only inputs.

    Row 1: five layer-1 neurons ``h_k = ReLU(R - n_k . (p - c))``. Each one is a
    line across the plate; ReLU keeps only the side facing the hole (value 0 on
    the other side), and the value grows with the distance from the line.
    Row 2: the five answers summed (weights -1, plus a bias, then ReLU) give
    "inside all five lines" = close to the hole. One more layer-1 neuron asks
    "near the bottom edge?". A layer-2 neuron combines the two answers into
    "close to the hole AND near the edge" = ``ReLU(A + B - 1)``.

    Returns the figure.
    """
    from .physics import in_L
    cx, cy, r = hole
    R = r + margin
    s = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(s, s)
    plate_mask = (~((X > 0.5) & (Y > 0.5))) & (np.hypot(X - cx, Y - cy) >= r)
    show = lambda v: np.where(plate_mask, v, np.nan)
    angles = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
    normals = np.c_[np.cos(angles), np.sin(angles)]
    h = [np.maximum(R - (nx * (X - cx) + ny * (Y - cy)), 0) for nx, ny in normals]
    total = sum(h)                                   # = 5R inside the pentagon, larger outside
    delta = 0.1
    A = np.maximum(5 * R + delta - total, 0) / delta  # "close to the hole", 1 inside, 0 far away
    B = np.maximum(0.2 - Y, 0) / 0.2                 # "near the bottom edge"
    AND = np.maximum(A + B - 1, 0)                   # layer 2: both answers yes

    def panel(ax, v, title, cmap="Blues", vmax=None, lines=()):
        ax.imshow(show(v), origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=vmax)
        plate(ax, hole)
        for nx, ny in lines:   # the line n . (p - c) = R, drawn across the plate
            t = np.linspace(-2, 2, 2)
            px, py = cx + R * nx - t * ny, cy + R * ny + t * nx
            ax.plot(px, py, color="tab:red", lw=1.2)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, fontsize=9)

    fig, axes = plt.subplots(2, 5, figsize=(17, 8.4), gridspec_kw=dict(hspace=0.35))
    for k, (ax, hk) in enumerate(zip(axes[0], h)):
        nx, ny = normals[k]
        panel(ax, hk, f"layer 1, neuron {k + 1}\nReLU({R:.2f} − ({nx:+.2f}·(x−{cx}) {ny:+.2f}·(y−{cy})))",
              lines=[normals[k]])
    panel(axes[1, 0], total, "sum of the 5 answers\n(flat inside the pentagon, grows outside)", "viridis",
          lines=normals)
    panel(axes[1, 1], A, "A = ReLU(bias − sum)\n\"close to the hole?\"", lines=normals)
    panel(axes[1, 2], B, "B = ReLU(0.2 − y) / 0.2\n\"near the bottom edge?\" (one more line)")
    panel(axes[1, 3], AND, "layer 2: ReLU(A + B − 1)\n\"close to the hole AND near the edge?\"", "Oranges")
    ax = axes[1, 4]; ax.axis("off")
    ax.text(0, 0.95, "How to read it\n\n"
            "• Each layer-1 neuron draws one straight\n  line (red) and asks \"which side, and\n  how far?\". ReLU zeroes the other side.\n\n"
            "• Width: more neurons = more lines =\n  more questions about the same input.\n  Five lines make a pentagon that\n  approximates the round hole.\n\n"
            "• Depth: a layer-2 neuron draws lines in\n  the space of layer-1 answers (A, B), so\n  it can ask follow-ups like A AND B.\n\n"
            "• White = 0: the neuron is off there.",
            fontsize=8.5, va="top", family="sans-serif")
    fig.suptitle("What a hidden layer does: lines, ReLU, and sums (hand-picked weights, inputs x and y only)",
                 fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.08)
    return fig
