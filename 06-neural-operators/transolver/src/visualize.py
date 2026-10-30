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
