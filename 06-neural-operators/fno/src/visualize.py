"""Plotting helpers for the FNO notebook: annotated heatmaps of every tensor."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .model import Trace


def matrix(ax, M, row_labels=None, col_labels=None, title="", vmax=None, fmt=None, cmap="RdBu_r",
           annotate=True, fontsize=6):
    """Heatmap of a small real matrix with every value written in its cell (unless ``annotate=False``).

    ``fmt`` defaults to 2 decimals, or 3 when every entry is below 0.1.
    """
    M = np.asarray(M)
    vmax = vmax if vmax is not None else max(float(np.abs(M).max()), 1e-6)
    fmt = fmt or ("{:.3f}" if vmax < 0.1 else "{:.2f}")
    vmin = 0 if cmap in ("magma", "viridis", "Greys") else -vmax
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if annotate:
        for (i, j), v in np.ndenumerate(M):
            dark = abs(v) > 0.6 * vmax if vmin < 0 else v < 0.5 * vmax
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "black")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels if row_labels is not None else range(M.shape[0]), fontsize=6)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels if col_labels is not None else range(M.shape[1]), fontsize=7)
    ax.set_title(title, fontsize=9)


def channel_labels(d):
    return [f"c{i}" for i in range(d)]


def grid_labels(n):
    return [f"x{j}" for j in range(n)]


def mode_labels(k):
    return [f"k={i}" for i in range(k)]


def fourier_layer_flow(tr: Trace, t: int, k_max: int, title=""):
    """One Fourier layer as a row of heatmaps.

    ``v_t -> |FFT| (kept modes boxed) -> |R v_hat| -> K v`` on the top row and
    ``W v``, ``K v + W v`` and the activation output on the bottom row. Complex
    tensors are drawn as magnitudes.
    """
    v_in = np.asarray(tr.lifted if t == 0 else tr.v[t - 1])
    n, d = v_in.shape
    fig, axes = plt.subplots(2, 4, figsize=(17, 0.21 * n * 2 + 2.2),
                             gridspec_kw=dict(height_ratios=[1, 1]))
    rows, ch = grid_labels(n), channel_labels(d)
    vh = np.abs(np.asarray(tr.v_hat[t]))
    matrix(axes[0, 0], v_in, rows, ch, f"v_{t}  input to layer {t + 1}  ({n}×{d})")
    matrix(axes[0, 1], vh, mode_labels(vh.shape[0]), ch,
           f"|FFT(v_{t})|  all {vh.shape[0]} modes  ({vh.shape[0]}×{d})", cmap="magma")
    axes[0, 1].add_patch(Rectangle((-0.5, -0.5), d, k_max, fill=False, ec="cyan", lw=2.5))
    axes[0, 1].text(d - 0.5, k_max - 0.5, " keep\n k < k_max", color="tab:cyan", fontsize=8, va="top")
    matrix(axes[0, 2], np.abs(np.asarray(tr.Rv_hat[t])), mode_labels(k_max), ch,
           f"|R · v̂|  per-mode d×d product  ({k_max}×{d})", cmap="magma")
    matrix(axes[0, 3], tr.spectral[t], rows, ch, f"K v = iFFT(zero-pad(R v̂))  ({n}×{d})")
    axes[1, 0].axis("off")
    matrix(axes[1, 1], tr.local[t], rows, ch, f"W v + b  pointwise  ({n}×{d})")
    matrix(axes[1, 2], tr.pre_act[t], rows, ch, f"K v + W v + b  ({n}×{d})")
    last = t == tr.v.shape[0] - 1
    matrix(axes[1, 3], tr.v[t], rows, ch, f"v_{t + 1} = {'(no activation, last layer)' if last else 'GELU(…)'}  ({n}×{d})")
    axes[1, 0].text(0.0, 0.5,
                    "Top row: the GLOBAL path.\nEvery output point depends on\nevery input point via the FFT.\n\n"
                    "Bottom row: the LOCAL path\n(same d×d matrix W at every x)\nand the sum + activation.",
                    fontsize=10, va="center")
    fig.suptitle(title or f"Fourier layer {t + 1}", fontsize=12, weight="bold", x=0.02, ha="left")
    plt.tight_layout()
    return fig


def complex_weights(ax_re, ax_im, R_k, k):
    """Real and imaginary parts of one mode's ``d x d`` weight matrix ``R[k]``."""
    d = R_k.shape[0]
    vmax = float(max(np.abs(R_k.real).max(), np.abs(R_k.imag).max()))
    matrix(ax_re, R_k.real, channel_labels(d), channel_labels(d), f"Re R[k={k}]  (d×d, in → out)", vmax=vmax)
    matrix(ax_im, R_k.imag, channel_labels(d), channel_labels(d), f"Im R[k={k}]", vmax=vmax)


def field(ax, x, y, label=None, **kw):
    """A function on the periodic grid, drawn with markers at the grid points."""
    ax.plot(x, y, marker=".", ms=4, label=label, **kw)
    ax.set_xlim(0, 1)
