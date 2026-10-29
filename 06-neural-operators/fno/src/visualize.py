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


# --- 2D helpers -----------------------------------------------------------------------------------


def image(ax, F, title="", cmap="RdBu_r", vmax=None):
    """One ``(n, n)`` field indexed ``[x, y]``, drawn with x to the right and y up."""
    F = np.asarray(F)
    vmax = vmax if vmax is not None else max(float(np.abs(F).max()), 1e-6)
    vmin = 0 if cmap in ("magma", "viridis") else -vmax
    ax.imshow(F.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def spectrum_image(ax, v_hat, m, title=""):
    """``log10 |v_hat|`` of one channel's ``rfft2`` ``(n, n/2+1)``, k_x centred, with the kept modes boxed."""
    S = np.fft.fftshift(np.abs(np.asarray(v_hat)), axes=0)
    n = S.shape[0]
    ax.imshow(np.log10(S + 1e-6).T, origin="lower", cmap="magma", vmin=-4,
              extent=[-n // 2 - 0.5, n // 2 - 0.5, -0.5, S.shape[1] - 0.5], aspect="auto")
    ax.add_patch(Rectangle((-m - 0.5, -0.5), 2 * m, m, fill=False, ec="cyan", lw=1.5))
    ax.set_xticks([-n // 2, 0, n // 2 - 1]), ax.set_yticks([0, S.shape[1] - 1])
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)


def layer_flow_2d(tr, t: int, m: int, title=""):
    """One 2D Fourier layer: one row per stage, one column per channel.

    Rows: input ``v_t``, ``log |FFT|`` (kept modes boxed), ``K v`` (global path),
    ``W v + b`` (local path), the sum, and ``v_{t+1}``.
    """
    v_in = np.asarray(tr.lifted if t == 0 else tr.v[t - 1])
    d = v_in.shape[-1]
    last = t == tr.v.shape[0] - 1
    rows = [("v_%d (input)" % t, v_in), ("log|FFT|", np.asarray(tr.v_hat[t])),
            ("K v  (global)", np.asarray(tr.spectral[t])), ("W v + b  (local)", np.asarray(tr.local[t])),
            ("K v + W v + b", np.asarray(tr.pre_act[t])),
            ("v_%d = %s" % (t + 1, "no act." if last else "GELU"), np.asarray(tr.v[t]))]
    fig, axes = plt.subplots(len(rows), d, figsize=(1.75 * d + 1.2, 1.75 * len(rows) + 0.6))
    for r, (name, T) in enumerate(rows):
        vmax = None if r == 1 else float(np.abs(T).max())
        for c in range(d):
            ax = axes[r, c]
            if r == 1:
                spectrum_image(ax, T[..., c], m)
            else:
                image(ax, T[..., c], vmax=vmax)
            if r == 0:
                ax.set_title(f"channel c{c}", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=9)
        axes[r, 0].yaxis.set_visible(True)
    fig.suptitle(title or f"2D Fourier layer {t + 1}", fontsize=12, weight="bold", x=0.02, ha="left")
    plt.tight_layout()
    return fig
