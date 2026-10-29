"""Fourier Neural Operator, 2D (Li et al., 2021, Sec. 4 and the 2D experiments
of Sec. 5.2-5.3), for a 2D velocity field.

Architecture
------------
::

    input       [u, v, x, y] at every grid point                  (n, n, 4)
    lift P      v_0 = input P + b_P   (same 4 x d matrix at every point)  (n, n, d)

    Fourier layer t = 0 .. T-1 (own R_t, W_t, b_t):
        FFT        v_hat = rfft2(v_t), separately for each of the d channels
                                                                   (n, n/2+1, d) complex
        truncate   keep |k_x| < m and k_y < m: two corner blocks
                   ("low" k_x = 0..m-1 and "high" k_x = -m..-1)      (2, m, m, d)
        multiply   (R v_hat)[block, kx, ky] = v_hat[block, kx, ky] @ R_t[block, kx, ky]
                   (one complex d x d matrix per kept mode)          (2, m, m, d)
        inverse    K v = irfft2(R v_hat placed back, zeros elsewhere)  (n, n, d)
        local      W v = v_t W_t + b_t                              (n, n, d)
        activate   v_{t+1} = GELU(K v + W v)   (no GELU after the last layer)

    project Q   [u, v](t_final) = GELU(v_T Q1 + c1) Q2 + c2          (n, n, 2)

Why two corner blocks: ``rfft2`` keeps only ``k_y >= 0`` (the ``k_y < 0`` half is
the complex conjugate for a real field), but ``k_x`` runs over both signs, stored
as ``0, 1, ..., n/2-1, -n/2, ..., -1``. The lowest frequencies therefore sit in
the first ``m`` and the last ``m`` rows (paper code: ``weights1`` and ``weights2``).

Everything else is as in the 1D model (``model.py``): pointwise P, W, Q; learned
weights on Fourier modes; ``norm="forward"`` so coefficients are actual wave
amplitudes. Sizes chosen to be shown: ``d = 8``, ``m = 8``, ``T = 4``, ``n = 64``.
The paper's 2D models use ``d = 20-32`` and ``m = 12``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class FNO2DConfig(NamedTuple):
    """Static hyperparameters.

    Attributes:
        in_channels: Input channels per grid point (``u, v, x, y``).
        out_channels: Output channels per grid point (``u, v``).
        width: Channel width ``d``.
        modes: Modes ``m`` kept per direction (per corner block).
        layers: Number of Fourier layers ``T``.
        proj_hidden: Hidden width of the projection MLP ``Q``.
    """

    in_channels: int = 4
    out_channels: int = 2
    width: int = 8
    modes: int = 8
    layers: int = 4
    proj_hidden: int = 32


def init_params(key: Array, cfg: FNO2DConfig) -> dict:
    """Initialise P, the Fourier layers and Q (same scheme as ``model.init_params``).

    Returns:
        ``{"P": (W, b), "layers": [{"R_re", "R_im", "W", "b"}, ...], "Q": [(W1, b1), (W2, b2)]}``
        with ``R_re, R_im`` of shape ``(2, m, m, d, d)``.
    """
    d, m = cfg.width, cfg.modes
    k_p, k_q1, k_q2, k_layers = jax.random.split(key, 4)

    def dense(key, i, o):
        return jax.random.normal(key, (i, o)) / jnp.sqrt(i), jnp.zeros((o,))

    layers = []
    for lk in jax.random.split(k_layers, cfg.layers):
        k_re, k_im, k_w = jax.random.split(lk, 3)
        W, b = dense(k_w, d, d)
        layers.append({
            "R_re": jax.random.uniform(k_re, (2, m, m, d, d)) / d**2,
            "R_im": jax.random.uniform(k_im, (2, m, m, d, d)) / d**2,
            "W": W,
            "b": b,
        })
    return {
        "P": dense(k_p, cfg.in_channels, d),
        "layers": layers,
        "Q": [dense(k_q1, d, cfg.proj_hidden), dense(k_q2, cfg.proj_hidden, cfg.out_channels)],
    }


def make_input(w: Array) -> Array:
    """Append the coordinates: velocity ``(n, n, 2)`` -> ``[u, v, x, y]`` ``(n, n, 4)``."""
    n = w.shape[0]
    X, Y = jnp.meshgrid(jnp.arange(n) / n, jnp.arange(n) / n, indexing="ij")
    return jnp.concatenate([w, X[..., None], Y[..., None]], axis=-1)


def spectral_conv2d(v: Array, R: Array) -> tuple[Array, Array, Array, Array]:
    """``K v = irfft2(R . rfft2(v))`` keeping the two low-frequency corner blocks.

    Args:
        v: ``(n, n, d)``.
        R: Complex weights ``(2, m, m, d, d)``.

    Returns:
        ``(v_hat, kept, Rv, Kv)`` with shapes ``(n, n/2+1, d)``, ``(2, m, m, d)``,
        ``(2, m, m, d)``, ``(n, n, d)``.
    """
    n, m = v.shape[0], R.shape[1]
    v_hat = jnp.fft.rfft2(v, axes=(0, 1), norm="forward")
    kept = jnp.stack([v_hat[:m, :m], v_hat[-m:, :m]])
    Rv = jnp.einsum("bxyi,bxyio->bxyo", kept, R)
    out = jnp.zeros_like(v_hat).at[:m, :m].set(Rv[0]).at[-m:, :m].set(Rv[1])
    Kv = jnp.fft.irfft2(out, s=(n, n), axes=(0, 1), norm="forward")
    return v_hat, kept, Rv, Kv


class Trace(NamedTuple):
    """Every intermediate tensor of one 2D forward pass (per-layer entries stacked on axis 0)."""

    inputs: Array        # (n, n, 4)
    lifted: Array        # (n, n, d)
    v_hat: Array         # (T, n, n/2+1, d) complex
    v_hat_kept: Array    # (T, 2, m, m, d) complex
    Rv_hat: Array        # (T, 2, m, m, d) complex
    spectral: Array      # (T, n, n, d)
    local: Array         # (T, n, n, d)
    pre_act: Array       # (T, n, n, d)
    v: Array             # (T, n, n, d)
    proj_hidden: Array   # (n, n, proj_hidden)
    output: Array        # (n, n, 2)


def apply_with_trace(params: dict, w: Array) -> Trace:
    """Run the 2D FNO on one velocity field ``(n, n, 2)`` and keep every tensor."""
    inputs = make_input(w)
    W_P, b_P = params["P"]
    v = inputs @ W_P + b_P
    lifted = v

    rec = {k: [] for k in ("v_hat", "v_hat_kept", "Rv_hat", "spectral", "local", "pre_act", "v")}
    n_layers = len(params["layers"])
    for t, layer in enumerate(params["layers"]):
        R = layer["R_re"] + 1j * layer["R_im"]
        v_hat, kept, Rv, Kv = spectral_conv2d(v, R)
        Wv = v @ layer["W"] + layer["b"]
        pre = Kv + Wv
        v = jax.nn.gelu(pre) if t < n_layers - 1 else pre
        for name, val in zip(rec, (v_hat, kept, Rv, Kv, Wv, pre, v)):
            rec[name].append(val)

    (W1, b1), (W2, b2) = params["Q"]
    hidden = jax.nn.gelu(v @ W1 + b1)
    out = hidden @ W2 + b2
    return Trace(inputs, lifted, **{k: jnp.stack(vs) for k, vs in rec.items()}, proj_hidden=hidden, output=out)


def apply(params: dict, w: Array) -> Array:
    """Predicted velocity at ``t_final``, ``(n, n, 2) -> (n, n, 2)``."""
    return apply_with_trace(params, w).output
