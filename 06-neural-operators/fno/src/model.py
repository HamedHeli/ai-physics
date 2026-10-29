"""Fourier Neural Operator, 1D (Li et al., 2021, "Fourier Neural Operator for
Parametric Partial Differential Equations", ICLR 2021, arXiv:2010.08895).

Architecture (paper Sec. 4, Fig. 2)
-----------------------------------
::

    input       a(x_j) and the coordinate x_j, stacked        (N, 2)
    lift P      v_0 = [a, x] P + b_P   (pointwise linear)      (N, d)

    Fourier layer t = 0 .. T-1  (each layer has its own R_t, W_t, b_t):
        FFT        v_hat = rfft(v_t)                            (N/2+1, d) complex
        truncate   keep the lowest k_max modes                  (k_max, d)
        multiply   (R v_hat)_k = v_hat_k @ R_t[k]  per mode     (k_max, d)
        inverse    K v = irfft(zero-padded R v_hat)             (N, d)
        local      W v = v_t W_t + b_t   (pointwise linear)     (N, d)
        activate   v_{t+1} = GELU(K v + W v)   (no GELU after the last layer)

    project Q   u = GELU(v_T Q1 + c1) Q2 + c2  (pointwise MLP)  (N, 1)

The kernel integral ``(K v)(x) = int kappa(x - y) v(y) dy`` (paper Eq. 3) is a
convolution, so by the convolution theorem it is a multiplication in Fourier
space (paper Eq. 4 and Sec. 4, "Fourier integral operator"). ``R_t[k]`` is a
learned complex ``d x d`` matrix for every kept mode ``k``, which mixes channels
at that frequency. All other modes are set to zero.

Everything that depends on ``x`` is either pointwise (P, W, Q, GELU) or acts on
Fourier modes (R). Nothing is tied to the grid size ``N``, which is why the same
weights run on 32, 64 or 256 points (paper Sec. 5.4, zero-shot super-resolution).

FFT normalisation
-----------------
``norm="forward"`` puts the ``1/N`` on the forward transform. Then ``v_hat_k`` is
the actual amplitude of the wave ``e^{2 pi i k x}`` and does not grow with ``N``,
which makes the numbers in the notebook comparable across resolutions. The paper
code uses the default normalisation (``1/N`` on the inverse); the composed map
FFT -> multiply -> inverse FFT is identical either way.

Sizes here are chosen to fit on screen: ``d = 8`` channels, ``k_max = 8`` modes,
``T = 4`` layers, ``N = 32`` grid points. The paper uses ``d = 64``,
``k_max = 16``, ``T = 4`` and ``N`` up to 8192.

Design notes
------------
* Pure JAX, no NN framework, matching the rest of this repository: parameters
  are a pytree of arrays. ``R`` is stored as separate real and imaginary parts so
  every leaf is a real array.
* :func:`apply_with_trace` returns every intermediate tensor. That trace is what
  the notebook visualises.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class FNOConfig(NamedTuple):
    """Static hyperparameters.

    Attributes:
        in_channels: Input channels per grid point (``a(x)`` and ``x``).
        width: Channel width ``d`` of the hidden function ``v``.
        modes: Number of Fourier modes ``k_max`` kept in every layer.
        layers: Number of Fourier layers ``T``.
        proj_hidden: Hidden width of the projection MLP ``Q``.
    """

    in_channels: int = 2
    width: int = 8
    modes: int = 8
    layers: int = 4
    proj_hidden: int = 32


def init_params(key: Array, cfg: FNOConfig) -> dict:
    """Initialise P, the Fourier layers and Q.

    ``P``, ``W`` and ``Q``: ``W ~ N(0, 1/fan_in)``, ``b = 0``.
    ``R``: real and imaginary parts ``~ U(0, 1) / d^2``, as in the paper code.

    Returns:
        ``{"P": (W, b), "layers": [{"R_re", "R_im", "W", "b"}, ...], "Q": [(W1, b1), (W2, b2)]}``
    """
    d, k = cfg.width, cfg.modes
    k_p, k_q1, k_q2, k_layers = jax.random.split(key, 4)

    def dense(key, i, o):
        return jax.random.normal(key, (i, o)) / jnp.sqrt(i), jnp.zeros((o,))

    layers = []
    for lk in jax.random.split(k_layers, cfg.layers):
        k_re, k_im, k_w = jax.random.split(lk, 3)
        W, b = dense(k_w, d, d)
        layers.append({
            "R_re": jax.random.uniform(k_re, (k, d, d)) / d**2,
            "R_im": jax.random.uniform(k_im, (k, d, d)) / d**2,
            "W": W,
            "b": b,
        })
    return {
        "P": dense(k_p, cfg.in_channels, d),
        "layers": layers,
        "Q": [dense(k_q1, d, cfg.proj_hidden), dense(k_q2, cfg.proj_hidden, 1)],
    }


def make_input(a: Array) -> Array:
    """Stack ``a(x_j)`` with the coordinate ``x_j = j / N``: ``(N,) -> (N, 2)``.

    The paper appends the grid coordinates to the input so the network knows
    where each point is (code: ``get_grid``).
    """
    n = a.shape[-1]
    x = jnp.broadcast_to(jnp.arange(n) / n, a.shape)
    return jnp.stack([a, x], axis=-1)


def spectral_conv(v: Array, R: Array) -> tuple[Array, Array, Array, Array]:
    """The global part of a Fourier layer, ``K v = irfft(R . rfft(v))``.

    Args:
        v: Hidden function on the grid, ``(N, d)``.
        R: Complex weights, ``(k_max, d, d)``.

    Returns:
        ``(v_hat, v_hat_kept, Rv_hat, Kv)``: all ``N/2 + 1`` modes ``(N/2+1, d)``,
        the kept modes ``(k_max, d)``, the kept modes after the per-mode matrix
        product ``(k_max, d)``, and the result back on the grid ``(N, d)``.
    """
    n, k_max = v.shape[0], R.shape[0]
    v_hat = jnp.fft.rfft(v, axis=0, norm="forward")
    kept = v_hat[:k_max]
    Rv = jnp.einsum("ki,kio->ko", kept, R)
    padded = jnp.zeros_like(v_hat).at[:k_max].set(Rv)
    Kv = jnp.fft.irfft(padded, n=n, axis=0, norm="forward")
    return v_hat, kept, Rv, Kv


class Trace(NamedTuple):
    """Every intermediate tensor of one forward pass.

    Per-layer entries are stacked on a leading axis of length ``T``.

    Attributes:
        inputs: ``[a, x]``, ``(N, 2)``.
        lifted: ``v_0 = P(inputs)``, ``(N, d)``.
        v_hat: All Fourier modes of ``v_t``, ``(T, N/2+1, d)`` complex.
        v_hat_kept: The lowest ``k_max`` modes, ``(T, k_max, d)`` complex.
        Rv_hat: Kept modes after multiplying by ``R``, ``(T, k_max, d)`` complex.
        spectral: ``K v_t`` back on the grid, ``(T, N, d)``.
        local: ``W v_t + b``, ``(T, N, d)``.
        pre_act: ``K v_t + W v_t + b``, ``(T, N, d)``.
        v: ``v_{t+1}`` after the activation, ``(T, N, d)``.
        proj_hidden: Hidden layer of ``Q``, ``(N, proj_hidden)``.
        output: Predicted ``u(x, 1)``, ``(N,)``.
    """

    inputs: Array
    lifted: Array
    v_hat: Array
    v_hat_kept: Array
    Rv_hat: Array
    spectral: Array
    local: Array
    pre_act: Array
    v: Array
    proj_hidden: Array
    output: Array


def apply_with_trace(params: dict, a: Array) -> Trace:
    """Run the FNO on one input function ``a`` of shape ``(N,)`` and keep every tensor."""
    inputs = make_input(a)
    W_P, b_P = params["P"]
    v = inputs @ W_P + b_P
    lifted = v

    rec = {k: [] for k in ("v_hat", "v_hat_kept", "Rv_hat", "spectral", "local", "pre_act", "v")}
    n_layers = len(params["layers"])
    for t, layer in enumerate(params["layers"]):
        R = layer["R_re"] + 1j * layer["R_im"]
        v_hat, kept, Rv, Kv = spectral_conv(v, R)       # global: convolution via FFT
        Wv = v @ layer["W"] + layer["b"]                # local: same 1x1 map at every point
        pre = Kv + Wv
        v = jax.nn.gelu(pre) if t < n_layers - 1 else pre
        for name, val in zip(rec, (v_hat, kept, Rv, Kv, Wv, pre, v)):
            rec[name].append(val)

    (W1, b1), (W2, b2) = params["Q"]
    hidden = jax.nn.gelu(v @ W1 + b1)
    out = (hidden @ W2 + b2)[:, 0]
    return Trace(inputs, lifted, **{k: jnp.stack(vs) for k, vs in rec.items()}, proj_hidden=hidden, output=out)


def apply(params: dict, a: Array) -> Array:
    """Predicted ``u(x, 1)`` for one input ``a``, ``(N,) -> (N,)``. Works for any ``N >= 2 k_max``."""
    return apply_with_trace(params, a).output
