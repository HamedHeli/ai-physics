"""Method 6 (topic 06): a Fourier neural operator [Li+ 2020].

What is learned: **the solution operator** ``G: u(., 0) -> u(., T)`` for every
initial condition at once, in one shot (no time stepping).

::

    v_0(x)     = P (u0(x), sin x, cos x)                 lift to `width` channels
    v_{l+1}(x) = gelu( W_l v_l(x) + IFFT( R_l . FFT(v_l) )(x) )   l = 0..L-1
    u_T(x)     = Q v_L(x)                                project back to 1 channel

``R_l`` is a complex ``(width, width)`` matrix for each of the lowest ``modes``
Fourier modes; higher modes are dropped by the spectral branch (``W_l``, a
pointwise linear map, still passes them through). Because ``R_l`` is attached
to wavenumbers, not to grid points, the same weights run on any grid size:
train on 64 points, evaluate on 128 or 256 (zero-shot super-resolution).

A separate thread is writing an FNO notebook in ``06-neural-operators``; this
one is kept self-contained so the folder runs on its own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .nn import init_linear, init_mlp, linear, mlp


def init_params(key: Array, width: int = 32, modes: int = 16, layers: int = 4) -> dict:
    """The complex weights ``R_l`` are stored as a real array of shape
    ``(2, modes, width, width)`` (real and imaginary parts): Adam's
    second-moment estimate ``g * g`` is only meaningful for real numbers."""
    keys = jax.random.split(key, 2 + 3 * layers)
    scale = 1.0 / width
    spectral = [scale * jax.random.normal(keys[2 + 3 * l], (2, modes, width, width)) for l in range(layers)]
    return {
        "lift": init_linear(keys[0], 3, width),
        "spectral": spectral,
        "pointwise": [init_linear(keys[4 + 3 * l], width, width) for l in range(layers)],
        "proj": init_mlp(keys[1], [width, 64, 1]),
    }


def spectral_conv(r: Array, v: Array) -> Array:
    """``IFFT(R . FFT(v))`` keeping ``modes`` modes. ``v``: ``(n, width)``,
    ``r``: ``(2, modes, width, width)``."""
    n = v.shape[0]
    r = r[0] + 1j * r[1]
    modes = r.shape[0]
    v_hat = jnp.fft.rfft(v, axis=0)  # (n//2+1, width)
    out = jnp.zeros_like(v_hat).at[:modes].set(jnp.einsum("kio,ki->ko", r, v_hat[:modes]))
    return jnp.fft.irfft(out, n=n, axis=0)


def apply(params: dict, u0: Array, return_hidden: bool = False):
    """``u(., T)`` from ``u0`` of shape ``(n,)``; any ``n`` works."""
    n = u0.shape[0]
    x = jnp.arange(n) * (2 * jnp.pi / n)
    v = linear(params["lift"], jnp.stack([u0, jnp.sin(x), jnp.cos(x)], -1))
    hidden = [v]
    n_layers = len(params["spectral"])
    for l, (r, w) in enumerate(zip(params["spectral"], params["pointwise"])):
        v = spectral_conv(r, v) + linear(w, v)
        if l < n_layers - 1:
            v = jax.nn.gelu(v)
        hidden.append(v)
    out = mlp(params["proj"], v)[:, 0]
    return (out, hidden) if return_hidden else out


def make_loss(u0: Array, u_t: Array, batch: int = 32):
    """Relative L2 loss on random minibatches of ``(u0, u(T))`` pairs."""

    def loss(params, key):
        idx = jax.random.choice(key, u0.shape[0], (batch,), replace=False)
        pred = jax.vmap(lambda a: apply(params, a))(u0[idx])
        y = u_t[idx]
        return jnp.mean(jnp.linalg.norm(pred - y, axis=-1) / jnp.linalg.norm(y, axis=-1))

    return loss
