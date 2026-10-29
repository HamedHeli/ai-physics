"""Method 7 (topic 06/07): a Transolver-style transformer [Wu+ 2024].

What is learned: the same solution operator as the FNO,
``G: u(., 0) -> u(., T)``, but the global mixing is attention instead of an
FFT, so nothing assumes a regular grid.

Each grid point is a token with features ``(u0(x), sin x, cos x)``. Plain
self-attention between all ``n`` tokens costs ``O(n^2)``. Transolver's
**Physics-Attention** instead runs attention among ``M << n`` *slices*:

::

    w_i   = softmax_m( Linear(x_i) / tau )       soft assignment of point i to M slices   (n, M)
    z_m   = sum_i w_im x_i / sum_i w_im           one token per slice ("physical state")  (M, C)
    z'    = MultiHeadSelfAttention(z)             attention among M slice tokens
    x_i' += sum_m w_im z'_m                       broadcast back to the points            (n, C)

followed by the usual LayerNorm + MLP block. Points that are in a similar
physical state (here: on the steep front, on a plateau, in a rising part) end
up in the same slice wherever they are, which is what the slice-weight figure
in the notebook shows. Cost is ``O(n M)``, linear in the number of points.

Simplifications versus the paper, to keep it small: one set of slices per
head, no learned temperature per head, LayerNorm without learned affine
parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .nn import init_linear, init_mlp, layer_norm, linear, mlp


def init_params(key: Array, width: int = 64, slices: int = 16, heads: int = 4, layers: int = 3) -> dict:
    keys = jax.random.split(key, 2 + 6 * layers)
    blocks = []
    for l in range(layers):
        k = keys[2 + 6 * l: 8 + 6 * l]
        blocks.append({
            "in": init_linear(k[0], width, width),  # features fed to the slices
            "slice": init_linear(k[1], width // heads, slices),  # per head: point -> slice logits
            "qkv": init_linear(k[2], width // heads, 3 * width // heads),
            "out": init_linear(k[3], width, width),
            "mlp": init_mlp(k[4], [width, 2 * width, width]),
        })
    return {
        "embed": init_mlp(keys[0], [3, width, width]),
        "blocks": blocks,
        "head": init_mlp(keys[1], [width, width, 1]),
    }


def physics_attention(block: dict, x: Array, heads: int, tau: float = 0.5) -> tuple[Array, Array]:
    """One Physics-Attention layer. ``x``: ``(n, C)``.

    Returns the update ``(n, C)`` and slice weights ``(heads, n, M)``.
    """
    n, c = x.shape
    d = c // heads
    f = linear(block["in"], x).reshape(n, heads, d).transpose(1, 0, 2)  # (H, n, d)
    w = jax.nn.softmax(linear(block["slice"], f) / tau, axis=-1)  # (H, n, M)
    z = jnp.einsum("hnm,hnd->hmd", w, f) / (w.sum(1)[..., None] + 1e-5)  # (H, M, d)
    q, k, v = jnp.split(linear(block["qkv"], z), 3, axis=-1)  # (H, M, d) each
    att = jax.nn.softmax(jnp.einsum("hmd,hpd->hmp", q, k) / jnp.sqrt(d), axis=-1)
    z = jnp.einsum("hmp,hpd->hmd", att, v)
    out = jnp.einsum("hnm,hmd->hnd", w, z).transpose(1, 0, 2).reshape(n, c)
    return linear(block["out"], out), w


def apply(params: dict, u0: Array, heads: int = 4, return_slices: bool = False):
    """``u(., T)`` from ``u0`` of shape ``(n,)``. Any ``n`` (or point set) works.

    ``heads`` must match :func:`init_params`."""
    n = u0.shape[0]
    xg = jnp.arange(n) * (2 * jnp.pi / n)
    x = mlp(params["embed"], jnp.stack([u0, jnp.sin(xg), jnp.cos(xg)], -1))
    all_w = []
    for block in params["blocks"]:
        dx, w = physics_attention(block, layer_norm(x), heads)
        x = x + dx
        x = x + mlp(block["mlp"], layer_norm(x))
        all_w.append(w)
    out = mlp(params["head"], layer_norm(x))[:, 0]
    return (out, all_w) if return_slices else out


def make_loss(u0: Array, u_t: Array, batch: int = 32):
    """Relative L2 loss on random minibatches, as for the FNO."""

    def loss(params, key):
        idx = jax.random.choice(key, u0.shape[0], (batch,), replace=False)
        pred = jax.vmap(lambda a: apply(params, a))(u0[idx])
        y = u_t[idx]
        return jnp.mean(jnp.linalg.norm(pred - y, axis=-1) / jnp.linalg.norm(y, axis=-1))

    return loss
