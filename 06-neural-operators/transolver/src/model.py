"""Transolver (Wu et al., 2024, "Transolver: A Fast Transformer Solver for PDEs
on General Geometries", ICML 2024, arXiv:2402.02366).

Architecture (paper Sec. 3, Fig. 3, and the official ``Physics_Attention_Irregular_Mesh``)
-----------------------------------------------------------------------------------------
Every point of the mesh or point cloud is one row. Nothing assumes a grid::

    input       features of point i: (x_i, y_i, d_i)                    (N, 3)
    embed       x = Linear(GELU(Linear(features)))   (pointwise MLP)    (N, C)

    Transolver block l = 0 .. L-1:
        x = x + PhysicsAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Physics-Attention, for each head h (C = H * d channels, split into H heads of d):
        f_i    = x_i W_fx                    point features ("values" to aggregate) (H, N, d)
        g_i    = x_i W_x                     point features used to pick a slice    (H, N, d)
        w_i    = softmax_m( g_i W_s / tau_h )  slice weights, rows sum to 1         (H, N, M)
        z_m    = sum_i w_im f_i / sum_i w_im   one token per slice                  (H, M, d)
        z'     = softmax(q k^T / sqrt d) v, with q, k, v = z W_q, z W_k, z W_v       (H, M, d)
        f'_i   = sum_m w_im z'_m               deslice: back to the points          (H, N, d)
        out    = concat_h(f') W_o + b_o                                             (N, C)

    decode      u_i = LayerNorm(x_i) W_out + b_out  (pointwise)          (N,)

The four steps *slice -> tokens -> attention -> deslice* are the paper's
Physics-Attention. Points anywhere in the domain that are in a similar physical
state get large weights for the same slice, so a slice token is "the average
state of all points that look like this". Attention then runs among the ``M``
tokens, not the ``N`` points: ``O(N M d + M^2 d)`` instead of ``O(N^2 d)``.

Why it is an operator (paper Sec. 3, Theorem 3.1): ``sum_i w_im f_i / sum_i w_im``
is a Monte-Carlo estimate of an integral over the domain, so the model is a
learnable kernel integral operator ``u(x) -> int kappa(x, y) u(y) dy`` with a
low-rank kernel built from the slices, just like FNO's ``K`` but without an FFT.
More points give a better estimate of the same integrals; the weights do not
depend on ``N``.

Sizes here are chosen to fit on screen: ``C = 16`` channels, ``H = 2`` heads of
``d = 8``, ``M = 8`` slices, ``L = 3`` blocks. The paper uses ``C = 128 to 256``,
``H = 8``, ``M = 32 to 64`` and ``L = 8``.

Design notes
------------
* Pure JAX, no NN framework, matching the rest of this repository: parameters
  are a pytree of arrays, every linear map is ``(W, b)``.
* ``tau_h`` is learned per head and starts at 0.5, as in the paper code.
  Transolver++ (Luo et al., 2025) replaces it by a per-point temperature; see
  the notebook's last section.
* The paper code adds a learned "placeholder" vector after the embedding and
  uses dropout; both are left out.
* :func:`apply_with_trace` returns every intermediate tensor. That trace is what
  the notebook visualises.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class TransolverConfig(NamedTuple):
    """Static hyperparameters.

    Attributes:
        in_dim: Input features per point (``x``, ``y``, distance to the hole).
        width: Channels ``C`` of the hidden state of every point.
        heads: Attention heads ``H``; each head has ``d = C / H`` channels and its own slices.
        slices: Number of slices (physical-state tokens) ``M`` per head.
        layers: Number of Transolver blocks ``L``.
        mlp_ratio: Hidden width of the per-block MLP is ``mlp_ratio * C``.
    """

    in_dim: int = 3
    width: int = 16
    heads: int = 2
    slices: int = 8
    layers: int = 3
    mlp_ratio: int = 2

    @property
    def head_dim(self) -> int:
        return self.width // self.heads


def _dense(key: Array, i: int, o: int, bias: bool = True):
    """``W ~ N(0, 1/fan_in)``, ``b = 0``."""
    W = jax.random.normal(key, (i, o)) / jnp.sqrt(i)
    return (W, jnp.zeros((o,))) if bias else W


def _ln_params(c: int):
    return {"gamma": jnp.ones((c,)), "beta": jnp.zeros((c,))}


def init_params(key: Array, cfg: TransolverConfig) -> dict:
    """Initialise the embedding, the blocks and the decoder.

    ``W_s`` (point -> slice logits) is orthogonal, as in the paper code; all
    other weights are ``N(0, 1/fan_in)``.

    Returns:
        ``{"embed": [(W, b), (W, b)], "blocks": [...], "ln_out": ..., "out": (W, b)}``
    """
    C, d, M = cfg.width, cfg.head_dim, cfg.slices
    k_e1, k_e2, k_out, k_blocks = jax.random.split(key, 4)
    blocks = []
    for bk in jax.random.split(k_blocks, cfg.layers):
        ks = jax.random.split(bk, 9)
        blocks.append({
            "ln1": _ln_params(C),
            "W_fx": _dense(ks[0], C, C),
            "W_x": _dense(ks[1], C, C),
            "W_s": (jax.nn.initializers.orthogonal()(ks[2], (d, M)), jnp.zeros((M,))),
            "log_tau": jnp.full((cfg.heads,), jnp.log(0.5)),
            "W_q": _dense(ks[3], d, d, bias=False),
            "W_k": _dense(ks[4], d, d, bias=False),
            "W_v": _dense(ks[5], d, d, bias=False),
            "W_o": _dense(ks[6], C, C),
            "ln2": _ln_params(C),
            "mlp": [_dense(ks[7], C, cfg.mlp_ratio * C), _dense(ks[8], cfg.mlp_ratio * C, C)],
        })
    return {
        "embed": [_dense(k_e1, cfg.in_dim, 2 * C), _dense(k_e2, 2 * C, C)],
        "blocks": blocks,
        "ln_out": _ln_params(C),
        "out": _dense(k_out, C, 1),
    }


def layer_norm(p: dict, x: Array, eps: float = 1e-5) -> Array:
    """Normalise every row (point) to mean 0, variance 1 over its channels, then scale and shift."""
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * p["gamma"] + p["beta"]


def linear(p, x: Array) -> Array:
    W, b = p
    return x @ W + b


def split_heads(x: Array, heads: int) -> Array:
    """``(N, C) -> (H, N, C/H)``: channels ``h*d .. (h+1)*d - 1`` belong to head ``h``."""
    n, c = x.shape
    return x.reshape(n, heads, c // heads).transpose(1, 0, 2)


def physics_attention(p: dict, x: Array, heads: int) -> dict:
    """One Physics-Attention layer on a normalised input ``x`` of shape ``(N, C)``.

    Returns a dict with every intermediate (shapes for ``H`` heads, ``M`` slices,
    ``d = C/H``): ``f (H,N,d)``, ``g (H,N,d)``, ``logits (H,N,M)``, ``tau (H,)``,
    ``w (H,N,M)``, ``mass (H,M)``, ``z (H,M,d)``, ``q, k, v (H,M,d)``,
    ``attn (H,M,M)``, ``z_out (H,M,d)``, ``f_out (H,N,d)``, ``out (N,C)``.
    """
    f = split_heads(linear(p["W_fx"], x), heads)                      # (H, N, d)
    g = split_heads(linear(p["W_x"], x), heads)                       # (H, N, d)
    tau = jnp.exp(p["log_tau"])                                       # (H,)
    logits = linear(p["W_s"], g)                                      # (H, N, M)
    w = jax.nn.softmax(logits / tau[:, None, None], axis=-1)          # 1. slice: each row sums to 1
    mass = w.sum(1)                                                   # (H, M) how many points per slice
    z = jnp.einsum("hnm,hnd->hmd", w, f) / (mass[..., None] + 1e-5)   # 2. tokens: weighted averages
    q, k, v = z @ p["W_q"], z @ p["W_k"], z @ p["W_v"]                # (H, M, d)
    attn = jax.nn.softmax(q @ k.transpose(0, 2, 1) / jnp.sqrt(q.shape[-1]), axis=-1)
    z_out = attn @ v                                                  # 3. attention among M tokens
    f_out = jnp.einsum("hnm,hmd->hnd", w, z_out)                      # 4. deslice: back to points
    n = x.shape[0]
    out = linear(p["W_o"], f_out.transpose(1, 0, 2).reshape(n, -1))   # (N, C)
    return dict(f=f, g=g, logits=logits, tau=tau, w=w, mass=mass, z=z, q=q, k=k, v=v, attn=attn,
                z_out=z_out, f_out=f_out, out=out)


class Trace(NamedTuple):
    """Every intermediate tensor of one forward pass.

    Attributes:
        inputs: Point features, ``(N, in_dim)``.
        embedded: ``x`` after the embedding MLP, ``(N, C)``.
        blocks: One dict per block with ``x_in``, ``ln1`` (the normalised input),
            everything from :func:`physics_attention`, ``x_mid`` (after the
            attention residual), ``mlp_out`` and ``x_out``.
        output: Predicted temperature at every point, ``(N,)``.
    """

    inputs: Array
    embedded: Array
    blocks: list
    output: Array


def apply_with_trace(params: dict, features: Array, cfg: TransolverConfig) -> Trace:
    """Run Transolver on one point cloud ``(N, in_dim)`` and keep every tensor. Any ``N`` works."""
    (W1, b1), (W2, b2) = params["embed"]
    x = jax.nn.gelu(features @ W1 + b1) @ W2 + b2
    embedded = x
    blocks = []
    for p in params["blocks"]:
        rec = {"x_in": x, "ln1": layer_norm(p["ln1"], x)}
        rec.update(physics_attention(p, rec["ln1"], cfg.heads))
        x = x + rec["out"]
        rec["x_mid"] = x
        (Wa, ba), (Wb, bb) = p["mlp"]
        rec["mlp_out"] = jax.nn.gelu(layer_norm(p["ln2"], x) @ Wa + ba) @ Wb + bb
        x = x + rec["mlp_out"]
        rec["x_out"] = x
        blocks.append(rec)
    out = linear(params["out"], layer_norm(params["ln_out"], x))[:, 0]
    return Trace(features, embedded, blocks, out)


def apply(params: dict, features: Array, cfg: TransolverConfig) -> Array:
    """Predicted temperature at every point, ``(N, in_dim) -> (N,)``."""
    return apply_with_trace(params, features, cfg).output
