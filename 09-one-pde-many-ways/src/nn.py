"""The few neural-network pieces every learned method shares.

Pure JAX, no NN framework, as in the rest of the repository: an MLP is a list
of ``(W, b)`` pairs and every model's parameters are a pytree of such lists.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

Layer = tuple[Array, Array]
MLPParams = list[Layer]


def init_linear(key: Array, n_in: int, n_out: int, scale: float = 1.0) -> Layer:
    """Glorot-uniform weights, zero bias."""
    lim = scale * jnp.sqrt(6.0 / (n_in + n_out))
    return jax.random.uniform(key, (n_in, n_out), minval=-lim, maxval=lim), jnp.zeros(n_out)


def init_mlp(key: Array, sizes: list[int]) -> MLPParams:
    """MLP with layer widths ``sizes`` (input first, output last)."""
    keys = jax.random.split(key, len(sizes) - 1)
    return [init_linear(k, a, b) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def linear(layer: Layer, x: Array) -> Array:
    w, b = layer
    return x @ w + b


def mlp(params: MLPParams, x: Array, act: Callable[[Array], Array] = jax.nn.gelu) -> Array:
    """Apply an MLP; the last layer is linear."""
    for layer in params[:-1]:
        x = act(linear(layer, x))
    return linear(params[-1], x)


def layer_norm(x: Array, eps: float = 1e-5) -> Array:
    """LayerNorm without learned scale and shift (enough at this size)."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps)


def count_params(params) -> int:
    return sum(int(p.size) for p in jax.tree_util.tree_leaves(params))


def train(
    loss_fn: Callable[[object, Array], Array],
    params,
    key: Array,
    steps: int,
    lr: float = 1e-3,
    log_every: int = 100,
) -> tuple[object, list[float], float]:
    """Adam with cosine learning-rate decay.

    Args:
        loss_fn: ``loss_fn(params, key) -> scalar``. The key lets the loss draw
            its own minibatch or collocation points.
        params: Initial parameters (any pytree).
        key: PRNG key.
        steps: Number of optimizer steps.
        lr: Peak learning rate.
        log_every: Record the loss every this many steps.

    Returns:
        ``(params, losses, seconds)``: trained parameters, the recorded loss
        curve and the wall-clock training time (compilation included).
    """
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=1e-2))
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    t0 = time.perf_counter()
    for i in range(steps):
        key, sub = jax.random.split(key)
        params, state, loss = step(params, state, sub)
        if i % log_every == 0 or i == steps - 1:
            losses.append(float(loss))
    jax.block_until_ready(params)
    return params, losses, time.perf_counter() - t0


def timed(fn: Callable, *args, repeats: int = 3) -> tuple[object, float]:
    """Run ``fn(*args)`` once to compile, then return the result and the best of
    ``repeats`` wall-clock times in seconds."""
    out = jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return out, best
