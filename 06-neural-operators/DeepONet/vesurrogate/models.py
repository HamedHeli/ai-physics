"""Two surrogates of the map  theta = (phi, delta/phi, alpha, g_1..3, tau0)  ->  sigma(t)/mu.

Both "learn the function": the input is a handful of numbers, the output is a whole
curve sigma(t), queried at any time t.

1. DeepONet (black box, Lu et al. 2021)
       log sigma(t)/mu  ~=  sum_j  branch_j(theta) * trunk_j(u(t))
   branch: MLP on theta; trunk: MLP on a canonical time coordinate u(t) (linear in
   the ramp, logarithmic in the hold). Learns elastic AND viscous behaviour from data.

2. QLV-DeepONet (physics-structured, recommended)
       sigma_E(t)/mu = exp(MLP(log d(t), alpha, phi))             <- learned
       sigma(t)/mu   = QLV_convolution(sigma_E, g, tau)            <- exact, qlv.py
   Equivalently sigma = B_0(t) + sum_k g_k B_k(t): a DeepONet whose trunk basis
   functions B_k are produced by physics and whose branch coefficients are (1, g_k).
   Only the elastic indentation function (the thing Abaqus is slow at) is learned.

Pure JAX, params are pytrees of (W, b) lists, like the rest of the course repo.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import qlv


# ---------------------------------------------------------------------- MLP basics
def mlp_init(key, sizes, scale=1.0):
    params = []
    for k, (m, n) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (m, n)) * scale * np.sqrt(2.0 / (m + n))
        params.append((W, jnp.zeros(n)))
    return params


def mlp_apply(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def features(theta):
    """Normalise theta -> roughly O(1) inputs."""
    phi, d, alpha, g, tau0 = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3:6], theta[..., 6]
    return jnp.concatenate([((phi - 6.5) / 2.5)[..., None], (4 * d - 1)[..., None],
                            ((alpha - 5) / 3)[..., None], 2 * g - 1,
                            (jnp.log10(tau0) + 0.65)[..., None] * 1.5], axis=-1)


def canonical_u(t, tau0):
    """0..0.2 along the ramp (t/tau0), 0.2..1 along log10(t - tau0) in [-3, 1]."""
    hold = 0.2 + 0.8 * (jnp.log10(jnp.maximum(t - tau0, 1e-3)) + 3.0) / 4.0
    return jnp.where(t <= tau0, 0.2 * t / tau0, hold)


# ---------------------------------------------------------------------- DeepONet
P_LATENT = 64


def deeponet_init(key):
    k1, k2 = jax.random.split(key)
    return dict(branch=mlp_init(k1, [7, 128, 128, 128, P_LATENT]),
                trunk=mlp_init(k2, [1, 64, 64, 64, P_LATENT]),
                bias=jnp.zeros(()))


def deeponet_log(params, theta, t):
    """log(sigma/mu) at times t (t > 0)."""
    b = mlp_apply(params["branch"], features(theta))                    # (P,)
    tr = mlp_apply(params["trunk"], canonical_u(t, theta[6])[:, None])  # (n, P)
    return tr @ b / np.sqrt(P_LATENT) + params["bias"]


# ---------------------------------------------------------------------- QLV-DeepONet
def qlv_init(key):
    return dict(elastic=mlp_init(key, [3, 64, 64, 64, 1]))


def elastic_fn(params, d, alpha, phi):
    """Learned sigma_E/mu(d, alpha, phi), exactly 0 at d = 0.

    The MLP works in log-log space, log sigma_E = MLP(log d, alpha, phi), because
    indentation laws are close to power laws in depth (linear: exponent 1, Hertz-like
    stiffening: > 1). A pure power law is then just a linear function for the MLP.
    """
    safe = jnp.maximum(d, 1e-6)
    x = jnp.stack([jnp.log(safe / 0.2), (alpha - 5) / 3 * jnp.ones_like(d),
                   (phi - 6.5) / 2.5 * jnp.ones_like(d)], axis=-1)
    return jnp.where(d > 1e-6, jnp.exp(mlp_apply(params["elastic"], x)[..., 0]), 0.0)


def qlv_sigma(params, theta, t):
    """sigma/mu on an increasing time vector t that starts at 0."""
    phi, d, alpha, g, tau0 = theta[0], theta[1], theta[2], theta[3:6], theta[6]
    d_t = d * jnp.clip(t / tau0, 0.0, 1.0)
    se = elastic_fn(params, d_t, alpha, phi)
    return qlv.qlv_response(se, t, g)


def qlv_log(params, theta, t):
    """log(sigma/mu) at times t[1:] (t[0] must be 0)."""
    return jnp.log(jnp.maximum(qlv_sigma(params, theta, t)[1:], 1e-12))


MODELS = {
    "DeepONet": dict(init=deeponet_init, log=lambda p, th, t: deeponet_log(p, th, t[1:])),
    "QLV-DeepONet": dict(init=qlv_init, log=qlv_log),
}
