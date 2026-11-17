"""Training, evaluation and inversion through a trained (differentiable) surrogate."""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .models import MODELS


# ---------------------------------------------------------------------- training
def train(name, data, seed=0, steps=4000, lr=2e-3, verbose=True):
    m = MODELS[name]
    params = m["init"](jax.random.PRNGKey(seed))
    theta, t, y = map(jnp.asarray, (data["theta"], data["t"], data["y"]))
    logy = jnp.log(jnp.maximum(y[:, 1:], 1e-12))
    batch_log = jax.vmap(m["log"], in_axes=(None, 0, 0))

    def loss_fn(p):
        return jnp.mean((batch_log(p, theta, t) - logy) ** 2)

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, 200, steps, lr * 1e-3)
    opt = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, gr = jax.value_and_grad(loss_fn)(p)
        u, s = opt.update(gr, s, p)
        return optax.apply_updates(p, u), s, l

    t0 = time.time()
    for i in range(steps):
        params, state, l = step(params, state)
        if verbose and (i % 1000 == 0 or i == steps - 1):
            print(f"  [{name}] step {i:5d}  loss {float(l):.3e}")
    return params, time.time() - t0


def predict(name, params, theta, t):
    """sigma/mu on t (t[0]=0 -> returns 0 there)."""
    lg = MODELS[name]["log"](params, theta, t)
    return jnp.concatenate([jnp.zeros(1), jnp.exp(lg)])


def rel_l2(name, params, data):
    f = jax.jit(jax.vmap(lambda th, t: predict(name, params, th, t)))
    pred = np.asarray(f(jnp.asarray(data["theta"]), jnp.asarray(data["t"])))
    err = np.linalg.norm(pred - data["y"], axis=1) / np.linalg.norm(data["y"], axis=1)
    return err, pred


# ---------------------------------------------------------------------- inversion
# unknowns z = [log mu, a, r1, r2, r3]; alpha = 2 + 6 sigmoid(a), g_k = sigmoid(r_k)
def z_to_phys(z):
    return jnp.concatenate([z[:1], (2 + 6 * jax.nn.sigmoid(z[1]))[None], jax.nn.sigmoid(z[2:5])])


def phys_to_z(p):
    a = jnp.clip((p[1] - 2) / 6, 1e-4, 1 - 1e-4)
    g = jnp.clip(p[2:5], 1e-4, 1 - 1e-4)
    return jnp.concatenate([p[:1], jnp.log(a / (1 - a))[None], jnp.log(g / (1 - g))])


def stack_curves(curves):
    """list of dicts (phi, d, tau0, t, y) with equal-length t -> dict of stacked arrays."""
    return {k: jnp.asarray(np.stack([np.asarray(c[k], np.float32) for c in curves]))
            for k in ("phi", "d", "tau0", "t", "y")}


def _residuals(name, params, phys, C, fixed_alpha=None):
    """phys = [log mu, alpha, g1, g2, g3]; C: stacked curves (see stack_curves)."""
    logmu, alpha, g = phys[0], phys[1], phys[2:5]
    if fixed_alpha is not None:
        alpha = jnp.asarray(fixed_alpha, jnp.float32)

    def one(phi, d, tau0, t, y):
        theta = jnp.concatenate([jnp.stack([phi, d, alpha]), g, tau0[None]])
        return logmu + MODELS[name]["log"](params, theta, t) - jnp.log(y[1:])
    return jax.vmap(one)(C["phi"], C["d"], C["tau0"], C["t"], C["y"]).ravel()


@functools.lru_cache(maxsize=None)
def _inverter(name, steps, lr, fixed_alpha):
    opt = optax.adam(lr)

    def loss(z, params, C):
        r = _residuals(name, params, z_to_phys(z), C, fixed_alpha)
        return jnp.mean(r ** 2)

    def run(z, params, C):
        s = opt.init(z)

        def body(carry, _):
            z, s = carry
            g = jax.grad(loss)(z, params, C)
            u, s = opt.update(g, s, z)
            return (optax.apply_updates(z, u), s), None
        (z, _), _ = jax.lax.scan(body, (z, s), None, length=steps)
        return z, loss(z, params, C)
    return jax.jit(jax.vmap(run, in_axes=(0, None, None)))


def invert(name, params, curves, n_starts=16, steps=800, lr=0.05, seed=0, fixed_alpha=None):
    """Recover (mu, alpha, g1..3) by gradient descent through the surrogate.

    Multi-start Adam in an unconstrained space. `curves` is a list of dicts with
    phi, d, tau0, t (starting at 0) and y (same units as mu). All curves in one call
    must share the same number of time points. Returns best physical params
    [log mu, alpha, g1, g2, g3], final loss and wall time (incl. compile on 1st call).
    """
    C = stack_curves(curves)
    rng = np.random.default_rng(seed)
    y_scale = float(np.log(np.median(np.asarray(C["y"])[:, 1:])))
    z0 = np.column_stack([np.full(n_starts, y_scale), rng.normal(0, 1, n_starts),
                          rng.normal(-1, 1, (n_starts, 3))]).astype(np.float32)
    t0 = time.time()
    zs, ls = _inverter(name, steps, lr, fixed_alpha)(jnp.asarray(z0), params, C)
    ls.block_until_ready()
    wall = time.time() - t0
    best = int(jnp.argmin(ls))
    return np.asarray(z_to_phys(zs[best])), float(ls[best]), wall


def laplace_cov(name, params, phys, curves, noise=0.01, fixed_alpha=None):
    """Gauss-Newton (Laplace) covariance of [log mu, alpha, g1, g2, g3] assuming
    i.i.d. relative measurement noise `noise` (1% default, a typical load cell)."""
    C = stack_curves(curves)
    J = jax.jacfwd(lambda p: _residuals(name, params, p, C, fixed_alpha))(jnp.asarray(phys, jnp.float32))
    J = np.asarray(J, np.float64)
    keep = [0, 2, 3, 4] if fixed_alpha is not None else [0, 1, 2, 3, 4]
    J = J[:, keep]
    cov = noise ** 2 * np.linalg.pinv(J.T @ J)
    return cov, keep
