"""Synthetic ramp-and-hold indentation curves (stand-in for more Abaqus runs).

Truth = exact QLV convolution (paper eq. 3 + 8) of the calibrated stand-in elastic law
(qlv.elastic_standin). Everything produced here is SYNTHETIC. Targets are sigma/mu, so
mu is a pure scale and is recovered separately in the inverse problem.

Parameter vector theta (per curve), in this order:
    phi [mm], d = delta/phi, alpha, g1, g2, g3, tau0 [s]
Prony relaxation times are fixed at the paper's 0.1, 1, 10 s.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import qlv

N_RAMP, N_HOLD = 24, 104
FINE = 8  # truth uses an 8x finer ramp than the model grid

RANGES = dict(phi=(4.0, 9.0), d=(0.05, 0.5), alpha=(2.0, 8.0), g=(0.0, 1.0),
              log_tau0=(np.log10(0.05), np.log10(1.0)))
OOD_RANGES = dict(RANGES, log_tau0=(np.log10(1.0), np.log10(3.0)), d=(0.5, 0.7))


def sample_theta(rng, n, ranges=RANGES):
    u = lambda lo_hi, size=None: rng.uniform(*lo_hi, size=(n,) if size is None else (n, size))
    phi, d, alpha = u(ranges["phi"]), u(ranges["d"]), u(ranges["alpha"])
    g = u(ranges["g"], 3)
    tau0 = 10 ** u(ranges["log_tau0"])
    return np.column_stack([phi, d, alpha, g, tau0]).astype(np.float32)


def depth_history(t, d, tau0):
    """Linear displacement ramp to d at tau0, then hold (Abaqus default amplitude)."""
    return d * jnp.clip(t / tau0, 0.0, 1.0)


def truth_curve(theta, K, n0):
    """Return (t, sigma/mu) on the model grid, t[0] = 0."""
    phi, d, alpha, g, tau0 = theta[0], theta[1], theta[2], theta[3:6], theta[6]
    tf = qlv.time_grid(tau0, n_ramp=N_RAMP * FINE, n_hold=N_HOLD)
    se = qlv.elastic_standin(depth_history(tf, d, tau0), alpha, phi, K, n0)
    sig = qlv.qlv_response(se, tf, g)
    keep = jnp.concatenate([jnp.arange(0, N_RAMP * FINE + 1, FINE),
                            N_RAMP * FINE + 1 + jnp.arange(N_HOLD)])
    return tf[keep], sig[keep]


def make_dataset(rng, n, K, n0, ranges=RANGES, noise=0.0):
    theta = sample_theta(rng, n, ranges)
    t, y = jax.vmap(lambda th: truth_curve(th, K, n0))(jnp.asarray(theta))
    y = np.asarray(y)
    if noise > 0:
        y = y * (1 + noise * rng.standard_normal(y.shape))
    return dict(theta=theta, t=np.asarray(t), y=y.astype(np.float32))
