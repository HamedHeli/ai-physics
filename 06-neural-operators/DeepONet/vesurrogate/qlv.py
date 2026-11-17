"""Differentiable quasi-linear viscoelastic (QLV) layer for ramp-and-hold indentation.

Physics (Helisaz, Bacca & Chiao, Exp. Mech. 2022, eqs. 1, 3, 8):

    sigma(t) = integral_0^t G(t - s) d sigma_E(s)
    G(t)     = 1 + sum_k g_k exp(-t / tau_k)          (Prony series, eq. 8)

where sigma_E(delta) is the steady-state ("elastic") indentation stress. The paper
shows (Figs. 6, 7) that the relaxation is independent of indenter size phi and depth
delta, i.e. the response factorises into

    geometry/elastic part   sigma_E(delta/phi, phi, alpha) * mu     (needs FE: ~10 h per run)
    time/viscous part       the Prony convolution above              (exact, cheap)

This module implements the viscous part *exactly* and differentiably in JAX, so a
neural network only has to learn the elastic indentation function. For piecewise
linear sigma_E between grid times the recursion below is exact:

    h_k(t_{n+1}) = a h_k(t_n) + tau_k (1 - a) / dt * (sigma_E(t_{n+1}) - sigma_E(t_n)),
    a = exp(-dt / tau_k),       sigma = sigma_E + sum_k g_k h_k.

During the hold sigma_E is constant, so any (e.g. log-spaced) hold grid is exact.
"""
import jax
import jax.numpy as jnp
import numpy as np

# Values from Table 1 of the paper (Ecoflex 00-10, from uniaxial compression).
TABLE1 = dict(mu=3.98e-3,  # MPa (= 3.98 kPa)
              alpha=5.39, g=(0.33, 0.48, 0.13), tau=(0.1, 1.0, 10.0))
PRONY_TAUS = np.array(TABLE1["tau"])
SPECIMEN_H = 18.88   # mm, specimen height used in all of the paper's simulations
HOLD_TIME = 10.0     # s, hold step length in the paper's Abaqus model


def time_grid(tau0, n_ramp=24, n_hold=104, hold=HOLD_TIME, t_min=1e-3):
    """Canonical grid: n_ramp uniform ramp steps (t=0 included) + log-spaced hold.

    Returns times of length n_ramp + 1 + n_hold. Index 0 is t = 0 (sigma = 0).
    """
    ramp = tau0 * jnp.arange(n_ramp + 1) / n_ramp
    hold_pts = tau0 + jnp.logspace(np.log10(t_min), np.log10(hold), n_hold)
    return jnp.concatenate([ramp, hold_pts])


def qlv_response(sig_e, t, g, tau=PRONY_TAUS):
    """Exact Prony convolution of a piecewise-linear elastic history.

    sig_e: (n,) elastic stress at times t (sig_e[0] should be 0 at t=0)
    g, tau: (K,) Prony coefficients and relaxation times
    returns sigma: (n,)
    """
    tau = jnp.asarray(tau)
    dt = jnp.diff(t)
    ds = jnp.diff(sig_e)
    a = jnp.exp(-dt[:, None] / tau[None, :])
    safe_dt = jnp.where(dt > 0, dt, 1.0)
    b = jnp.where(dt[:, None] > 0, tau[None, :] * (1 - a) / safe_dt[:, None], 1.0)

    def step(h, inp):
        a_n, b_n, ds_n = inp
        h = a_n * h + b_n * ds_n
        return h, h

    h0 = jnp.zeros_like(tau, dtype=sig_e.dtype)
    _, hs = jax.lax.scan(step, h0, (a, b, ds))
    hs = jnp.concatenate([h0[None], hs], axis=0)       # (n, K)
    return sig_e + hs @ jnp.asarray(g)


def qlv_basis(sig_e, t, tau=PRONY_TAUS):
    """Return the K+1 'basis curves' [sigma_E, h_1, ..., h_K] so that
    sigma = basis[0] + sum_k g_k basis[k]. This is the paper's eq. (13): the
    response is *linear* in the g_k (a DeepONet with a physics trunk)."""
    eye = jnp.eye(len(tau))
    hs = jax.vmap(lambda e: qlv_response(sig_e, t, e, tau) - sig_e)(eye)
    return jnp.concatenate([sig_e[None], hs], axis=0)


# --------------------------------------------------------------------------------
# Stand-in elastic indentation law (used ONLY to make synthetic training data until
# more Abaqus runs arrive). It is NOT the true FE elastic response:
#
#   sigma_E / mu = K * (d / 0.2) ** (n0 + c (alpha - 5.39)) * Hayes(phi/H) / Hayes(4.95/H)
#
# a power law in normalised depth d = delta/phi (so K is the value at d = 0.2),
# a Hayes-style finite-thickness factor for the indenter size, and an exponent that
# grows with the Ogden stiffening parameter alpha. K and n0 are calibrated on the
# 4.95 mm / 1 mm Abaqus curve (run_all.py step 1); c = 0.08 is an ASSUMPTION (one
# Abaqus curve cannot tell how the shape changes with alpha). Replace this whole
# function by real Abaqus runs as soon as they exist.
# --------------------------------------------------------------------------------
STANDIN_C = 0.08


def hayes_factor(phi, H=SPECIMEN_H):
    r = phi / (2.0 * H)            # a/h
    return 1.0 + 1.33 * r + 1.33 * r ** 3


def elastic_standin(d, alpha, phi, K, n0, c=STANDIN_C):
    """sigma_E / mu as a function of normalised depth d = delta/phi (d >= 0)."""
    n = n0 + c * (alpha - TABLE1["alpha"])
    x = jnp.maximum(d / 0.2, 0.0)
    return K * jnp.where(x > 0, x ** n, 0.0) * hayes_factor(phi) / hayes_factor(4.95)
