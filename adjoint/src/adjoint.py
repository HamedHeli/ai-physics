"""Gradients of ``J(theta)`` four ways, so they can be compared.

1. :func:`finite_difference_gradient` -- perturb each parameter, re-simulate.
   Cost: ``2 P`` simulations for ``P`` parameters.
2. :func:`discrete_adjoint_gradient` -- the hand-written discrete adjoint.
   Cost: one forward + one backward pass, whatever ``P`` is.
3. :func:`checkpointed_adjoint_gradient` -- the same, but storing only every
   ``k``-th state and recomputing the rest during the backward pass
   (trades memory for compute; this is what Allen et al. 2022 do).
4. :func:`continuous_adjoint_gradient` -- the continuous adjoint ODE,
   re-integrating the state *backward in time* instead of storing it
   (the Neural ODE trick, Chen et al. 2018). Almost no memory, but drag is
   dissipative forwards, so running it backwards amplifies error.

and :func:`jax_gradient`, reverse-mode autodiff of the same simulator, which
is the discrete adjoint generated automatically.

Discrete adjoint derivation
---------------------------
Write one Euler step as ``s_{k+1} = F(s_k, u_k)`` with ``s = (x, v)``. Its
Jacobians are

    dF/ds = [[ I,  dt I ],          A_k = I - dt c ( |v_k| I + v_k v_k^T / |v_k| )
             [ 0,  A_k  ]]
    dF/du = [[ 0 ], [ dt I ]]

Define the adjoint ``lambda_k = dJ/ds_k``. At the end it is ``dJ/ds_N``, and
the chain rule run backwards gives

    lambda_x_k = lambda_x_{k+1}
    lambda_v_k = dt * lambda_x_{k+1} + A_k^T lambda_v_{k+1}
    dJ/du_k    = dt * lambda_v_{k+1}            (summed within each segment)
    dJ/dv0     = lambda_v_0

``A_k`` contains ``v_k``, so the backward pass needs the forward velocities:
that is the memory cost of an adjoint.
"""

from __future__ import annotations

import numpy as np

from .ball import GRAVITY, Problem, objective, simulate, step, thrust_at


def drag_jacobian(v: np.ndarray, prob: Problem) -> np.ndarray:
    """``A = d v_{k+1} / d v_k`` for the Euler step, shape ``(2, 2)``."""
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return np.eye(2)
    return np.eye(2) - prob.dt * prob.drag * (speed * np.eye(2) + np.outer(v, v) / speed)


def terminal_adjoint(xN: np.ndarray, prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    """``(dJ/dx_N, dJ/dv_N)``."""
    return 2.0 * (xN - prob.target), np.zeros(2)


def _reg_grad(U: np.ndarray, prob: Problem) -> np.ndarray:
    return prob.reg * 2.0 * U / U.size


def discrete_adjoint_gradient(v0, U, prob: Problem, return_history: bool = False):
    """Hand-written discrete adjoint.

    Args:
        v0: Launch velocity ``(2,)``.
        U: Thrust table ``(K, 2)``.
        prob: Fixed settings.
        return_history: Also return every ``lambda_k`` (for plotting).

    Returns:
        ``(dJ/dv0, dJ/dU)`` and, if requested, a dict with the forward states
        ``X, V`` and the adjoints ``LX, LV`` of shape ``(N + 1, 2)``.
    """
    X, V = simulate(v0, U, prob)  # store the whole forward trajectory
    lx, lv = terminal_adjoint(X[-1], prob)
    gU = _reg_grad(U, prob)
    LX, LV = np.zeros_like(X), np.zeros_like(V)
    LX[-1], LV[-1] = lx, lv
    for k in reversed(range(prob.n_steps)):
        gU[k * prob.n_segments // prob.n_steps] += prob.dt * lv
        lx, lv = lx, prob.dt * lx + drag_jacobian(V[k], prob).T @ lv
        LX[k], LV[k] = lx, lv
    if return_history:
        return lv, gU, {"X": X, "V": V, "LX": LX, "LV": LV}
    return lv, gU


def checkpointed_adjoint_gradient(v0, U, prob: Problem, every: int):
    """Discrete adjoint storing only every ``every``-th state.

    During the backward pass each segment of ``every`` steps is recomputed
    from its checkpoint before being walked backwards.

    Returns:
        ``(dJ/dv0, dJ/dU, stats)`` with ``stats = {"stored_states",
        "forward_steps"}`` to compare memory and compute with the plain
        adjoint (which stores ``N + 1`` states and runs ``N`` forward steps).
    """
    N = prob.n_steps
    x, v = np.zeros(2), np.asarray(v0, float)
    checkpoints = {0: (x, v)}
    for k in range(N):
        x, v = step(x, v, thrust_at(U, k, prob), prob)
        if (k + 1) % every == 0:
            checkpoints[k + 1] = (x, v)
    forward_steps = N
    lx, lv = terminal_adjoint(x, prob)
    gU = _reg_grad(U, prob)
    for start in sorted(checkpoints, reverse=True):
        if start == N:
            continue
        end = min(start + every, N)
        # Recompute this segment's velocities from its checkpoint.
        xs, vs = checkpoints[start]
        seg_v = [vs]
        for k in range(start, end - 1):
            xs, vs = step(xs, vs, thrust_at(U, k, prob), prob)
            seg_v.append(vs)
        forward_steps += end - 1 - start
        for k in reversed(range(start, end)):
            gU[k * prob.n_segments // N] += prob.dt * lv
            lx, lv = lx, prob.dt * lx + drag_jacobian(seg_v[k - start], prob).T @ lv
    stats = {"stored_states": len(checkpoints) + every, "forward_steps": forward_steps}
    return lv, gU, stats


def continuous_adjoint_gradient(v0, U, prob: Problem, return_history: bool = False):
    """Continuous adjoint with the state re-integrated backward in time.

    Only the final state is kept from the forward pass. Then, walking from
    ``T`` back to 0, the state ODE is integrated backwards *together with*
    the adjoint ODE

        d lambda_x / dt = 0
        d lambda_v / dt = -lambda_x + c ( |v| lambda_v + v (v . lambda_v) / |v| )

    and ``dJ/du`` accumulates ``lambda_v dt``. If the backward state drifts
    from the true one, the gradient inherits the error.
    """
    X, V = simulate(v0, U, prob)
    x, v = X[-1].copy(), V[-1].copy()
    lx, lv = terminal_adjoint(x, prob)
    gU = _reg_grad(U, prob)
    Xb, Vb = np.zeros_like(X), np.zeros_like(V)
    Xb[-1], Vb[-1] = x, v
    dt, c = prob.dt, prob.drag
    for k in reversed(range(prob.n_steps)):
        u = thrust_at(U, k, prob)
        gU[k * prob.n_segments // prob.n_steps] += dt * lv
        speed = np.linalg.norm(v)
        dlv = -lx + c * (speed * lv + (v @ lv) * v / max(speed, 1e-12))
        lv = lv - dt * dlv
        # Undo one step of the state ODE (explicit Euler, backwards).
        x, v = x - dt * v, v - dt * (GRAVITY + u - c * speed * v)
        Xb[k], Vb[k] = x, v
    if return_history:
        return lv, gU, {"X": X, "V": V, "Xb": Xb, "Vb": Vb}
    return lv, gU


def finite_difference_gradient(v0, U, prob: Problem, eps: float = 1e-6):
    """Central differences: two simulations per parameter."""
    theta = np.concatenate([np.ravel(v0), np.ravel(U)])
    g = np.zeros_like(theta)

    def J(t):
        return objective(t[:2], t[2:].reshape(U.shape), prob)

    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        g[i] = (J(theta + e) - J(theta - e)) / (2 * eps)
    return g[:2], g[2:].reshape(U.shape)


def jax_gradient(v0, U, prob: Problem):
    """Reverse-mode autodiff of the same simulator (the automatic discrete adjoint)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    g = jnp.asarray(GRAVITY)

    def J(v0, U):
        def body(carry, k):
            x, v = carry
            u = U[k * prob.n_segments // prob.n_steps]
            speed = jnp.linalg.norm(v)
            return (x + prob.dt * v, v + prob.dt * (g + u - prob.drag * speed * v)), None

        (xN, _), _ = jax.lax.scan(body, (jnp.zeros(2), v0), jnp.arange(prob.n_steps))
        return jnp.sum((xN - prob.target) ** 2) + prob.reg * jnp.mean(U**2)

    gv, gU = jax.grad(J, argnums=(0, 1))(jnp.asarray(v0, float), jnp.asarray(U, float))
    return np.asarray(gv), np.asarray(gU)
