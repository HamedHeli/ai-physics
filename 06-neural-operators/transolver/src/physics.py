"""Ground truth for the Transolver notebook: heat in an L-shaped plate with a cooling hole.

The problem
-----------
A thin L-shaped plate is heated uniformly (think of electronics bolted to a
bracket) and cooled along its whole outer edge and through a round cooling
pipe that goes through it. The steady temperature ``u(x, y)`` solves the
Poisson equation

    -(u_xx + u_yy) = f        in  Omega = L-shape minus the hole,
    u = 0                     on the outer edge and on the hole's edge,

with a constant heat source ``f``. ``Omega`` is the unit square without its
top-right quarter (the L), minus a disc of centre ``c`` and radius ``r``. The
hole changes from sample to sample, so the operator to learn is

    G:  geometry (where the hole is)  ->  temperature field u on that geometry.

This mirrors the *Elasticity* benchmark of the Transolver paper (Wu et al.,
2024, Sec. 4.1: a unit cell with a hole of varying shape, given as a point
cloud), and the domain is neither a rectangle nor a grid, which is exactly what
the FNO's FFT cannot handle.

Solver
------
Second-order finite differences (5-point Laplacian) on a ``65 x 65`` grid with
spacing ``h = 1/64``. Nodes outside the L or inside the hole are fixed at 0, the
rest are unknowns, and the sparse linear system is solved with SciPy. The hole's
edge is a staircase at this resolution, which is fine for a demo.

The point cloud
---------------
The model never sees the grid. Each sample is a set of points drawn uniformly
at random inside the plate (any number of points, in no particular order, not
on the grid), with the temperature there, interpolated bilinearly from the
solver's grid, as the target. Plain NumPy/SciPy: nothing here needs gradients.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator


class PlateParams(NamedTuple):
    """Physical and numerical constants.

    Attributes:
        n_grid: Grid nodes per side of the reference solver (``h = 1 / (n_grid - 1)``).
        source: Heat source ``f``. Chosen so the hottest point is about 1.
        r_min, r_max: Range of the hole radius.
        clearance: Minimum gap between the hole and the plate's edge.
    """

    n_grid: int = 65
    source: float = 35.0
    r_min: float = 0.08
    r_max: float = 0.16
    clearance: float = 0.06


def grid_xy(params: PlateParams = PlateParams()) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates ``X, Y`` of shape ``(n, n)``; ``X[i, j] = x_j``, ``Y[i, j] = y_i``."""
    s = np.linspace(0.0, 1.0, params.n_grid)
    return np.meshgrid(s, s)


def in_L(x, y):
    """True strictly inside the L-shape (unit square minus the top-right quarter)."""
    return (x > 0) & (x < 1) & (y > 0) & (y < 1) & ~((x >= 0.5) & (y >= 0.5))


def sample_hole(rng: np.random.Generator, params: PlateParams = PlateParams()) -> np.ndarray:
    """A random hole ``(cx, cy, r)`` that stays at least ``clearance`` away from the edges of the L."""
    while True:
        r = rng.uniform(params.r_min, params.r_max)
        cx, cy = rng.uniform(0, 1, size=2)
        # Check the disc grown by the clearance against the L, using points on its rim and centre.
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        rr = r + params.clearance
        xs, ys = np.r_[cx, cx + rr * np.cos(t)], np.r_[cy, cy + rr * np.sin(t)]
        # The re-entrant corner (0.5, 0.5) can poke into the disc between rim samples.
        far_from_corner = np.hypot(cx - 0.5, cy - 0.5) > rr
        if in_L(xs, ys).all() and far_from_corner:
            return np.array([cx, cy, r])


def free_mask(hole: np.ndarray, params: PlateParams = PlateParams()) -> np.ndarray:
    """``(n, n)`` boolean: True where the temperature is unknown (inside the L, outside the hole)."""
    X, Y = grid_xy(params)
    cx, cy, r = hole
    return in_L(X, Y) & (np.hypot(X - cx, Y - cy) > r)


def solve(hole: np.ndarray, params: PlateParams = PlateParams()) -> np.ndarray:
    """Steady temperature on the grid, ``(n, n)``; zero outside the plate and inside the hole."""
    n = params.n_grid
    h = 1.0 / (n - 1)
    free = free_mask(hole, params).ravel()
    # 5-point Laplacian on the full grid, then keep only the rows/columns of free nodes.
    # Fixed nodes are 0, so dropping their columns is exactly the Dirichlet condition.
    d1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    A = (sp.kron(sp.eye(n), d1) + sp.kron(d1, sp.eye(n))) / h**2
    idx = np.flatnonzero(free)
    A = A.tocsr()[idx][:, idx]
    u = np.zeros(n * n)
    u[idx] = spla.spsolve(A.tocsc(), np.full(idx.size, params.source))
    return u.reshape(n, n)


def make_features(points: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Per-point input features ``(N, 3)``: ``x``, ``y`` and the distance to the hole's edge.

    The distance ``|p - c| - r`` is how the geometry enters, the same way the
    Transolver paper feeds signed distances and normals for Shape-Net Car.
    """
    cx, cy, r = hole
    d = np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r
    return np.c_[points, d]


def sample_cloud(rng: np.random.Generator, u_grid: np.ndarray, hole: np.ndarray, n_points: int,
                 params: PlateParams = PlateParams()) -> tuple[np.ndarray, np.ndarray]:
    """``n_points`` uniform random points in the plate: positions ``(N, 2)`` and temperatures ``(N,)``."""
    cx, cy, r = hole
    pts = np.empty((0, 2))
    while len(pts) < n_points:   # rejection sampling: keep points inside the L and outside the hole
        cand = rng.uniform(0, 1, size=(2 * n_points, 2))
        ok = in_L(cand[:, 0], cand[:, 1]) & (np.hypot(cand[:, 0] - cx, cand[:, 1] - cy) > r)
        pts = np.r_[pts, cand[ok]]
    pts = pts[:n_points]
    s = np.linspace(0.0, 1.0, params.n_grid)
    u = RegularGridInterpolator((s, s), u_grid)(pts[:, ::-1])   # u_grid is indexed [y, x]
    return pts, u


def make_dataset(seed: int, n_samples: int, n_points: int, params: PlateParams = PlateParams()) -> dict:
    """Solve ``n_samples`` plates and draw one point cloud of ``n_points`` from each.

    Returns:
        ``{"hole": (S, 3), "u_grid": (S, n, n), "points": (S, N, 2), "features": (S, N, 3), "u": (S, N)}``
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("hole", "u_grid", "points", "features", "u")}
    for _ in range(n_samples):
        hole = sample_hole(rng, params)
        u_grid = solve(hole, params)
        pts, u = sample_cloud(rng, u_grid, hole, n_points, params)
        for k, v in zip(out, (hole, u_grid, pts, make_features(pts, hole), u)):
            out[k].append(v)
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def resample(data: dict, n_points: int, seed: int = 0, params: PlateParams = PlateParams()) -> dict:
    """New point clouds of a different size on the same plates (for evaluating at any ``N``)."""
    rng = np.random.default_rng(seed)
    pts, feats, us = [], [], []
    for hole, u_grid in zip(data["hole"], data["u_grid"]):
        p, u = sample_cloud(rng, u_grid, hole, n_points, params)
        pts.append(p); feats.append(make_features(p, hole)); us.append(u)
    return {"points": np.stack(pts).astype(np.float32), "features": np.stack(feats).astype(np.float32),
            "u": np.stack(us).astype(np.float32)}
