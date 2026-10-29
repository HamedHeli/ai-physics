# 09 One PDE, many ways

**Cross-topic** · [back to the graph](../README.md)

Every topic of the course is a different answer to the same question: *how do I get the solution of a PDE?* This
folder takes **one** physically meaningful equation and solves it with each approach from topics 01–07, on the same
data, so the differences you see are about the methods and not about the problems.

## The PDE: 1D viscous Burgers

```
u_t + u u_x = nu u_xx,      x in [0, 2 pi) periodic,   t in [0, 1],   nu = 0.1
```

`u(x, t)` is a velocity. Burgers is the momentum equation of Navier–Stokes with the pressure taken out: **advection**
`u u_x` makes fast fluid catch up with slow fluid, so smooth waves steepen into fronts (a 1D shock, or a traffic jam
forming), and **viscosity** `nu u_xx` rounds those fronts off. It is the simplest PDE that is both nonlinear and
physical. (The heat equation is simpler but linear, and would make POD and the FNO trivially exact.)

Data: 512 training and 64 test trajectories from random smooth initial conditions, solved on 256 points and kept on
a 64-point grid at 21 times.

## Start here

[`notebooks/one_pde_many_ways.ipynb`](notebooks/one_pde_many_ways.ipynb), about 10 minutes on a laptop CPU.

| § | Method | Topic | What is learned | Module |
| --- | --- | --- | --- | --- |
| 1 | Pseudo-spectral IFRK4 solver, plus finite differences | baseline | nothing: the PDE is known | [`burgers.py`](src/burgers.py) |
| 2 | Differentiable simulation (adjoint by `jax.grad`) | [01](../01-differentiable-simulation/) | the unknown inputs: viscosity, initial state | [`inverse.py`](src/inverse.py) |
| 3 | PINN with hard periodic and initial constraints | [02](../02-physics-informed-neural-networks/) | one solution `u(x, t)` | [`pinn.py`](src/pinn.py) |
| 4 | POD-Galerkin reduced-order model | [03](../03-reduced-order-modeling/) | a 16-dim. linear subspace | [`rom.py`](src/rom.py) |
| 5 | Latent neural ODE on POD coefficients | [04](../04-neural-odes/) | `dz/dt = f(z)` in that subspace | [`neural_ode.py`](src/neural_ode.py) |
| 6 | GNS-style graph network on the grid | [05](../05-graph-neural-networks/) | the time step `Δt = 0.05` | [`gnn.py`](src/gnn.py) |
| 7 | Fourier neural operator | [06](../06-neural-operators/) | the operator `u(·,0) ↦ u(·,1)` | [`fno.py`](src/fno.py) |
| 8 | Transolver (Physics-Attention) | [06](../06-neural-operators/) / [07](../07-transformers/) | the same operator, with attention over slices | [`transolver.py`](src/transolver.py) |

## Results

All errors are relative L2 of `u(x, T = 1)` on 64 held-out initial conditions (PINN: 3 of them), times from one
run on a 4-core CPU. "Per new sample" is the cost of an answer for one new initial condition; for the PINN that is a
full training run.

| Method | Rel. L2 error at T = 1 | Training | Per new sample | Data needed |
| --- | --- | --- | --- | --- |
| Spectral solver, 64 points | 0.18% | none | 1.2 ms | none |
| Finite differences, 64 points | 0.90% | none | 1.6 ms | none |
| Differentiable sim (inverse) | recovers `nu` to 0.5%, `u0` to 18% | none | 28 s for both problems | 1 observation |
| PINN | **0.09%** | 31 s per instance | 31 s (it retrains) | none, PDE only |
| POD-Galerkin, r = 16 | 5.3% (projection floor 4.4%) | one SVD | 1.0 ms | 512 trajectories |
| Latent neural ODE, r = 16 | 7.7% | 72 s | 0.23 ms | 512 trajectories |
| GNS on the grid graph | 1.8% after 20 steps | 153 s | 7.8 ms | 512 trajectories |
| FNO | 0.38% (same on 128 and 256 points) | 40 s | **0.14 ms** | 512 `(u0, u(T))` pairs |
| Transolver | 5.8% (same on 128 and 256 points) | 150 s | 0.30 ms | 512 `(u0, u(T))` pairs |

The short version:

- **The classical solver is the bar.** On a small periodic 1D problem nothing learned beats it on accuracy. Learned
  surrogates pay off when the solver is expensive or the physics is only partly known.
- **Differentiable simulation answers a different question:** inverse problems, with exact gradients and no
  training data. Its limit is physical (detail lost to viscosity cannot be recovered).
- **PINNs are accurate but per-instance:** no data, but a full training run for every new initial condition.
- **Linear ROMs struggle with fronts.** Translation-invariant data gives Fourier-like POD modes with slowly decaying
  energy, which is the motivation for CROM's nonlinear decoder.
- **The latent neural ODE** has the same space as POD-Galerkin but must learn the physics from data.
- **GNS** is translation-equivariant by construction and accurate over a 20-step rollout, with error growing per step.
- **Neural operators amortize:** after a minute of training the FNO answers a new initial condition in one forward
  pass and runs on finer grids unchanged. Transolver learns the same map without assuming a grid, at a higher price
  on this regular one.

## Layout

```
09-one-pde-many-ways/
├── src/
│   ├── burgers.py      the PDE: IFRK4 pseudo-spectral solver, finite differences, datasets
│   ├── nn.py           MLPs as lists of (W, b), Adam training loop, timing
│   ├── inverse.py      §2 viscosity identification and initial-state recovery through the solver
│   ├── pinn.py         §3 PINN with hard constraints
│   ├── rom.py          §4 POD basis and quadratic Galerkin tensors
│   ├── neural_ode.py   §5 latent neural ODE, RK4 with backprop through the integrator
│   ├── gnn.py          §6 encode-process-decode on the ring graph
│   ├── fno.py          §7 Fourier neural operator
│   └── transolver.py   §8 Physics-Attention transformer
├── notebooks/
│   └── one_pde_many_ways.ipynb
└── requirements.txt
```

## Run

```bash
cd 09-one-pde-many-ways
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/one_pde_many_ways.ipynb
```

## Notes

- The solver uses the same scheme (IFRK4, 2/3 dealiasing) as the FNO notebook in `06-neural-operators/fno`, but on
  `[0, 2π)` with `nu = 0.1` and with full trajectories saved, because the trajectory methods (ROM, neural ODE, GNS)
  need them. The FNO here is a small self-contained copy so this folder runs on its own.
- All sizes are small on purpose (64 grid points, 512 trajectories, widths of 32–64) so everything trains on a CPU.
  The ranking between methods can change at scale, especially for the FNO and Transolver, whose papers use thousands
  of samples and much larger models.
