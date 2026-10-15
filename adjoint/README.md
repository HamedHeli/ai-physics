# The adjoint method, by hand: aiming a thrown ball

A worked, inspectable example of the **adjoint method** for computing gradients of a simulation with respect to
design parameters. It is the idea behind DAFoam (the airfoil baseline in Allen et al. 2022), behind reverse-mode
autodiff through a learned simulator (the GNN design paper), and behind the constant-memory Neural ODE gradient
(Week 5).

## The problem

A ball is thrown from the origin through air with quadratic drag and an optional piecewise-constant thrust:

```
dx/dt = v
dv/dt = g + u(t) - c |v| v
```

The design parameters are the launch velocity `v0` and the thrust table `U` (K segments × 2). The objective is to
land on a target at the final time, `J = |x_N - x_target|^2`.

## Notes: the adjoint in five lines

For a time-stepped simulator `s_{k+1} = F(s_k, θ)` and an objective on the final state `J(s_N)`:

1. Start from `λ_N = ∂J/∂s_N`.
2. Walk backwards: `λ_k = (∂F/∂s_k)ᵀ λ_{k+1}`.
3. Collect parameter gradients on the way: `dJ/dθ += (∂F/∂θ)ᵀ λ_{k+1}`.
4. One forward + one backward pass gives the gradient for **all** parameters, versus `2P` simulations for finite
   differences.
5. If `∂F/∂s` depends on the state (here through drag), the backward pass needs the forward trajectory: store it,
   checkpoint it, or re-integrate it backwards and accept drift.

Reverse-mode autodiff (`jax.grad`) applied to the simulator code produces exactly this discrete adjoint
automatically. The notebook checks that the hand-written version matches it to machine precision.

## Start here

[`notebooks/adjoint_explained.ipynb`](notebooks/adjoint_explained.ipynb):

1. no drag: the adjoint is a constant 4×4 Jacobian, transposed and applied 200 times;
2. with drag: a state-dependent Jacobian, checked against finite differences and `jax.grad`;
3. **slider:** step through the backward sweep and watch the sensitivity flow from the target to the launch;
4. cost vs number of parameters (adjoint ~300× faster than finite differences at 402 parameters);
5. memory: store-all vs checkpointing (identical gradients, ~2√N states);
6. **slider:** the continuous adjoint with backward re-integration, and how drag makes it drift;
7. **slider:** gradient descent with adjoint gradients on 22 parameters to hit the target.

The sliders need a running Jupyter kernel. The saved outputs (with static versions of each slider) render on
GitHub.

## Layout

```
adjoint/
├── src/
│   ├── ball.py       forward simulator (explicit Euler), objective
│   ├── adjoint.py    discrete, checkpointed and continuous adjoints; finite differences; jax.grad check
│   └── optimize.py   gradient descent driven by the adjoint
├── notebooks/
│   └── adjoint_explained.ipynb
└── requirements.txt
```

The simulator and adjoints are plain NumPy so every line of the adjoint is visible. JAX is used only to confirm that
autodiff gives the same answer.

## Run

```bash
cd adjoint
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/adjoint_explained.ipynb
```
