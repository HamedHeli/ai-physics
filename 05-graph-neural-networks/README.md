# 05 Graph neural networks

**Week 6 (Oct 15), Unit I: the fundamentals** · [back to the graph](../README.md)

**What is learned:** the time step. A graph network predicts per-particle (or per-node) accelerations, and an
integrator rolls the state forward: `x_{t+1} = x_t + Δt · v_{t+1}` with `v_{t+1} = v_t + Δt · GNN(x_t)`.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Sanchez-Gonzalez+ 2020] Learning to Simulate Complex Physics with Graph Networks (GNS).** Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, Peter W. Battaglia | ICML 2020 | [arXiv:2002.09405](https://arxiv.org/abs/2002.09405) |
| R1 | **[Allen 2022] Physical Design using Differentiable Learned Simulators.** Kelsey R. Allen, Tatiana Lopez-Guevara, Kimberly Stachenfeld, Alvaro Sanchez-Gonzalez, Peter Battaglia, Jessica Hamrick, Tobias Pfaff | NeurIPS 2022 | [arXiv:2202.00728](https://arxiv.org/abs/2202.00728) |

## Key ideas

- **Encode, process, decode.** Particles become nodes, neighbours within a radius become edges, and `M` rounds of
  message passing propagate interactions before a decoder reads out accelerations.
- **Noise injection** during training makes long rollouts stable by teaching the model to correct its own errors.
- **Locality and translation invariance** (relative positions on edges) let one model generalize to more particles
  and larger domains than it was trained on.
- **Physical design (Allen+ 2022):** because the learned simulator is a neural network, `jax.grad` through a
  multi-step rollout gives design gradients (airfoil shapes, fluid-directing tools) without writing an adjoint
  solver, and it is competitive with the specialized DAFoam adjoint baseline.

A longer walk-through of both papers is in the project notes (`gnn-papers/week6-gnn-papers-explained.md`).

## Code in this folder

- [`gnn/`](gnn/): a minimal GNS in pure JAX on three colliding balls, with a notebook that shows every matrix of one
  forward pass, noise-injected training, an interactive explorer, and generalization from 3 to 5 balls.

## Connections

- ← [01 Differentiable simulation](../01-differentiable-simulation/): Allen+ 2022's design loop is the adjoint method
  through a learned simulator.
- ← [04 Neural ODEs](../04-neural-odes/): another learned time-stepper.
- → [06 Neural operators](../06-neural-operators/): GNO-style operators and Transolver generalize message passing on
  meshes to learned, discretization-independent kernels and attention.
- ↔ [09 One PDE, many ways](../09-one-pde-many-ways/): §6 runs a GNS-style simulator on the grid graph of Burgers,
  next to every other method of the course.
