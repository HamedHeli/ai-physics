# 01 Differentiable simulation

**Week 2 (Sep 17), Unit I: the fundamentals** · [back to the graph](../README.md)

**What is learned:** nothing yet. The point of this topic is that a physics solver, written the right way, is a
differentiable function of its inputs, so `d(loss)/d(parameters)` is available for control, design and system
identification.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Hu+ 2019] DiffTaichi: Differentiable Programming for Physical Simulation.** Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, Frédo Durand | ICLR 2020 | [arXiv:1910.00935](https://arxiv.org/abs/1910.00935) |
| R2 | **[Macklin 2024] Warp: Differentiable Spatial Computing for Python.** Miles Macklin (NVIDIA) | SIGGRAPH 2024 course | [code](https://github.com/NVIDIA/warp) |

## Key ideas

- **Source-to-source autodiff for simulation kernels.** DiffTaichi differentiates mega-kernels (particle-to-grid
  loops, collision handling) instead of tracing every scalar operation, which keeps the backward pass about as fast
  as the forward pass.
- **The tape and its memory cost.** Reverse mode needs the forward trajectory. DiffTaichi records kernel launches on
  a light tape and recomputes or checkpoints the states, the same trade-off as store-all vs checkpointing in the
  adjoint method.
- **Gradients can be wrong even when they are exact.** Contacts and time-of-impact discontinuities make naive
  gradients useless; DiffTaichi's continuous collision fix is the standard example.
- **Warp** brings the same idea to Python: kernels written in Python, JIT-compiled to CUDA, with `wp.Tape` for
  reverse-mode gradients and interop with PyTorch and JAX.

## Code in this folder

- [`adjoint/`](adjoint/): the adjoint method written by hand for a thrown ball with drag and thrust, checked against
  finite differences and `jax.grad`, with checkpointing and the continuous adjoint. This is the mechanism under every
  "gradient through the simulator" in the rest of the course.

## Connections

- → [04 Neural ODEs](../04-neural-odes/): the Neural ODE gradient is the continuous adjoint from `adjoint/`.
- → [05 Graph neural networks](../05-graph-neural-networks/): Allen+ 2022 replaces the differentiable solver with a
  learned one and runs the same gradient-based design loop.
- → [08 Reconstructing physics](../08-reconstructing-physics/): PAC-NeRF identifies material parameters by
  differentiating an MPM simulator of the DiffTaichi family.
- ↔ [09 One PDE, many ways](../09-one-pde-many-ways/): §2 recovers the viscosity and the initial state of Burgers by
  differentiating through the solver, next to every other method of the course.
