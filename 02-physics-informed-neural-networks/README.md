# 02 Physics-informed neural networks

**Week 3 (Sep 24), Unit I: the fundamentals** · [back to the graph](../README.md)

**What is learned:** one solution. The network *is* the field `u(x, t)` for a single PDE instance; change the
boundary condition or coefficients and you train again.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Raissi+ 2019] Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.** Maziar Raissi, Paris Perdikaris, George Em Karniadakis | J. Comput. Phys. 378 (2019) | [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) |
| R2 | **[Chen+ 2023] Implicit Neural Spatial Representations for Time-dependent PDEs (INSR).** Honglin Chen, Rundi Wu, Eitan Grinspun, Changxi Zheng, Peter Yichen Chen | ICML 2023 | [arXiv:2210.00124](https://arxiv.org/abs/2210.00124) · [project](https://www.cs.columbia.edu/cg/INSR-PDE) |

## Key ideas

- **PINNs:** an MLP `u_θ(x, t)` trained on data points plus the PDE residual at collocation points, with derivatives
  from autodiff. The same loss solves forward problems and inverse problems (unknown coefficients become trainable).
- **Soft constraints are hard to balance.** Residual, boundary and data terms compete, and stiff or multi-scale
  problems often fail to train.
- **INSR** keeps a classical time integrator and uses the neural field only as the *spatial* discretization: at each
  step it optimizes the network weights so the field satisfies one implicit time step. Time is handled by the
  solver, space by the network, so memory does not grow with resolution and adaptivity comes for free.

## Connections

- → [03 Reduced-order modeling](../03-reduced-order-modeling/): CROM (same group as INSR) uses neural fields as the
  decoder of a reduced space.
- → [06 Neural operators](../06-neural-operators/): from learning one solution to learning the map from inputs to
  solutions.
