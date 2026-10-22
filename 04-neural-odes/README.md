# 04 Neural ODEs

**Week 5 (Oct 8), Unit I: the fundamentals** · [back to the graph](../README.md)

**What is learned:** the time derivative (or the latent time step). The network is `f_θ` in `dz/dt = f_θ(z, t)`, and
an ODE solver produces the trajectory.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Chen+ 2018] Neural Ordinary Differential Equations.** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud | NeurIPS 2018 | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| R2 | **[Wiewel+ 2019] Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow.** Steffen Wiewel, Moritz Becher, Nils Thuerey | Eurographics 2019 (CGF) | [arXiv:1802.10123](https://arxiv.org/abs/1802.10123) |

## Key ideas

- **Residual nets as Euler steps.** `h_{t+1} = h_t + f(h_t)` is one Euler step; take the limit and use any adaptive
  ODE solver for the forward pass.
- **Constant-memory gradients with the adjoint.** Solve an augmented ODE backwards in time for `a(t) = ∂L/∂z(t)`
  and the parameter gradient, instead of storing the forward pass. The cost is drift when the backward
  re-integration does not retrace the forward path.
- **Latent-space physics:** a convolutional autoencoder compresses each pressure field to a small code, and an LSTM
  predicts the next codes from the previous ones. Learning temporal evolution in a reduced space is much cheaper
  than in the full grid.

## Connections

- ← [01 Differentiable simulation](../01-differentiable-simulation/): the continuous adjoint, implemented by hand in
  [`adjoint/`](../01-differentiable-simulation/adjoint/), including the drift problem.
- ← [03 Reduced-order modeling](../03-reduced-order-modeling/): Wiewel+ is an encoder-decoder reduced model plus
  learned latent dynamics.
- → [05 Graph neural networks](../05-graph-neural-networks/): GNS is also a learned time-stepper, but on particles
  and graphs instead of a single latent vector.
