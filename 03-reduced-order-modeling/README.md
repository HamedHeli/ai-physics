# 03 Reduced-order modeling

**Week 4 (Oct 1), Unit I: the fundamentals** · [back to the graph](../README.md)

**What is learned:** a low-dimensional manifold of solutions. A decoder maps a few latent coordinates `z` to a full
field or configuration, and the simulation runs in `z`.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Chen+ 2023] CROM: Continuous Reduced-Order Modeling of PDEs Using Implicit Neural Representations.** Peter Yichen Chen, Jinxu Xiang, Dong Heon Cho, Yue Chang, G A Pershing, Henrique Teles Maia, Maurizio M. Chiaramonte, Kevin Carlberg, Eitan Grinspun | ICLR 2023 | [arXiv:2206.02607](https://arxiv.org/abs/2206.02607) |
| R2 | **[Sharp+ 2023] Data-Free Learning of Reduced-Order Kinematics.** Nicholas Sharp, Cristian Romero, Alec Jacobson, Etienne Vouga, Paul G. Kry, David I.W. Levin, Justin Solomon | SIGGRAPH 2023 | [arXiv:2305.03846](https://arxiv.org/abs/2305.03846) · [project](https://research.nvidia.com/labs/toronto-ai/neural-physics-subspaces) |

## Key ideas

- **CROM:** the decoder is a neural field `g(x, z)`, so the reduced model is discretization-agnostic: train on one
  mesh, evaluate at any point. Dynamics are projected into latent space and only a few sample points are evaluated
  per step (network-based hyper-reduction).
- **Data-free subspaces (Sharp+ 2023):** no simulation data at all. Given only an energy `E(q)` and a seed
  configuration, train `q = f_θ(z)` so that the image is low-energy *and* diverse (a distance-preserving term keeps
  it from collapsing to one point). Works for elastic bodies, cloth, rigid bodies in contact and linkages.
- Both replace the linear basis of classical model reduction (PCA/POD, modal analysis) with a nonlinear decoder.

## Code in this folder

- [`data-free-reduced-kinematics/`](data-free-reduced-kinematics/): a JAX reimplementation of Sharp+ 2023 with the
  Klann linkage and the cloth-ball experiments (`experiments/`), the energy and model in `src/`, and the paper text
  in `reference/`.

## Connections

- ← [02 Physics-informed NNs](../02-physics-informed-neural-networks/): CROM's decoder is a neural field like INSR.
- → [04 Neural ODEs](../04-neural-odes/): once you have a latent space, learn the dynamics in it (Wiewel+ 2019).
