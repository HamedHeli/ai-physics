# 06 Neural operators

**Week 7 (Oct 22), Unit II: learning all the way** · [back to the graph](../README.md)

**What is learned:** the solution operator. The network maps an input *function* (initial condition, coefficient
field, geometry) to an output *function*, `G_θ: a(x) ↦ u(x)`, for every instance at once. This is "learning the
function" in the operator-learning sense, as opposed to a PINN, which learns one `u`.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Li+ 2020] Fourier Neural Operator for Parametric Partial Differential Equations (FNO).** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar | ICLR 2021 | [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) |
| R2 | **[Wu+ 2024] Transolver: A Fast Transformer Solver for PDEs on General Geometries.** Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, Mingsheng Long | ICML 2024 | [arXiv:2402.02366](https://arxiv.org/abs/2402.02366) |

The schedule page lists only FNO for Week 7; Transolver was announced as the second reading on the course Slack.

## Key ideas

- **Operator layers.** Each layer is `v ↦ σ(W v + K v)`, where `K` is an integral operator. FNO computes `K` as a
  convolution in Fourier space: FFT, keep the lowest `k_max` modes, multiply by learned complex weights, inverse FFT.
- **Discretization invariance.** The learned weights live on Fourier modes, not grid points, so a model trained at
  64×64 can be evaluated at 256×256 (zero-shot super-resolution). The FFT ties FNO to regular grids.
- **Transolver: attention over physical states, not mesh points.** Each mesh point is softly assigned to `M` learned
  *slices*; slice tokens are aggregated, attention runs among the `M` slice tokens, and results are broadcast back.
  Cost is linear in the number of points, and it works on unstructured meshes and point clouds.
- **The lineage studied for the Archaeologist role:** Attention (Vaswani+ 2017) → GNOT (Hao+ 2023) → Transolver →
  Transolver++ (2025). Study notes: `week7/transolver-study-notes.md`; FNO outline:
  `week7/fno-hierarchical-outline.md`; FFT demo: `week7/fft_demo.py` (all in the project files).

## Connections

- ← [02 Physics-informed NNs](../02-physics-informed-neural-networks/): from one solution to the whole operator.
- ← [05 Graph neural networks](../05-graph-neural-networks/): message passing on a mesh is a discretized kernel
  integral; Transolver replaces it with global attention over learned slices.
- → [07 Transformers](../07-transformers/): PDE-Transformer scales attention-based surrogates further.
- ↔ [09 One PDE, many ways](../09-one-pde-many-ways/): §7–8 train an FNO and a Transolver on the Burgers operator
  `u(·,0) ↦ u(·,1)` and compare them, next to every other method of the course.
