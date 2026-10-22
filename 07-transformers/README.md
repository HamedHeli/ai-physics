# 07 Transformers

**Week 8 (Oct 29), Unit II: learning all the way** · [back to the graph](../README.md)

**What is learned:** a general-purpose surrogate for many PDEs at once, trained like a foundation model. The
network still maps input fields to output fields, but one backbone covers many equation families.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Holzschuh+ 2025] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations.** Benjamin Holzschuh, Qiang Liu, Georg Kohl, Nils Thuerey | ICML 2025 | [arXiv:2505.24717](https://arxiv.org/abs/2505.24717) · [PMLR](https://proceedings.mlr.press/v267/holzschuh25a.html) |

Week 8 also has the first project check-in.

## Key ideas

- **Diffusion-transformer backbone for physics.** Fields are patchified into tokens on regular grids and processed
  by a multi-scale transformer with shifted-window attention, so cost stays manageable at high resolution.
- **Mixed-channel vs separate-channel embeddings.** Embedding each physical channel separately lets one model handle
  PDEs with different numbers and kinds of variables.
- **Pre-train, then fine-tune.** Trained on a large mix of PDE datasets, it transfers to new downstream PDEs better
  than training from scratch.

## Connections

- ← [06 Neural operators](../06-neural-operators/): Transolver is attention on general meshes; PDE-Transformer is
  attention on grids at scale.
