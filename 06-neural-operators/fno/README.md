# Fourier Neural Operator: Burgers' equation

A minimal, fully inspectable version of the **Fourier Neural Operator (FNO)** from Li et al., *Fourier Neural
Operator for Parametric Partial Differential Equations* (ICLR 2021, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)),
in pure JAX like the rest of this repository.

The task is the paper's 1D benchmark: learn the operator `u0(x) ↦ u(x, t=1)` of viscous Burgers' equation. It is
small on purpose: 32 grid points, 8 channels, 8 Fourier modes, so every tensor in physical space and in Fourier
space fits on screen.

## Start here

[`notebooks/fno_explained.ipynb`](notebooks/fno_explained.ipynb) walks through:

1. the ground-truth solver and the operator being learned,
2. what goes in (`[a(x), x]`, 32×2) and what comes out (32×1),
3. the architecture with every shape (lift P → 4 Fourier layers → project Q),
4. one forward pass, tensor by tensor, with a worked numeric example for each step: lifting, FFT, mode
   truncation, spectral weights `R`, inverse FFT, pointwise `W`, GELU, projection,
5. the relative L2 loss and where its gradient comes from,
6. training with a weight snapshot every 100 steps (about 10 seconds on a laptop CPU),
7. an **interactive explorer** with sliders for *training step*, *test sample* and *layer*,
8. predictions, **zero-shot super-resolution** (trained on 32 points, evaluated on 64 to 256 with the same
   weights), and the learned spectral weights,
9. how the FNO relates to the **adjoint method**: the FFT's adjoint is the inverse FFT, and the gradient of an
   objective through the trained FNO is compared with the gradient through the differentiable solver.

### The 2D version

[`notebooks/fno2d_explained.ipynb`](notebooks/fno2d_explained.ipynb) is the same walk-through on a **64 × 64
velocity field** (2D viscous Burgers): `64×64×4 [u, v, x, y] → P → 64×64×8 → 4 Fourier layers → Q → 64×64×2`.
It opens with Fourier and convolution from scratch on 8 numbers (what "converting to waves" means, why an "upstream"
rule is a convolution, and why that becomes one multiplication per wave), then draws every stage as one image per
channel: the input as both a `64×64×4` tensor and a `4096×4` matrix, the lift `P`, the 2D FFT of each channel with
the kept low modes boxed, the spectral weights `R` for one mode, the inverse FFT, `W`, GELU and `Q`, each with a worked
number. Test error is about 8% with 8 channels (about 5% with 16); training takes a few minutes on a CPU.

The sliders need a running Jupyter kernel. The saved outputs (including a static version of the explorer) render on
GitHub.

## Results (saved run)

| | |
|---|---|
| test relative L2, N = 32 (training resolution) | 0.48% |
| same weights, N = 64 / 128 / 256 | 0.35% / 0.35% / 0.35% |
| ∂u(0.5,1)/∂u0 through the FNO vs the solver | same shape up to a constant offset (cosine 0.98 to 1.00 after removing the mean), ~200× faster |

The constant offset in the gradient comes from the training data: every input has zero mean, so the FNO never
learned how the output responds to a constant shift. Section 9 explains it.

## Layout

```
fno/
├── src/
│   ├── physics.py     Burgers' equation: random initial conditions, pseudo-spectral solver, datasets
│   ├── physics2d.py   2D Burgers (velocity field u, v) on a 64×64 periodic grid
│   ├── model.py       FNO-1D (P, Fourier layers, Q); apply_with_trace returns every intermediate tensor
│   ├── model2d.py     FNO-2D, same structure, two corner blocks of kept modes
│   ├── training.py    relative L2 loss, Adam loop with snapshots, evaluation (1D and 2D)
│   └── visualize.py   annotated heatmaps, the one-layer flow figures (1D and 2D)
├── notebooks/
│   ├── fno_explained.ipynb     1D, 32 points: every number
│   └── fno2d_explained.ipynb   2D, 64×64: every channel image
└── requirements.txt
```

## Run

```bash
cd 06-neural-operators/fno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/fno_explained.ipynb
```

## Differences from the paper (all for readability)

| | FNO paper (Burgers) | here |
|---|---|---|
| viscosity ν | 0.1 | 0.02 (at 0.1 the solution at t = 1 is ~93% plain diffusion) |
| reference / training resolution | 8192 / 256 to 8192 | 256 / 32 |
| channel width d | 64 | 8 |
| Fourier modes k_max | 16 | 8 |
| Fourier layers | 4 | 4 |
| projection Q | 64 → 128 → 1 | 8 → 32 → 1 |
| training pairs | 1000 | 1000 |
| optimiser | Adam, step decay, 500 epochs | Adam, cosine decay, 3000 mini-batch steps |
