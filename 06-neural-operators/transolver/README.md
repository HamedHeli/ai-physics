# Transolver: heat in an L-shaped plate with a hole

A minimal, fully inspectable version of **Transolver** from Wu et al., *Transolver: A Fast Transformer Solver for
PDEs on General Geometries* (ICML 2024, [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)), in pure JAX like the
rest of this repository.

The task is one the [FNO](../fno/) cannot take directly. A uniformly heated **L-shaped plate** is cooled along its
edge and through a **round cooling hole that moves and changes size from sample to sample**. The model gets each
plate as a **cloud of scattered points** with features `(x, y, distance to the hole)` and predicts the steady
temperature at every point. This mirrors the paper's Elasticity benchmark (a plate with a varying hole, given as a
point cloud). It is small on purpose: 32 points in the walkthrough, 16 channels, 2 heads, 8 slices, so every
tensor of Physics-Attention fits on screen.

## Start here

[`notebooks/transolver_explained.ipynb`](notebooks/transolver_explained.ipynb) walks through:

1. the ground truth (`-Δu = f` on the plate, finite differences) and why an FFT-based FNO does not fit,
2. what goes in (`N×3`: rows are points, columns are `x, y, d`) and what comes out (`N×1`, one temperature per
   point), and why the row order does not matter,
3. the architecture with every shape (embed → 3 Transolver blocks → decode), as a table and as a diagram,
4. one forward pass, tensor by tensor, with a worked numeric example checked against the code for each step:
   embedding, LayerNorm, **slice weights** (softmax over `M` slices), **slice tokens** (weighted averages over
   points), **attention among the slices**, **deslice** back to the points, MLP, decoder, plus the `N²` vs `N·M`
   cost,
5. the relative L2 loss and where its gradient comes from,
6. training with a weight snapshot every 100 steps (about 1.5 minutes on a laptop CPU),
7. an **interactive explorer** with sliders for training step, test plate, block and head,
8. what it learned: predictions on unseen plates, **are the slices physical states?** (here, honestly, mostly not:
   they are learned soft regions of the L, and the hole reaches each point through the slice tokens instead), the
   same weights on 32 to 2000 points, and what happens without the geometry feature,
9. where the physics is: the PDE residual of the prediction on the full grid,
10. Transolver++ (Ada-Temp, Rep-Slice, parallelism) in one page and an FNO vs Transolver table.

The sliders need a running Jupyter kernel (press *Run Interact*). The saved outputs, including a static version of
the explorer, render on GitHub.

## Results (saved run)

| | |
|---|---|
| test relative L2, 256 points per plate (training size) | 3.0% |
| same weights, 32 / 64 / 2000 points | 5.0% / 3.9% / 2.7% |
| "average plate" baseline (ignores the hole) | 34.9% |
| without the distance feature `d`, 64 / 256 / 2000 points | 30% / 19% / 13% |
| share of the slice assignment that changes when the hole moves | 1% to 7% per head (the slices are mostly fixed soft regions) |
| PDE residual `|-Δu - f|` of the prediction, mean on the 65×65 grid | about 40% of `f` |

The model learns the field to a few percent, but its second derivatives are noisy: nothing in the loss asks it to
satisfy the PDE.

## Layout

```
transolver/
├── src/
│   ├── physics.py     L-shaped plate with a hole: random holes, sparse finite-difference solver, point clouds
│   ├── model.py       Transolver (embed, Physics-Attention blocks, decoder); apply_with_trace returns every tensor
│   ├── training.py    relative L2 loss, AdamW one-cycle loop with snapshots, evaluation on any N
│   └── visualize.py   annotated heatmaps, fields and slice maps, the Physics-Attention flow and architecture figures
├── notebooks/
│   └── transolver_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 06-neural-operators/transolver
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/transolver_explained.ipynb
```

## Differences from the paper (all for readability)

| | Transolver paper (Elasticity benchmark) | here |
|---|---|---|
| problem | hyperelastic plate with a varying hole → stress, 972 points | heated L-plate with a varying hole → temperature, 256 points |
| input features | point coordinates | coordinates and distance to the hole |
| channels `C` | 128 to 256 | 16 |
| heads `H` | 8 | 2 |
| slices `M` | 32 to 64 | 8 |
| blocks `L` | 8 | 3 |
| temperature `τ` | learned per head, init 0.5 | same (stored as `log τ`) |
| placeholder vector, dropout | yes | no |
| training pairs | 1000 | 1000 |
| optimiser | AdamW, one-cycle, 500 epochs | AdamW, one-cycle, 3000 mini-batch steps |
