# Viscoelastic indentation surrogate (CPSC 532Z course project)

AI surrogate + inverse solver for the ramp-and-hold flat-punch indentation in
Helisaz, Bacca & Chiao, *Exp. Mech.* 62:893-908 (2022). One Abaqus run takes about 10 h.
The surrogate takes well under 1 ms per curve, and recovering the parameters from a
measured curve takes about 1 s.

## The idea in one line: learn only the part physics can't give you

The QLV model splits the response into two parts (paper eqs. 1, 3 and 8; Figs. 6 and 7):

```
sigma(t) = integral_0^t G(t-s) d sigma_E(s),      G(t) = 1 + sum_k g_k exp(-t/tau_k)
           \_______ exact, cheap _______/  \___ elastic indentation law: needs FE ___/
```

* **Viscous part:** an exact, differentiable Prony convolution (`vesurrogate/qlv.py`).
* **Elastic part:** sigma_E/mu = f(delta/phi, alpha, phi). This is the only thing the
  10 h Abaqus run really computes. A small MLP learns it in log-log space.

Put together, this is a DeepONet with a physics trunk:
`sigma = B_0(t) + sum_k g_k B_k(t)`. The basis curves B_k come from the convolution and
the coefficients are (1, g_k). That is the paper's eq. (13), turned into a network.
The black-box DeepONet baseline learns everything from data.

## Files

| file | what it does |
|---|---|
| `vesurrogate/qlv.py` | exact differentiable QLV/Prony layer, Table 1 values, the stand-in elastic law used for synthetic data |
| `vesurrogate/data.py` | loads a folder of `Abaqus_Results_<size>_<depth>mm.txt` files (`495` -> 4.95 mm, `1mm` -> 1 mm). Ramp time tau0 is found from the duplicated step-boundary time stamp |
| `vesurrogate/synthetic.py` | synthetic ramp-and-hold curves (SYNTHETIC: exact QLV physics + stand-in elastic law) |
| `vesurrogate/models.py` | `DeepONet` (black box) and `QLV-DeepONet` (physics-structured) in pure JAX |
| `vesurrogate/train.py` | training, error metrics, multi-start gradient inversion through the surrogate, Laplace (Gauss-Newton) uncertainty |
| `run_all.py` | the full experiment. Writes `figures/`, `results.json` and `calibration.json` |

## Run

```bash
pip install -r requirements.txt
python run_all.py --quick                 # ~1-2 min smoke test
python run_all.py --data data/abaqus      # full run (~15-25 min on a laptop CPU)
```

Drop more Abaqus files into `data/abaqus/` with the same naming. Every file is then
validated (Fig. 4), with no code changes.

## What is real and what is synthetic

* **Real:** `data/abaqus/Abaqus_Results_495_1mm.txt` (phi = 4.95 mm, delta = 1 mm,
  ramp tau0 = 0.12 s, then a 10 s hold). This file is used for validation only, except
  for one step: two numbers (K, n0) of the stand-in elastic law are calibrated on it.
* **Synthetic:** every training and test curve. These use the exact QLV physics plus
  `elastic_standin`, a power law in delta/phi with a Hayes finite-thickness factor.
  Its alpha-dependence (`STANDIN_C = 0.08`) is an assumption. **Synthetic numbers
  measure the ML, not the mechanics.** Once real Abaqus runs over (phi, delta, alpha)
  exist, train on them instead: replace `make_dataset` with a loader.

## Swapping in real Abaqus training data

Each Abaqus run gives one curve for one theta = (phi, delta/phi, alpha, g1, g2, g3, tau0).
Build `dict(theta=(N,7), t=(N,n), y=(N,n))` with y = output / mu and t starting at 0,
resampled to a common grid (see `run_all.real_eval_times`). Then call
`train.train("QLV-DeepONet", data)`.

The material parameters can't be read from the file name. Keep them in a CSV next to
the files, or in a name suffix: anything after `mm` is kept in `tag`.
