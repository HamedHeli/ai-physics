# Experimental log (generated 2026-11-17 from viscoelastic-surrogate/run_all.py; see results.json)

DATA PROVENANCE: one REAL Abaqus curve (phi = 4.95 mm, delta = 1 mm, ramp tau0 = 0.12 s,
hold 10 s, Ecoflex 00-10 Table-1 material: mu = 3.98 kPa, alpha = 5.39, g = 0.33/0.48/0.13,
tau = 0.1/1/10 s). ALL training and test curves are SYNTHETIC. They use the exact QLV convolution
plus a stand-in power-law elastic law (2 parameters calibrated on the real curve; its
alpha-dependence is assumed). Replace these with real Abaqus runs before submission.

## Physics check (real data)
- The QLV model with Table-1 parameters reproduces the Abaqus indenter curve with 0.27% relative L2 error.
- Classical least squares with the same physics recovers g = 0.344 / 0.485 / 0.126 from the curve (truth 0.33 / 0.48 / 0.13).

## Forward surrogate (synthetic, 256 test curves, median relative L2)
Input ranges: phi 4-9 mm, delta/phi 0.05-0.5, alpha 2-8, g_k 0-1, tau0 0.05-1 s.
OOD = tau0 1-3 s and delta/phi 0.5-0.7.
| training curves | DeepONet | QLV-DeepONet | DeepONet OOD | QLV-DeepONet OOD |
| 16   | 7.13% | 0.71% | 15.94% | 2.83% |
| 64   | 2.77% | 0.45% | 5.01%  | 1.46% |
| 256  | 1.70% | 0.43% | 3.39%  | 1.42% |
| 1024 | 1.59% | 0.36% | 3.39%  | 1.31% |
- QLV-DeepONet trained on 16 curves is more accurate than DeepONet trained on 1024.
- Surrogate forward evaluation: 0.07 ms per curve on CPU, against about 10 h per Abaqus run.

## Real Abaqus curve (models trained on 1024 synthetic curves)
- Forward error with Table-1 parameters: DeepONet 1.83%, QLV-DeepONet 0.27%.
- Inverse, alpha known: QLV-DeepONet g = 0.322 / 0.493 / 0.109; DeepONet g = 0.368 / 0.491 / 0.043.
- Inverse, mu, alpha, g all free: QLV-DeepONet g = 0.341 / 0.487 / 0.110 in 1.4 s; DeepONet 0.386 / 0.481 / 0.043.

## Joint elastic + viscous inversion (synthetic, 60 materials, 3 depths 0.1/0.2/0.4, phi 6.33 mm, 1% noise)
Median errors (mu relative, others absolute):
| param | DeepONet | QLV-DeepONet |
| mu    | 11.6%  | 3.9%  |
| alpha | 0.065  | 0.098 |
| g1    | 0.078  | 0.024 |
| g2    | 0.052  | 0.022 |
| g3    | 0.222  | 0.058 |
- g3 is systematically under-recovered when the true g3 > 0.5, for both models. tau3 = 10 s equals
  the hold length, so g3 and mu are confounded. Recommendation: hold >= 3 tau_max.

## Figures (viscoelastic-surrogate/figures/)
- fig1: Abaqus curve vs QLV model, and its decomposition into Prony basis curves.
- fig2: error vs number of training simulations, in-distribution and OOD.
- fig3: example synthetic test curves.
- fig4: forward prediction on the real Abaqus curve.
- fig5: recovered vs true g_k (synthetic), and recovered g on the real curve with Laplace error bars.
