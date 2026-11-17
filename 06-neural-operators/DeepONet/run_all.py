"""End-to-end experiment: calibrate -> synthetic data -> train 2 surrogates -> validate
on the Abaqus curve(s) -> inverse problem. Writes figures/ and results.json.

    python run_all.py                     # full run (~10-20 min on a laptop CPU)
    python run_all.py --quick             # smoke test (~2 min)
    python run_all.py --data data/abaqus  # folder of Abaqus_Results_<size>_<depth>mm.txt

What is real and what is synthetic:
  REAL      the Abaqus curve(s) in --data (validation only, never used for training
            except the 2-number calibration of the stand-in elastic law, step 1)
  SYNTHETIC all training/test curves (exact QLV viscous physics + stand-in elastic law)
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from vesurrogate import data as D, qlv, synthetic as S, train as T

INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e1e0d9"
COL = {"DeepONet": "#eb6834", "QLV-DeepONet": "#2a78d6", "aux": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})


def real_eval_times(c, n_hold=150):
    """All ramp samples + a log-spaced subset of the hold (keeps every curve the same length)."""
    t, y, tau0 = c["t"], c["y"], c["tau0"]
    ramp = np.where(t <= tau0 + 1e-9)[0]
    th = t[t > tau0]
    targets = tau0 + np.logspace(np.log10(th[0] - tau0), np.log10(th[-1] - tau0), n_hold)
    hold = np.unique(np.searchsorted(t, targets).clip(ramp[-1] + 1, len(t) - 1))
    idx = np.concatenate([ramp, hold])
    return t[idx].astype(np.float32), y[idx].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/abaqus")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}
    steps = 800 if a.quick else 4000
    Ns = [16, 64] if a.quick else [16, 64, 256, 1024]

    # ------------------------------------------------------------ 0. real data
    curves = D.load_folder(a.data)
    print(f"loaded {len(curves)} Abaqus curve(s)")
    for c in curves:
        c["te"], c["ye"] = real_eval_times(c)
        print(f"  phi={c['phi']} mm  delta={c['delta']} mm  d={c['d']:.3f}  tau0={c['tau0']} s  "
              f"peak={c['y'].max():.4g}  end={c['y'][-1]:.4g}")
    ref = next((c for c in curves if abs(c["phi"] - 4.95) < 1e-6 and abs(c["delta"] - 1) < 1e-6), curves[0])
    g_tab, mu_tab, al_tab = np.array(qlv.TABLE1["g"]), qlv.TABLE1["mu"], qlv.TABLE1["alpha"]

    # ------------------------------------------------------------ 1. calibration
    # (a) does the paper's QLV model + Table 1 reproduce the Abaqus curve?
    # (b) calibrate the stand-in elastic law (K, c) on that curve -> synthetic world
    te, ye = jnp.asarray(ref["te"]), ref["ye"]

    # fine ramp (Abaqus integrates at 1e-4 s) + the evaluation hold times
    t_fine = np.concatenate([np.linspace(0, ref["tau0"], 385), ref["te"][ref["te"] > ref["tau0"] + 1e-9]])

    def standin_curve(K, n0, g, alpha=al_tab, mu=mu_tab, cv=ref):
        t = jnp.asarray(t_fine)
        se = qlv.elastic_standin(S.depth_history(t, cv["d"], cv["tau0"]), alpha, cv["phi"], K, n0)
        return np.interp(cv["te"], t_fine, mu * np.asarray(qlv.qlv_response(se, t, jnp.asarray(g))))

    lin = lambda s: (s - ye) / ye.max()
    res = least_squares(lambda p: lin(standin_curve(p[0], p[1], g_tab)), [2.0, 1.0],
                        bounds=([0.01, 0.3], [100, 3]), diff_step=1e-3)
    K, n0 = map(float, res.x)
    fit_tab = standin_curve(K, n0, g_tab)
    relerr_tab = float(np.linalg.norm(fit_tab - ye) / np.linalg.norm(ye))
    # classical baseline: same physics, fit g's + elastic law directly (no ML)
    res2 = least_squares(lambda p: lin(standin_curve(p[0], p[1], p[2:5])), [K, n0, .3, .3, .1],
                         bounds=([0.01, 0.3, 0, 0, 0], [100, 3, 3, 3, 3]), diff_step=1e-3)
    R["calibration"] = dict(K=K, n0=n0, c_assumed=qlv.STANDIN_C, rel_l2_table1_qlv_vs_abaqus=relerr_tab,
                            classical_lsq_g=res2.x[2:5].tolist(),
                            sigmaE_over_mu_at_ref=float(ye[-1] / mu_tab))
    print(f"calibration K={K:.3f} n0={n0:.3f}; Table-1 QLV vs Abaqus rel L2 = {relerr_tab:.2e}; "
          f"classical LSQ g = {np.round(res2.x[2:5], 3)}")
    json.dump(dict(K=K, n0=n0, c=qlv.STANDIN_C), open(os.path.join(a.out, "..", "calibration.json"), "w"), indent=1)
    c_ = n0

    # Fig 1: Abaqus vs QLV + the basis decomposition (eq. 13)
    t_np = t_fine
    se = mu_tab * qlv.elastic_standin(S.depth_history(jnp.asarray(t_fine), ref["d"], ref["tau0"]),
                                      al_tab, ref["phi"], K, n0)
    B = np.asarray(qlv.qlv_basis(se, jnp.asarray(t_fine)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for axx, xs in zip(ax, ["linear", "log"]):
        axx.plot(ref["t"], ref["y"] * 1e3, color=INK, lw=3, alpha=0.35, label="Abaqus (real, 4.95 mm / 1 mm)")
        axx.plot(ref["te"], fit_tab * 1e3, color=COL["QLV-DeepONet"], ls="--", label="QLV, Table-1 g, τ (exact physics)")
        if xs == "log":
            for k, lab in enumerate(["σ_E (elastic part)", "g₁·B₁ (τ₁=0.1 s)", "g₂·B₂ (τ₂=1 s)", "g₃·B₃ (τ₃=10 s)"]):
                w = 1 if k == 0 else g_tab[k - 1]
                axx.plot(t_np[1:], w * B[k][1:] * 1e3, lw=1.3, color=[MUTED, "#eb6834", "#1baf7a", "#6250d6"][k], label=lab)
            axx.set_xscale("log"); axx.set_xlim(8e-3, 11)
        axx.set_xlabel("time [s]")
        axx.set_ylabel("indenter output ×10³ (likely MPa → kPa)")
    ax[0].set_title(f"Abaqus curve vs QLV model: rel. L2 error {relerr_tab:.2%}")
    ax[1].set_title("Same curve split into its Prony 'basis functions'")
    ax[0].legend(); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig1_abaqus_vs_qlv.png"), dpi=160); plt.close(fig)

    # ------------------------------------------------------------ 2. synthetic data
    rng = np.random.default_rng(0)
    pool = S.make_dataset(rng, max(Ns), K, c_)
    test = S.make_dataset(np.random.default_rng(1), 256, K, c_)
    ood = S.make_dataset(np.random.default_rng(2), 256, K, c_, ranges=S.OOD_RANGES)

    # ------------------------------------------------------------ 3. train surrogates
    R["forward"] = {}
    trained = {}
    for name in ["DeepONet", "QLV-DeepONet"]:
        for N in Ns:
            sub = {k: v[:N] for k, v in pool.items()}
            p, wall = T.train(name, sub, steps=steps, verbose=False)
            e_te, _ = T.rel_l2(name, p, test)
            e_ood, _ = T.rel_l2(name, p, ood)
            R["forward"][f"{name}/N={N}"] = dict(train_s=wall, test_median=float(np.median(e_te)),
                                                test_p90=float(np.percentile(e_te, 90)),
                                                ood_median=float(np.median(e_ood)))
            trained[(name, N)] = p
            print(f"{name:13s} N={N:5d}  test median rel L2 {np.median(e_te):.2%}  "
                  f"OOD {np.median(e_ood):.2%}  ({wall:.0f}s)")

    # Fig 2: data efficiency
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for name in ["DeepONet", "QLV-DeepONet"]:
        for axx, key, ttl in zip(ax, ["test_median", "ood_median"],
                                 ["In-distribution test", "Out-of-distribution (slower ramp τ₀∈[1,3] s, deeper d∈[0.5,0.7])"]):
            ys = [R["forward"][f"{name}/N={N}"][key] * 100 for N in Ns]
            axx.plot(Ns, ys, marker="o", ms=7, color=COL[name], label=name)
            axx.annotate(f"{ys[-1]:.2g}%", (Ns[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
                         va="center", fontsize=8, color=MUTED)
            axx.set_xscale("log"); axx.set_yscale("log"); axx.set_title(ttl, fontsize=9)
            axx.set_xlabel("number of training simulations (synthetic)")
    ax[0].set_ylabel("median relative L2 error [%]"); ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_data_efficiency.png"), dpi=160); plt.close(fig)

    # Fig 3: example test curves at small N (where the difference shows)
    Nsm = Ns[1]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for i, axx in enumerate(ax):
        th, t, y = test["theta"][i], test["t"][i], test["y"][i]
        axx.plot(t[1:], y[1:], color=INK, lw=3, alpha=0.35, label="truth (synthetic)")
        for name in ["DeepONet", "QLV-DeepONet"]:
            pr = np.asarray(T.predict(name, trained[(name, Nsm)], jnp.asarray(th), jnp.asarray(t)))
            axx.plot(t[1:], pr[1:], color=COL[name], lw=1.6, label=f"{name} (N={Nsm})")
        axx.set_xscale("log"); axx.set_xlabel("time [s]")
        axx.set_title(f"φ={th[0]:.1f} mm, δ/φ={th[1]:.2f}, α={th[2]:.1f}\n"
                      f"g=({th[3]:.2f},{th[4]:.2f},{th[5]:.2f}), τ₀={th[6]:.2f} s", fontsize=8)
    ax[0].set_ylabel("σ / μ"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_examples.png"), dpi=160); plt.close(fig)

    # ------------------------------------------------------------ 4. forward on REAL curves
    Nbig = Ns[-1]
    R["real_forward"] = {}
    fig, ax = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 3.8), squeeze=False)
    for j, c in enumerate(curves):
        th = jnp.array([c["phi"], c["d"], al_tab, *g_tab, c["tau0"]], jnp.float32)
        axx = ax[0, j]
        axx.plot(c["t"], c["y"] * 1e3, color=INK, lw=3, alpha=0.35, label="Abaqus (real)")
        for name in ["DeepONet", "QLV-DeepONet"]:
            t0 = time.time()
            pr = mu_tab * np.asarray(T.predict(name, trained[(name, Nbig)], th, jnp.asarray(c["te"])))
            err = float(np.linalg.norm(pr - c["ye"]) / np.linalg.norm(c["ye"]))
            R["real_forward"][f"{os.path.basename(c['path'])}/{name}"] = err
            axx.plot(c["te"], pr * 1e3, color=COL[name], lw=1.6, ls="--", label=f"{name}: {err:.2%}")
        axx.set_xscale("symlog", linthresh=0.1); axx.set_xlabel("time [s]")
        axx.set_title(f"φ={c['phi']} mm, δ={c['delta']} mm (Table-1 material)", fontsize=9)
        axx.legend(fontsize=8)
    ax[0, 0].set_ylabel("indenter output ×10³")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_real_forward.png"), dpi=160); plt.close(fig)
    # timing
    name = "QLV-DeepONet"
    f = jax.jit(lambda th, t: T.predict(name, trained[(name, Nbig)], th, t))
    th = jnp.array([ref["phi"], ref["d"], al_tab, *g_tab, ref["tau0"]], jnp.float32)
    f(th, te).block_until_ready(); t0 = time.time()
    for _ in range(100):
        f(th, te).block_until_ready()
    R["surrogate_ms_per_curve"] = (time.time() - t0) * 10
    print(f"surrogate forward: {R['surrogate_ms_per_curve']:.2f} ms per curve (Abaqus: ~10 h)")

    # ------------------------------------------------------------ 5. inverse problem
    # 5a. REAL curve: recover g1..g3 (+ mu, alpha) from the Abaqus curve alone
    R["real_inverse"] = {}
    rc = [dict(phi=ref["phi"], d=ref["d"], tau0=ref["tau0"], t=ref["te"], y=ref["ye"])]
    for name in ["DeepONet", "QLV-DeepONet"]:
        p = trained[(name, Nbig)]
        est, loss, wall = T.invert(name, p, rc, steps=400 if a.quick else 800)
        est_a, _, _ = T.invert(name, p, rc, steps=400 if a.quick else 800, fixed_alpha=al_tab)
        cov, keep = T.laplace_cov(name, p, est_a, rc, fixed_alpha=al_tab)
        sd = np.sqrt(np.diag(cov))
        R["real_inverse"][name] = dict(
            all_free=dict(mu=float(np.exp(est[0])), alpha=float(est[1]), g=est[2:].tolist(), seconds=wall),
            alpha_known=dict(mu=float(np.exp(est_a[0])), g=est_a[2:].tolist(), g_sd_1pct_noise=sd[1:].tolist()))
        print(f"[real inverse, {name}] g = {np.round(est[2:], 3)} (all free, {wall:.1f}s), "
              f"{np.round(est_a[2:], 3)} (alpha known)   Table 1: {g_tab}")

    # 5b. SYNTHETIC benchmark: joint elastic + viscous identification from 3 depths
    nm = 12 if a.quick else 60
    rngi = np.random.default_rng(3)
    truth, ests = [], {n: [] for n in ["DeepONet", "QLV-DeepONet"]}
    t_inv = time.time()
    for m in range(nm):
        mu = 10 ** rngi.uniform(-3, -2)        # 1-10 kPa in MPa
        alpha = rngi.uniform(2.5, 7.5)
        g = rngi.uniform(0.05, 0.9, 3)
        cs = []
        for dd in [0.1, 0.2, 0.4]:
            th = jnp.array([6.33, dd, alpha, *g, 0.12], jnp.float32)
            t, y = S.truth_curve(th, K, c_)
            y = mu * np.asarray(y) * (1 + 0.01 * rngi.standard_normal(len(y)))   # 1% noise
            y[0] = 0.0
            cs.append(dict(phi=6.33, d=dd, tau0=0.12, t=np.asarray(t), y=np.abs(y) + 1e-12))
        truth.append([np.log(mu), alpha, *g])
        for name in ests:
            est, _, _ = T.invert(name, trained[(name, Ns[-2] if len(Ns) > 2 else Ns[-1])], cs,
                                 steps=400 if a.quick else 800, seed=m)
            ests[name].append(est)
    truth = np.array(truth)
    R["synthetic_inverse"] = {"n_materials": nm, "noise": 0.01, "depths": [0.1, 0.2, 0.4]}
    labels = ["μ (rel. err)", "α (abs. err)", "g₁ (abs. err)", "g₂ (abs. err)", "g₃ (abs. err)"]
    for name, E in ests.items():
        E = np.array(E)
        err = np.abs(E - truth); err[:, 0] = np.abs(np.exp(E[:, 0] - truth[:, 0]) - 1)
        R["synthetic_inverse"][name] = {l: float(np.median(err[:, i])) for i, l in enumerate(labels)}
        print(f"[synthetic inverse, {name}] median errors:",
              {l: round(float(np.median(err[:, i])), 4) for i, l in enumerate(labels)})
    R["synthetic_inverse"]["seconds_per_material"] = (time.time() - t_inv) / nm / 2

    # Fig 5: inverse results
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    names_p = ["α", "g₁", "g₂", "g₃"]
    for i in range(3):
        k = i + 2
        axx = ax[i]
        for name in ests:
            E = np.array(ests[name])
            axx.scatter(truth[:, k], E[:, k], s=18, color=COL[name], alpha=0.8, label=name,
                        edgecolor="white", linewidth=0.6)
        axx.plot([0, 1], [0, 1], color=MUTED, lw=1)
        axx.set_xlabel(f"true {names_p[k - 1]}"); axx.set_ylabel(f"recovered {names_p[k - 1]}")
        axx.set_title(f"Synthetic: {names_p[k - 1]} from 3 depths, 1% noise", fontsize=9)
    ax[0].legend(fontsize=8)
    axx = ax[3]
    x = np.arange(3); w = 0.26
    axx.bar(x - w, g_tab, w, color=MUTED, label="Abaqus input (Table 1)")
    for s, name in zip([0, 1], ["DeepONet", "QLV-DeepONet"]):
        r = R["real_inverse"][name]["alpha_known"]
        axx.bar(x + s * w, r["g"], w, yerr=r["g_sd_1pct_noise"], color=COL[name], label=name, capsize=3)
    axx.set_xticks(x, ["g₁", "g₂", "g₃"]); axx.set_title("REAL Abaqus curve: recovered g (α known)", fontsize=9)
    axx.set_ylim(0, 0.75); axx.legend(fontsize=8, loc="upper right"); axx.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig5_inverse.png"), dpi=160); plt.close(fig)

    json.dump(R, open(os.path.join(a.out, "..", "results.json"), "w"), indent=1)
    print("wrote results.json and figures/")


if __name__ == "__main__":
    main()
