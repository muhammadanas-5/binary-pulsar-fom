#!/usr/bin/env python3
"""Full analysis: figure of merit for the literature sample (Table 1), the ATNF catalogue sample
(Table 2) and the synthetic population (Sect. 3.6); writes all tables, figures and a report.

    python run_upgrade.py                    # needs atnf_binaries.csv (or network for psrqpy)
    python run_upgrade.py --lit-only         # literature sample only (no catalogue needed)

Seeds: 0 for the per-system Monte Carlo, 42 for the synthetic population.
Outputs (in --outdir): table1_full.csv, table1_vs_paper.csv, table2_catalogue.csv,
synthetic_summary.json, report.json and fig1.pdf ... fig4.pdf.
"""
import os, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import lit_inputs, budget, population as popmod, analysis as ana
from analysis import regimes
from run_analysis import build_catalogue_sample

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
PT = np.log10(budget.THRESHOLDS)
LEV = {"d": ("tab:blue", "o"), "mu": ("tab:orange", "s"), "Gal": ("tab:green", "^")}


def fig1(t, path):
    t = t.sort_values("F").reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    geo = ~t.dist_type.isin(["DM", "unknown"])
    for filled, m in ((True, geo), (False, ~geo)):
        s = t[m]
        a.errorbar(s.d, s.F, yerr=[np.clip(s.F - s.F_lo, 0, None), np.clip(s.F_hi - s.F, 0, None)], fmt="o", color="k", ms=5,
                   mfc="k" if filled else "w", lw=0.8, capsize=0,
                   label="geometric/photometric/$\\dot P_b$" if filled else "DM / unknown")
    for i, r in t.iterrows():
        a.annotate(str(i + 1), (r.d, r.F), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for v, ls in zip(budget.THRESHOLDS, (":", "--")):
        a.axhline(v, color="0.35", ls=ls, lw=0.9)
    a.set_yscale("log"); a.set_xlabel("Distance $d$ (kpc)"); a.set_ylabel(r"$\mathcal{F}$")
    a.legend(fontsize=7, frameon=False)
    y = np.arange(len(t)) + 1
    for key, (c, mk), lab in (("Fd", LEV["d"], "distance"), ("Fmu", LEV["mu"], "proper motion"),
                              ("Fmod", LEV["Gal"], "Galactic model")):
        b.scatter(t[key].clip(lower=1e-3), y, c=c, marker=mk, s=18, label=lab, zorder=3)
    b.scatter(t.F, y, marker="|", c="k", s=120, label="total $\\mathcal{F}$", zorder=4)
    b.set_yticks(y, [f"{i} {n}" for i, n in zip(y, t.label)], fontsize=7)
    for v, ls in zip(budget.THRESHOLDS, (":", "--")):
        b.axvline(v, color="0.35", ls=ls, lw=0.9)
    b.set_xscale("log"); b.set_xlim(1e-3, 1e2); b.set_xlabel("component of $\\mathcal{F}$")
    b.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig2(p, cat, lit, gmm, path):
    lf = p.log10F.values
    lo, hi = np.percentile(lf, [0.1, 99.9])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(lf, 180, range=(lo, hi), density=True, color="0.85", edgecolor="0.6", lw=0.3,
            label="synthetic")
    kd = ana.antimodes(lf)
    ax.plot(kd["grid"], kd["density"], "k", lw=1.3, label="KDE")
    g, grid = gmm["models"][1], np.linspace(lo, hi, 800)
    m_, s_ = g.means_.ravel()[0], np.sqrt(g.covariances_.ravel()[0])
    ax.plot(grid, np.exp(-0.5 * ((grid - m_) / s_) ** 2) / (s_ * np.sqrt(2 * np.pi)), "--", lw=1.0,
            label="single Gaussian (BIC)")
    top = ax.get_ylim()[1]
    for v, tx in zip(PT, ("$\\mathcal{F}$=0.3", "$\\mathcal{F}$=1")):
        ax.axvline(v, color="0.35", ls="--", lw=0.9); ax.text(v, top * 0.97, " " + tx, fontsize=7, va="top")
    if cat is not None:
        ax.plot(np.log10(cat.F), np.full(len(cat), -0.02 * top), "|", color="tab:orange", ms=8,
                clip_on=False, label=f"catalogue (N={len(cat)})")
    ax.plot(np.log10(lit.F), np.full(len(lit), -0.06 * top), "v", color="k", ms=4, clip_on=False,
            label="Table 1")
    ax.set_ylim(-0.08 * top, top); ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\log_{10}\mathcal{F}$"); ax.set_ylabel("density"); ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig3(p, path):
    ms = ana.misclass(p, PT); M = ms["matrix"]; Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    im = a.imshow(Mn, cmap="magma_r", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                   color="k" if Mn[i, j] < 0.6 else "w")
    a.set_xticks(range(3), ["M", "B", "D"]); a.set_yticks(range(3), ["M", "B", "D"])
    a.set_xlabel("inferred"); a.set_ylabel("true")
    a.set_title(f"overall misclassification {ms['overall']:.3f}", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)
    p = p.assign(grp=p.lever.map({"d": "distance", "mu": "proper motion", "Gal": "Galactic model"})
                 + "\n" + np.where(p.dist_method == "parallax", "plx", "DM"))
    r = ana.misclass_by(p, PT, "grp"); keys = list(r)
    b.barh(keys, [r[k]["rate"] for k in keys])
    for i, k in enumerate(keys):
        b.text(r[k]["rate"], i, f' {100 * r[k]["share"]:.0f}% of pop.', va="center", fontsize=7)
    b.set_xlabel("misclassification rate"); b.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig4(p, path):
    lf = p.log10F.values
    sel = np.random.default_rng(0).choice(len(p), min(6000, len(p)), replace=False)
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.4))
    panels = [(np.log10(p.mu), r"$\log\mu$"), (np.log10(p.d), r"$\log d$"),
              (np.log10(p.pb_d), r"$\log P_b$"), (np.log10(p.sig_obs), r"$\log\sigma(\dot P_b^{\rm obs})$"),
              (p.abs_b, r"$|b|$ (deg)")]
    for a, (v, lab) in zip(ax.ravel(), panels):
        a.scatter(np.asarray(v)[sel], lf[sel], s=1.5, alpha=0.15, color="k", lw=0)
        for x in PT:
            a.axhline(x, color="0.35", ls="--", lw=0.9)
        a.set_xlabel(lab, fontsize=9); a.set_ylabel(r"$\log_{10}\mathcal{F}$", fontsize=9)
    a = ax.ravel()[5]; fr = ana.lever_fractions(p, PT); w = 0.27
    for k, (n, (c, _)) in enumerate(LEV.items()):
        a.bar(np.arange(3) + (k - 1) * w, [fr[c_][n] or 0 for c_ in "MBD"], w, color=c,
              label={"d": "distance", "mu": "proper motion", "Gal": "Galactic model"}[n])
    a.set_xticks(range(3), ["M", "B", "D"]); a.set_ylabel("fraction of class")
    a.set_title("dominant component", fontsize=9); a.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="results"); ap.add_argument("--cache", default="atnf_binaries.csv")
    ap.add_argument("--refresh", action="store_true"); ap.add_argument("--table1", default="paper_table1.csv")
    ap.add_argument("--n", type=int, default=200000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-seed", type=int, default=0); ap.add_argument("--dm-dist-err", type=float, default=0.25)
    ap.add_argument("--gr-snr-min", type=float, default=3.0); ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--parent-csv", default=None, help="start from a saved parent sample");
    ap.add_argument("--lit-only", action="store_true"); ap.add_argument("--mock", action="store_true")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    out = lambda f: os.path.join(a.outdir, f)
    rep = {"config": vars(a)}

    # ---- Table 1: literature sample -------------------------------------------------------
    lit = budget.evaluate(lit_inputs.literature_sample(), seed=a.mc_seed).sort_values("F").reset_index(drop=True)
    lit.to_csv(out("table1_full.csv"), index=False)
    pap = pd.read_csv(a.table1)
    cmp_ = lit.merge(pap, on="label")
    cmp_["F_ratio"] = cmp_.F / cmp_.F_paper
    cmp_[["label", "F_shk", "F_shk_paper", "F", "F_paper", "F_ratio", "lever", "lever_paper", "cls",
          "class_paper", "p_cls", "p_class_paper"]].to_csv(out("table1_vs_paper.csv"), index=False)
    rep["table1"] = {"n_class_match": int((cmp_.cls == cmp_.class_paper).sum()),
                     "n_lever_match": int((cmp_.lever == cmp_.lever_paper).sum()),
                     "F_ratio_min_max": [float(cmp_.F_ratio.min()), float(cmp_.F_ratio.max())],
                     "class_counts": lit.cls.value_counts().to_dict(),
                     "n_class_changed_vs_Shk": int((lit.cls != lit.cls_shk).sum())}
    fig1(lit, out("fig1.pdf"))
    print(f"Table 1: classes match {rep['table1']['n_class_match']}/16, levers "
          f"{rep['table1']['n_lever_match']}/16, F ratio {cmp_.F_ratio.min():.2f}-{cmp_.F_ratio.max():.2f}; "
          f"counts {rep['table1']['class_counts']}")
    if a.lit_only:
        json.dump(rep, open(out("report.json"), "w"), indent=2); return

    # ---- Table 2: catalogue sample ----------------------------------------------------------
    kept, excluded, meta, info = build_catalogue_sample(a.cache, a.refresh, a.dm_dist_err,
                                                         mock=a.mock, outdir=a.outdir, parent_csv=a.parent_csv)
    cat = budget.evaluate(kept, seed=a.mc_seed).sort_values("F").reset_index(drop=True)
    cat.to_csv(out("table2_catalogue.csv"), index=False)
    rep["catalogue"] = {"meta": meta, "selection": info, "class_counts": cat.cls.value_counts().to_dict(),
                        "median_F": float(cat.F.median()), "n_class_changed_vs_Shk": int((cat.cls != cat.cls_shk).sum()),
                        "lever_of_D": cat[cat.cls == "D"].lever.value_counts().to_dict()}
    print(f"Catalogue {meta.get('catalogue_version')}: {info['n_final']} systems, "
          f"classes {rep['catalogue']['class_counts']}, median F {cat.F.median():.2f}")

    # ---- Synthetic population ---------------------------------------------------------------
    p, gstats = popmod.generate(kept, a.n, a.seed)
    p.to_parquet(out("population.parquet"))
    lf = p.log10F.values
    gmm = ana.gmm_scan(lf, a.k_max, a.seed)
    kd = ana.antimodes(lf, a.seed)
    ms = ana.misclass(p, PT)
    rt = p[p.gr_snr >= a.gr_snr_min]
    fr = lambda q: [float(x) for x in ana.fractions(q.log10F, PT)]
    syn = {"generation": gstats, "n_synthetic": len(p), "sd_log10F": float(lf.std()),
           "gmm_bic": dict(zip(map(str, gmm["k"]), map(float, gmm["bic"]))), "k_bic": gmm["k_bic"],
           "delta_bic_k2_minus_k1": float(gmm["bic"][1] - gmm["bic"][0]),
           "kde_antimodes_log10F": [float(x) for x in kd["antimodes"]],
           "class_fractions_MBD": fr(p), "radiative_subset": {"n": int(len(rt)), "frac_of_pop": len(rt) / len(p),
                                                              "class_fractions_MBD": fr(rt) if len(rt) else None},
           "misclassification": {"overall": ms["overall"], "per_class_MBD": ms["per_regime"],
                                 "by_distance_method": ana.misclass_by(p, PT, "dist_method")},
           "lever_fractions": ana.lever_fractions(p, PT), "spearman": ana.spearman_all(p)}
    json.dump(syn, open(out("synthetic_summary.json"), "w"), indent=2, default=float)
    rep["synthetic"] = syn
    fig2(p, cat, lit, gmm, out("fig2.pdf")); fig3(p, out("fig3.pdf")); fig4(p, out("fig4.pdf"))
    json.dump(rep, open(out("report.json"), "w"), indent=2, default=float)
    print(f"synthetic N={len(p)}: classes {np.round(fr(p), 3)}, radiative subset {np.round(fr(rt), 3)} "
          f"(N={len(rt)}), misclass {ms['overall']:.3f} per class {np.round(ms['per_regime'], 3)}")
    print(f"  GMM k_bic={gmm['k_bic']}, dBIC(k=2)={syn['delta_bic_k2_minus_k1']:+.0f}, sd={lf.std():.2f}, "
          f"antimodes {syn['kde_antimodes_log10F']}")


if __name__ == "__main__":
    main()
