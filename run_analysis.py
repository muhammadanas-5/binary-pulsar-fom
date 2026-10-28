#!/usr/bin/env python3
"""Catalogue stage: ATNF query -> parent sample -> exclusions -> cross-check with Table 1.

Writes (in --outdir):
    parent_sample.csv          binaries with every input the figure of merit needs
    catalogue_sample.csv       parent sample after the exclusions of paper Sect. 3.5
    excluded_systems.csv       every excluded system and the reason
    crosscheck_table1.csv      Shklovskii-only F_Shk from the catalogue vs Table 1 (F_shk_paper)

The full figure of merit is computed by run_upgrade.py, which imports build_catalogue_sample().
"""
import os, json, argparse
import numpy as np, pandas as pd

import catalogue as cat
from physics import figure_of_merit


def build_catalogue_sample(cache="atnf_binaries.csv", refresh=False, dm_err=0.25, px_snr=2.0,
                           mock=False, outdir=None, parent_csv=None):
    """parent_csv: start from an already-built parent sample (columns of parent_sample.csv)
    instead of querying ATNF, e.g. results of an earlier run."""
    if parent_csv:
        parent = pd.read_csv(parent_csv).drop(columns=["F", "log10F"], errors="ignore")
        meta = {"catalogue_version": "2.8.1 (from " + os.path.basename(parent_csv) + ")", "n_rows": None}
        pinfo = {"n_input": int(len(parent)), "n_kept": int(len(parent)),
                 "n_pbdot_rescaled_from_1e-12": None, "n_dropped_missing_or_nonpositive": None}
    else:
        raw, meta = (cat.mock(), {"catalogue_version": "MOCK - NOT REAL DATA"}) if mock else \
            cat.query_atnf(cache, refresh)
        parent, pinfo = cat.parent_sample(raw, dm_err, px_snr)
    kept, excluded = cat.apply_exclusions(parent)
    info = {**pinfo, "n_excluded": int(len(excluded)), "n_final": int(len(kept)),
            "excluded_by_reason": excluded.reason.value_counts().to_dict(),
            "n_excluded_catalogue_artefact_sigma": int(excluded.catalogue_artefact_sigma.sum()),
            "n_catalogue_rows": meta.get("n_rows")}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        parent.to_csv(os.path.join(outdir, "parent_sample.csv"), index=False)
        kept.to_csv(os.path.join(outdir, "catalogue_sample.csv"), index=False)
        excluded.to_csv(os.path.join(outdir, "excluded_systems.csv"), index=False)
    return kept, excluded, meta, info


def crosscheck_table1(kept, table1_path):
    t1 = pd.read_csv(table1_path)
    k = kept.copy()
    k["F_shk_cat"] = figure_of_merit(k.pb_s, k.mu, k.d, k.sigma_d_frac, k.sigma_mu_frac,
                                     k.sig_obs, k.gl, k.gb, False)
    x = t1.merge(k[["jname", "F_shk_cat"]], on="jname", how="left")
    x["ratio_catalogue_to_paper"] = x.F_shk_cat / x.F_shk_paper
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="atnf_binaries.csv")
    ap.add_argument("--refresh", action="store_true", help="re-query ATNF (needs network)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--dm-dist-err", type=float, default=0.25)
    ap.add_argument("--table1", default="paper_table1.csv")
    ap.add_argument("--mock", action="store_true", help="fake catalogue; pipeline test only")
    a = ap.parse_args()
    kept, excluded, meta, info = build_catalogue_sample(a.cache, a.refresh, a.dm_dist_err,
                                                         mock=a.mock, outdir=a.outdir)
    x = crosscheck_table1(kept, a.table1)
    x.to_csv(os.path.join(a.outdir, "crosscheck_table1.csv"), index=False)
    json.dump({"catalogue": meta, "selection": info}, open(os.path.join(a.outdir, "catalogue_report.json"), "w"), indent=2)
    print(f"catalogue {meta.get('catalogue_version')}: {info['n_catalogue_rows']} rows -> "
          f"{info['n_input']} parent -> {info['n_final']} after exclusions ({info['n_excluded']} excluded)")
    for r, n in info["excluded_by_reason"].items():
        print(f"  excluded {n:3d}  {r}")
    print("Table 1 cross-check (Shklovskii-only F, catalogue / paper):")
    for r in x.itertuples():
        print(f"  {r.label:14s} paper {r.F_shk_paper:<7g} catalogue "
              f"{'missing' if pd.isna(r.F_shk_cat) else f'{r.F_shk_cat:.3g}':>8s}")


if __name__ == "__main__":
    main()
