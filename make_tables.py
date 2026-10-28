#!/usr/bin/env python3
"""Write the LaTeX tables of the paper from the results of run_upgrade.py.

    python make_tables.py [--results results] [--out tables]

Produces tables/table1_results.tex, table1_inputs.tex and table2_catalogue.tex, so that the
numbers in the paper are exactly those in table1_full.csv and table2_catalogue.csv.
"""
import os, argparse
import numpy as np, pandas as pd

# Input strings exactly as typeset in the paper: d (kpc) and Pbdot_obs (1e-12 s/s)
D_STR = {"J0348+0432": "2.10(20)", "J2234+0611": "0.95(4)", "J1913+1102": "7.14(1.5)",
         "J2222-0137": "0.267(1)", "J1946+2052": "3.5(1.0)", "J1756-2251": "0.73(40)",
         "J0737-3039A/B": "0.735(60)", "J1829+2456": "1.2(3)", "J1738+0333": "1.47(10)",
         "J1012+5307": "0.83(5)", "J1933-6211": "1.6(3)", "J2145-0750": "0.50(15)",
         "J1757-1854": "11.5(3.0)", "B1913+16": "9.0(2.0)", "J1909-3744": "1.158(3)",
         "B1534+12": "0.94(7)"}
PB_STR = {"J0348+0432": "-0.273(45)", "J2234+0611": "+3.1(2.5)", "J1913+1102": "-0.48(3)",
          "J2222-0137": "+0.251(8)", "J1946+2052": "-1.8296(15)", "J1756-2251": "-0.229(5)",
          "J0737-3039A/B": "-1.247920(78)", "J1829+2456": "-0.029(12)", "J1738+0333": "-0.0170(31)",
          "J1012+5307": "+0.061(5)", "J1933-6211": "+0.7(1)", "J2145-0750": "+0.13(2)",
          "J1757-1854": "-5.294(6)", "B1913+16": "-2.423(1)", "J1909-3744": "+0.5093(7)",
          "B1534+12": "-0.1366(3)"}
LEVER = {"d": r"$d$", "mu": r"$\mu$", "Gal": "Gal"}


def g(x, n=3):
    """n significant figures, no exponent, trailing zeros dropped."""
    x = float(x)
    if x == 0:
        return "0"
    s = f"{x:.{n}g}"
    if "e" in s:
        s = f"{x:.{n}f}"
    return s


def tex(s):
    return s.replace("-", "$-$").replace("+0", "+0") if False else s.replace("-", "$-$")


def name(s):
    return tex(s)


def rng(lo, hi):
    return f"{g(lo)}--{g(hi)}"


def table1_results(t):
    L = []
    for i, r in enumerate(t.itertuples(), 1):
        L.append(" & ".join([str(i), name(r.label), r.dist_type.replace("Pbdot", r"$\dot P_{\rm b}$"),
                             g(r.d), g(r.F_shk), g(r.F), rng(r.F_lo, r.F_hi), g(r.Fd, 2), g(r.Fmu, 2),
                             g(r.Fmod, 2), LEVER[r.lever], f"{r.cls} ({r.p_cls:.2f})"]) + "")
    return " \\\\\n".join(L)


def table1_inputs(t):
    L = []
    for i, r in enumerate(t.itertuples(), 1):
        smu = "---" if r.sigma_mu_frac == 0 else f"{r.sigma_mu_frac:.4f}"
        L.append(" & ".join([str(i), name(r.label), r.comp, g(r.pb_d, 5), g(r.mu, 4), smu,
                             D_STR[r.label], tex(PB_STR[r.label]), r.dist_type.replace("Pbdot", r"$\dot P_{\rm b}$"),
                             str(r.ref)]) + "")
    return " \\\\\n".join(L)


def table2(c, lit_j):
    L = []
    for i, r in enumerate(c.itertuples(), 1):
        nm = name(r.jname)
        alias = {"J1915+1606": " (B1913+16)", "J1537+1155": " (B1534+12)"}.get(r.jname, "")
        star = "$^{*}$" if r.jname in lit_j else ""
        meth = "plx" if r.dist_method == "parallax" else "DM"
        L.append(" & ".join([str(i), nm + alias + star, g(r.pb_s / 86400, 4), g(r.mu, 3), g(r.d, 3),
                             f"{r.sigma_d_frac:.3f}", meth, f"{r.sigma_mu_frac:.3f}", g(r.sig_obs / 1e-12, 2),
                             g(r.F_shk), g(r.F), rng(r.F_lo, r.F_hi), LEVER[r.lever],
                             f"{r.cls} ({r.p_cls:.2f})"]) + "")
    return " \\\\\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="tables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(os.path.join(a.results, "table1_full.csv"))
    c = pd.read_csv(os.path.join(a.results, "table2_catalogue.csv"))
    w = lambda f, s: open(os.path.join(a.out, f), "w").write(s + "\n")
    w("table1_results.tex", table1_results(t)); w("table1_inputs.tex", table1_inputs(t))
    w("table2_catalogue.tex", table2(c, set(t.jname)))
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()
