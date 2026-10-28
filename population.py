"""Synthetic population by smoothed-bootstrap (Gaussian-kernel) resampling of the parent
sample in log space (paper Sect. 3.6), followed by the figure of merit F.

By default F is the first-order form of Eq. (8) including the Galactic term and the
Galactic-model term sigma_mod (tabulated along each sightline and interpolated in distance), as
used in the paper. full=False gives the simplified Shklovskii-only F (Eq. 9)."""
import numpy as np, pandas as pd

from physics import figure_of_merit, pbdot_gr, pbdot_shk, m2_solve, DAY
from catalogue import comp_class
import budget

M1_PRIOR = {"NS": (1.33, 0.09), "default": (1.54, 0.23)}
M1_LIM = (1.17, 2.30)
M2_LIM = {"NS": (1.10, 2.30), "He": (0.10, 0.50), "CO": (0.40, 1.40),
          "MS": (0.08, 10.0), "UL": (0.005, 0.10), "UNK": (0.005, 2.30)}
KDE_COLS = ["d", "mu", "pb_s", "ecc", "fmass", "sig_obs"]


def assign_masses(f, cls, rng, tries=40):
    """Masses are needed only for the GR prediction (used to flag systems where a
    radiative test is feasible); F itself does not depend on them."""
    n = len(f)
    m1 = np.full(n, np.nan); m2 = np.full(n, np.nan)
    todo = np.arange(n)
    for _ in range(tries):
        if todo.size == 0:
            break
        c = cls[todo]
        mu_ = np.where(c == "NS", M1_PRIOR["NS"][0], M1_PRIOR["default"][0])
        sd_ = np.where(c == "NS", M1_PRIOR["NS"][1], M1_PRIOR["default"][1])
        m1t = np.clip(rng.normal(mu_, sd_), *M1_LIM)
        ct = rng.uniform(0, 1, todo.size)
        m2t = m2_solve(f[todo], m1t, np.sqrt(1 - ct**2))
        lo = np.array([M2_LIM.get(x, M2_LIM["UNK"])[0] for x in c])
        hi = np.array([M2_LIM.get(x, M2_LIM["UNK"])[1] for x in c])
        ok = np.isfinite(m2t) & (m2t >= lo) & (m2t <= hi)
        g = todo[ok]
        m1[g], m2[g] = m1t[ok], m2t[ok]
        todo = todo[~ok]
    return m1, m2


def _F_and_parts(pb, mu, d, sdf, smf, sig, l, b, full):
    """First-order F and its three components (zeros for F_mu-only/F_mod when full=False)."""
    if full:
        return budget.components(pb, mu, d, sdf, smf, sig, l, b)
    F = figure_of_merit(pb, mu, d, sdf, smf, sig, l, b, False)
    z = np.zeros(len(F))
    return z, z, z, np.asarray(F)


def generate(parent, n, seed=42, bw=1.0, full=True):
    rng = np.random.default_rng(seed)
    u = np.log10(parent[KDE_COLS].values)
    np_, dim = u.shape
    fac = np_ ** (-2 / (dim + 4))                     # Scott's rule, squared
    cov = np.cov(u, rowvar=False) * fac * bw**2
    s = 1 / np.sqrt(1 + fac * bw**2)                  # variance-preserving shrinkage
    m = u.mean(0)
    idx = rng.integers(0, np_, n)
    smp = m + (u[idx] - m) * s + rng.multivariate_normal(np.zeros(dim), cov, n) * s
    ecc = 10 ** smp[:, 3]
    bad = (ecc >= 0.95) | (ecc < 1e-8)
    n_iter = 0
    while bad.any() and n_iter < 50:                  # redraw eccentricity only
        n_iter += 1
        r = rng.integers(0, np_, bad.sum())
        k2 = rng.multivariate_normal(np.zeros(dim), cov, bad.sum())
        ecc[bad] = 10 ** (m[3] + (u[r, 3] - m[3]) * s + k2[:, 3] * s)
        bad = (ecc >= 0.95) | (ecc < 1e-8)
    d, mu, pb, fm, sig = (10 ** smp[:, i] for i in (0, 1, 2, 4, 5))
    cls = comp_class(parent["bincomp"].values[idx])
    m1, m2 = assign_masses(fm, cls, rng)

    p = pd.DataFrame({"parent_jname": parent["jname"].values[idx], "comp_class": cls,
                      "d": d, "mu": mu, "pb_s": pb, "pb_d": pb / DAY, "ecc": ecc,
                      "fmass": fm, "sig_obs": sig, "m1": m1, "m2": m2,
                      "gl": parent["gl"].values[idx], "gb": parent["gb"].values[idx],
                      # measurement-error properties are inherited from the parent system
                      "sigma_d_frac": parent["sigma_d_frac"].values[idx],
                      "sigma_mu_frac": parent["sigma_mu_frac"].values[idx],
                      "dist_method": parent["dist_method"].values[idx]})
    keep = ~bad
    stats = {"n_drawn": int(n), "n_ecc_rejected": int(bad.sum()), "ecc_redraw_iterations": n_iter,
             "n_mass_unassigned": int((~np.isfinite(m1)).sum())}
    p = p[keep].reset_index(drop=True)

    fd, fmu, fmod, F = _F_and_parts(p.pb_s.values, p.mu.values, p.d.values,
                                    p.sigma_d_frac.values, p.sigma_mu_frac.values,
                                    p.sig_obs.values, p.gl.values, p.gb.values, full)
    p["Fd"], p["Fmu"], p["Fmod"], p["F"] = fd, fmu, fmod, F
    p["lever"] = np.array(["d", "mu", "Gal"])[np.argmax(np.column_stack([fd, fmu, fmod]), axis=1)]
    p["abs_b"] = np.abs(p.gb)
    p["log10F"] = np.log10(p.F)
    p["pbdot_shk"] = pbdot_shk(p.pb_s, p.mu, p.d)
    p["pbdot_gr"] = pbdot_gr(p.pb_s, p.ecc, p.m1, p.m2)        # NaN where masses unassigned
    p["gr_snr"] = np.abs(p.pbdot_gr) / p.sig_obs
    p["v_trans"] = 4.74047 * p.mu * p.d

    # "Observed" F: what an observer computes from noisy d and mu with the same quoted errors
    d_obs = p.d.values * np.exp(rng.normal(0, np.log1p(p.sigma_d_frac.values)))
    mu_obs = np.abs(p.mu.values * (1 + rng.normal(0, p.sigma_mu_frac.values)))
    Fo = _F_and_parts(p.pb_s.values, mu_obs, d_obs, p.sigma_d_frac.values,
                      p.sigma_mu_frac.values, p.sig_obs.values, p.gl.values, p.gb.values, full)[3]
    p["d_obs"], p["mu_obs"], p["F_obs"], p["log10F_obs"] = d_obs, mu_obs, Fo, np.log10(Fo)
    return p, stats
