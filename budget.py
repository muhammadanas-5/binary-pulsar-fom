"""Full figure of merit F (paper Sect. 2.3, Eqs. 7-8): first-order components, Monte Carlo
adopted value, lever, class and class probability.

    F = sigma(Pbdot_kin) / sigma(Pbdot_obs),   Pbdot_kin = Pbdot_Shk + Pbdot_Gal

    sigma^2(Pbdot_kin) ~ (d Pbdot_kin/dd * sigma_d)^2 + (2 Pbdot_Shk sigma_mu/mu)^2 + sigma_mod^2

    sigma_mod = | Pbdot_Gal(reference, Eqs. 5-6) - Pbdot_Gal(McMillan17) |  at the nominal d.

Components F_d, F_mu, F_mod are the three terms divided by sigma(Pbdot_obs). The adopted F is
the Monte Carlo value: half the 16th-84th percentile width of Pbdot_kin over draws of (d, mu),
added in quadrature to sigma_mod. Class: M (F<0.3), B (0.3<=F<1), D (F>=1).
"""
import numpy as np, pandas as pd

from physics import pbdot_shk, pbdot_gal, MAS_YR
import galmodel

THRESHOLDS = (0.3, 1.0)
CLASSES = np.array(["M", "B", "D"])


def class_of(F):
    return CLASSES[np.digitize(np.asarray(F, float), THRESHOLDS)]


def sigma_mod(pb, l, b, d):
    """|reference - McMillan17| Galactic term (s/s); vectorised over systems that may share
    sightlines (one tabulated McMillan17 curve per sightline)."""
    pb, l, b, d = (np.atleast_1d(np.asarray(x, float)) for x in np.broadcast_arrays(pb, l, b, d))
    mcm = np.empty(d.shape)
    key = np.round(np.column_stack([l, b]), 3)
    for lb in np.unique(key, axis=0):
        sel = (key == lb).all(axis=1)
        mcm[sel] = galmodel.table(lb[0], lb[1])(pb[sel], d[sel])
    return np.abs(pbdot_gal(pb, l, b, d) - mcm)


def dkin_dd(pb, mu, d, l, b):
    """d(Pbdot_Shk + Pbdot_Gal)/dd (s/s per kpc), central difference for the Galactic part."""
    d = np.asarray(d, float)
    h = 1e-4 * d
    dg = (pbdot_gal(pb, l, b, d + h) - pbdot_gal(pb, l, b, d - h)) / (2 * h)
    return pbdot_shk(pb, mu, d) / d + dg


def components(pb, mu, d, sd_frac, smu_frac, sig_obs, l, b, with_model=True):
    """First-order (F_d, F_mu, F_mod, F). Arrays in, arrays out."""
    pb, mu, d, sd_frac, smu_frac, sig_obs, l, b = (
        np.asarray(x, float) for x in (pb, mu, d, sd_frac, smu_frac, sig_obs, l, b))
    shk = pbdot_shk(pb, mu, d)
    fd = np.abs(dkin_dd(pb, mu, d, l, b)) * sd_frac * d / sig_obs
    fmu = 2 * np.abs(shk) * smu_frac / sig_obs
    fmod = sigma_mod(pb, l, b, d) / sig_obs if with_model else np.zeros_like(fd)
    return fd, fmu, fmod, np.sqrt(fd**2 + fmu**2 + fmod**2)


def f_shk(pb, mu, d, sd_frac, smu_frac, sig_obs):
    """Simplified figure of merit, Eq. (9)."""
    shk = pbdot_shk(pb, mu, d)
    return np.abs(shk) * np.sqrt(np.asarray(sd_frac, float)**2 + (2 * np.asarray(smu_frac, float))**2) \
        / np.asarray(sig_obs, float)


def draw_d(d, sd_frac, method, n, rng):
    """Plausible distances: Gaussian parallax inverted (parallax/VLBI), else log-normal of
    width ln(1 + sigma_d/d) (Sect. 2.3)."""
    if method == "parallax":
        px = rng.normal(1.0 / d, sd_frac / d, n)
        while (px <= 0).any():                      # a distance needs a positive parallax
            bad = px <= 0
            px[bad] = rng.normal(1.0 / d, sd_frac / d, bad.sum())
        return 1.0 / px
    return d * np.exp(rng.normal(0.0, np.log1p(sd_frac), n))


def draw_mu(mu, smu_frac, n, rng):
    return np.abs(rng.normal(mu, smu_frac * mu, n))


def monte_carlo_F(pb, mu, d, sd_frac, smu_frac, sig_obs, l, b, method, n=20000, rng=None):
    """Adopted F: MC width of Pbdot_kin (16th-84th percentile / 2) plus sigma_mod in quadrature."""
    rng = rng or np.random.default_rng(0)
    ds, ms = draw_d(d, sd_frac, method, n, rng), draw_mu(mu, smu_frac, n, rng)
    kin = pbdot_shk(pb, ms, ds) + pbdot_gal(pb, l, b, ds)
    lo, hi = np.percentile(kin, [16, 84])
    smod = float(sigma_mod(pb, l, b, d)[0])
    return float(np.hypot(0.5 * (hi - lo), smod) / sig_obs)


def class_probability(pb, mu, d, sd_frac, smu_frac, sig_obs, l, b, method, assigned,
                      n=2000, rng=None):
    """Re-evaluate F at n plausible true (d, mu); return its 16-84th percentile range and the
    probability that the system lies in its assigned class (first-order F at each draw)."""
    rng = rng or np.random.default_rng(0)
    ds, ms = draw_d(d, sd_frac, method, n, rng), draw_mu(mu, smu_frac, n, rng)
    F = components(np.full(n, pb), ms, ds, sd_frac, smu_frac, sig_obs, np.full(n, l),
                   np.full(n, b))[3]
    lo, hi = np.percentile(F, [16, 84])
    return float(lo), float(hi), float((class_of(F) == assigned).mean())


def evaluate(df, n_mc=20000, n_true=2000, seed=0):
    """Full table for a DataFrame with pb_s, mu, d, sigma_d_frac, sigma_mu_frac, sig_obs,
    gl, gb, dist_method. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in df.itertuples():
        a = (r.pb_s, r.mu, r.d, r.sigma_d_frac, r.sigma_mu_frac, r.sig_obs, r.gl, r.gb)
        fd, fmu, fmod, f1 = (float(x[0]) if np.ndim(x) else float(x) for x in components(*a))
        fs = float(f_shk(r.pb_s, r.mu, r.d, r.sigma_d_frac, r.sigma_mu_frac, r.sig_obs))
        F = monte_carlo_F(*a, r.dist_method, n_mc, rng)
        cls = str(class_of(F))
        lo, hi, p = class_probability(*a, r.dist_method, cls, n_true, rng)
        lever = ["d", "mu", "Gal"][int(np.argmax([fd, fmu, fmod]))]
        rows.append(dict(F_shk=fs, Fd=fd, Fmu=fmu, Fmod=fmod, F_first=f1, F=F,
                         F_lo=lo, F_hi=hi, lever=lever, cls=cls, p_cls=p,
                         cls_shk=str(class_of(fs))))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
