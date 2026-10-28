import numpy as np
from scipy.stats import gaussian_kde, spearmanr
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from physics import logF_terms

PAPER_THRESHOLDS_LOG10F = np.log10([0.3, 1.0])    # paper Sect. 2.3 / Table 1 flags


def _sub(x, n, seed):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    r = np.random.default_rng(seed)
    return r.choice(x, n, replace=False) if x.size > n else x


def gmm_scan(logF, kmax=6, seed=0):
    X = _sub(logF, 50000, seed).reshape(-1, 1)
    r = {"k": [], "bic": [], "aic": [], "models": {}}
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=8,
                            random_state=seed, max_iter=1000).fit(X)
        r["k"].append(k); r["bic"].append(g.bic(X)); r["aic"].append(g.aic(X))
        r["models"][k] = g
    r["k_bic"] = r["k"][int(np.argmin(r["bic"]))]
    r["k_aic"] = r["k"][int(np.argmin(r["aic"]))]
    return r


def gmm_bounds(model, lo, hi, ngrid=200000):
    grid = np.linspace(lo, hi, ngrid).reshape(-1, 1)
    order = np.argsort(model.means_.ravel())
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    lab = rank[model.predict(grid)]
    return grid.ravel()[np.nonzero(np.diff(lab) != 0)[0]]


def boot_bounds(logF, k, nboot=200, seed=0, size=20000):
    x = np.asarray(logF, float); x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(x, [0.05, 99.95])
    got = []
    for b in range(nboot):
        g = GaussianMixture(k, covariance_type="full", n_init=2, random_state=b,
                            max_iter=500).fit(rng.choice(x, min(size, x.size)).reshape(-1, 1))
        bb = gmm_bounds(g, lo, hi, 20000)
        if len(bb) == k - 1:
            got.append(bb)
    if not got or k == 1:
        return None
    a = np.array(got)
    return {"median": np.median(a, 0), "lo16": np.percentile(a, 16, 0),
            "hi84": np.percentile(a, 84, 0), "n_valid": len(got), "n_boot": nboot}


def antimodes(logF, seed=0, ngrid=4000):
    x = _sub(logF, 60000, seed)
    grid = np.linspace(x.min(), x.max(), ngrid)
    dens = gaussian_kde(x)(grid)
    return {"grid": grid, "density": dens,
            "antimodes": grid[argrelextrema(dens, np.less)[0]]}


def kmeans_breaks(logF, k=3, seed=0):
    """1-D k-means breaks (closely related to, but not identical with, Jenks natural breaks)."""
    x = _sub(logF, 60000, seed)
    if k < 2:
        return {"breaks": np.array([]), "gvf": 0.0}
    km = KMeans(k, n_init=10, random_state=seed).fit(x.reshape(-1, 1))
    o = np.argsort(km.cluster_centers_.ravel())
    br = [0.5 * (x[km.labels_ == o[i]].max() + x[km.labels_ == o[i + 1]].min())
          for i in range(k - 1)]
    return {"breaks": np.array(br), "gvf": 1 - km.inertia_ / (x.var() * x.size)}


def var_decomp(p, gal=False):
    """Share of Var(log10 F) from each additive term: cov(T_i, logF)/var(logF).
    Exact (shares sum to 1) only for the Shklovskii-only F; with gal=True the
    unexplained remainder is reported as 'residual_gal'."""
    t = logF_terms(p.pb_s.values, p.mu.values, p.d.values, p.sigma_d_frac.values,
                   p.sigma_mu_frac.values, p.sig_obs.values)
    lf = p.log10F.values
    v = np.var(lf)
    share = {k: float(np.cov(a, lf)[0, 1] / v) for k, a in t.items() if k != "const"}
    if gal:
        share["residual_gal"] = 1 - sum(share.values())
    return {"share": share, "sd": float(np.sqrt(v)),
            "term_sd": {k: float(np.std(a)) for k, a in t.items() if k != "const"}}


def spearman_all(p):
    lf = p.log10F.values
    cols = {"log mu": np.log10(p.mu), "log d": np.log10(p.d), "log Pb": np.log10(p.pb_s),
            "log sig_obs": np.log10(p.sig_obs), "sigma_d/d": p.sigma_d_frac,
            "sigma_mu/mu": p.sigma_mu_frac, "log e": np.log10(p.ecc), "m2": p.m2,
            "log v_t": np.log10(p.v_trans), "abs b": p.abs_b}
    out = {}
    for k, v in cols.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v) & np.isfinite(lf)
        r, pv = spearmanr(v[ok], lf[ok])
        out[k] = {"rho": float(r), "p": float(pv), "n": int(ok.sum())}
    return out


def regimes(logF, b):
    return np.digitize(np.asarray(logF, float), np.sort(b))


def fractions(logF, b):
    return [float(x) for x in np.bincount(regimes(logF, b), minlength=len(b) + 1) / len(logF)]


def misclass(p, b):
    t, o = regimes(p.log10F, b), regimes(p.log10F_obs, b)
    k = len(b) + 1
    M = np.zeros((k, k), int)
    np.add.at(M, (t, o), 1)
    return {"matrix": M, "overall": float((t != o).mean()),
            "per_regime": [float((o[t == i] != i).mean()) if (t == i).any() else None
                           for i in range(k)]}


def misclass_by(p, b, key):
    """Misclassification rate and population share for each value of column `key`."""
    t, o = regimes(p.log10F, b), regimes(p.log10F_obs, b)
    out = {}
    for v in sorted(p[key].unique()):
        m = (p[key] == v).values
        out[str(v)] = {"rate": float((t[m] != o[m]).mean()), "share": float(m.mean())}
    return out


def lever_fractions(p, b):
    """Fraction of each class (M, B, D) for which each component dominates."""
    r = regimes(p.log10F, b)
    names = ["d", "mu", "Gal"]
    return {["M", "B", "D"][i]: {n: float((p.lever[r == i] == n).mean()) if (r == i).any() else None
                                 for n in names} for i in range(len(b) + 1)}
