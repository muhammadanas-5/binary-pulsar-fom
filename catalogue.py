"""ATNF query, parent sample and exclusions (paper Sect. 3.5).

Parent sample: binaries with a measured Pbdot (with uncertainty), proper motion (with
uncertainty), distance, eccentricity and mass function -- everything the figure of merit needs.
Exclusions: systems for which the kinematic decomposition does not apply (globular-cluster
pulsars, ultra-light / non-degenerate companions, the B pulsar of the Double Pulsar, and
J2043+1711). Every excluded system and its reason is returned so it can be written to a file."""
import os, json
import numpy as np, pandas as pd

from physics import DAY, T_SUN, mass_function

ATNF_P = ["JNAME", "GL", "GB", "PB", "PBDOT", "A1", "ECC", "EPS1", "EPS2", "PMRA",
          "PMDEC", "PX", "DIST", "DIST_DM", "BINCOMP", "ASSOC"]
PBDOT_UNIT_SWITCH = 1e-7   # |value| above this is taken to be in units of 1e-12 (tempo convention)


def query_atnf(cache, refresh=False):
    mp = os.path.splitext(cache)[0] + "_meta.json"
    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache), (json.load(open(mp)) if os.path.exists(mp) else {})
    from psrqpy import QueryATNF
    cond = "type(BINARY) && exist(PB) && exist(PBDOT) && exist(PMRA) && exist(PMDEC)"
    q = QueryATNF(params=ATNF_P, condition=cond)   # errors (_ERR columns) are included
    df = q.table.to_pandas()
    meta = {"catalogue_version": str(q.get_version), "condition": cond,
            "n_rows": int(len(df))}
    df.to_csv(cache, index=False)
    json.dump(meta, open(mp, "w"), indent=2)
    return df, meta


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def pbdot_si(val, err):
    """Return Pbdot and its error in s/s, handling values stored in units of 1e-12."""
    scale = np.where(np.abs(val) > PBDOT_UNIT_SWITCH, 1e-12, 1.0)
    return val * scale, err * scale, int((scale != 1.0).sum())


def parent_sample(df, dm_err=0.25, px_snr=2.0):
    n = lambda c: _num(df, c)
    e = n("ECC")
    ell1 = np.sqrt(n("EPS1").fillna(0) ** 2 + n("EPS2").fillna(0) ** 2)
    px, pxe = n("PX"), n("PX_ERR")
    good = px.notna() & pxe.notna() & (px > 0) & (pxe > 0) & (px / pxe > px_snr)
    pmra, pmdec = n("PMRA"), n("PMDEC")
    mu = np.hypot(pmra, pmdec)
    smu = np.sqrt((pmra * n("PMRA_ERR")) ** 2 + (pmdec * n("PMDEC_ERR")) ** 2) / mu
    pbd, pbde, n_rescaled = pbdot_si(n("PBDOT").values, n("PBDOT_ERR").values)
    p = pd.DataFrame({
        "jname": df["JNAME"] if "JNAME" in df else None,
        "pb_s": n("PB") * DAY, "mu": mu, "d": n("DIST"),
        "ecc": e.where(e.notna() & (e > 0), ell1.where(ell1 > 0)).clip(lower=1e-7),
        "fmass": mass_function((n("PB") * DAY).values, n("A1").values),
        "pbdot_obs": pbd, "sig_obs": pbde,
        "gl": n("GL"), "gb": n("GB"), "bincomp": df["BINCOMP"] if "BINCOMP" in df else None,
        "assoc": df["ASSOC"].astype(str) if "ASSOC" in df else "",
        "sigma_d_frac": np.where(good, pxe / px, dm_err),
        "sigma_mu_frac": smu,
        "dist_method": np.where(good, "parallax", "DM/other")})
    need = ["pb_s", "mu", "d", "ecc", "fmass", "sig_obs", "sigma_mu_frac"]
    ok = (p.ecc < 1) & np.isfinite(p[need]).all(axis=1) & (p[need] > 0).all(axis=1)
    info = {"n_input": int(len(p)), "n_kept": int(ok.sum()),
            "n_pbdot_rescaled_from_1e-12": n_rescaled,
            "n_dropped_missing_or_nonpositive": int((~ok).sum())}
    return p[ok].reset_index(drop=True), info


# Globular-cluster pulsars by name prefix: used when the cached query has no ASSOC column.
# 47 Tuc, NGC 1851, M5, M62, Terzan 5, NGC 6652, NGC 6752, M15
GC_PREFIXES = ("J0024-7204", "J0514-4002", "J1518+0204", "J1701-3006", "J1748-2446",
               "J1835-3259", "J1910-5959", "J2129+1210")


def _is_gc(assoc, jname):
    by_assoc = assoc.fillna("").astype(str).str.upper().str.contains(r"(?:^|[ ,;])GC:", regex=True)
    by_name = jname.astype(str).str.startswith(GC_PREFIXES)
    return by_assoc | by_name


def apply_exclusions(p, artefact_sig=1e-21):
    """Split the parent sample into (kept, excluded). `excluded` has a `reason` column.
    Systems whose catalogue sigma(Pbdot) is below `artefact_sig` s/s are flagged (not removed
    separately): no timing measurement reaches that precision, so they are catalogue artefacts."""
    cls = comp_class(p["bincomp"].values)
    reason = np.full(len(p), "", dtype=object)
    reason[_is_gc(p["assoc"] if "assoc" in p else pd.Series("", index=p.index), p["jname"]).values] = "globular-cluster pulsar (Pbdot dominated by cluster potential)"
    reason[np.isin(cls, ["UL", "MS"]) & (reason == "")] = \
        "ultra-light or non-degenerate companion (tidal / mass-loss period changes)"
    reason[(p["jname"] == "J0737-3039B").values & (reason == "")] = \
        "B pulsar of the Double Pulsar (shares its orbit with J0737-3039A)"
    reason[(p["jname"] == "J2043+1711").values & (reason == "")] = \
        "peculiar acceleration inconsistent with Galactic models (Fonseca et al. 2016)"
    ex = p.assign(reason=reason, comp_class=cls)
    ex["catalogue_artefact_sigma"] = ex.sig_obs < artefact_sig
    bad = reason != ""
    return p[~bad].reset_index(drop=True), ex[bad].reset_index(drop=True)


def comp_class(bincomp):
    out = []
    for b in bincomp:
        u = ("" if b is None or (isinstance(b, float) and np.isnan(b)) else str(b)).strip().upper()
        out.append(next((k for k, p in [("NS", "NS"), ("He", "HE"), ("CO", "CO"),
                                        ("CO", "ONE"), ("MS", "MS"), ("UL", "UL")]
                         if u.startswith(p)), "UNK"))
    return np.array(out)


def mock(n=280, seed=7):
    """Fake catalogue for testing the pipeline ONLY. Never use its output in a paper."""
    rng = np.random.default_rng(seed)
    nc = int(0.55 * n)
    pb = 10 ** np.concatenate([rng.normal(-0.4, 0.45, nc), rng.normal(1.3, 0.55, n - nc)])
    d = 10 ** rng.normal(0.35, 0.33, n)
    mu = (np.abs(rng.normal(0, 90, n)) + 15) / (4.74047 * d) * 10 ** rng.normal(0, 0.12, n)
    ns = rng.random(n) < 0.10
    ecc = np.clip(np.where(ns, 10 ** rng.normal(-0.9, 0.45, n),
                           10 ** rng.normal(-5.0, 1.1, n)), 1e-7, 0.9)
    m2 = np.where(ns, rng.normal(1.33, 0.09, n),
                  0.10 + 0.13 * pb**0.30 * 10 ** rng.normal(0, 0.15, n))
    sini = np.sqrt(1 - rng.random(n) ** 2)
    fm = (m2 * sini) ** 3 / (1.4 + m2) ** 2
    a1 = (fm * T_SUN * (pb * DAY) ** 2 / (4 * np.pi**2)) ** (1 / 3)
    ang = rng.uniform(0, 2 * np.pi, n)
    hp = rng.random(n) < 0.35
    from physics import pbdot_shk, pbdot_gr
    sig = 10 ** rng.normal(-14.0, 0.8, n)
    val = pbdot_shk(pb * DAY, mu, d) + pbdot_gr(pb * DAY, ecc, 1.4, m2) + rng.normal(0, sig)
    return pd.DataFrame({"JNAME": [f"MOCK{i:04d}" for i in range(n)], "PB": pb,
                         "PBDOT": val, "PBDOT_ERR": sig,
                         "A1": a1, "ECC": ecc, "EPS1": np.nan, "EPS2": np.nan,
                         "PMRA": mu * np.cos(ang), "PMDEC": mu * np.sin(ang),
                         "PMRA_ERR": mu * 10 ** rng.normal(-1.7, 0.5, n),
                         "PMDEC_ERR": mu * 10 ** rng.normal(-1.7, 0.5, n), "DIST": d,
                         "PX": np.where(hp, 1 / d, np.nan),
                         "PX_ERR": np.where(hp, rng.uniform(0.05, 0.30, n) / d, np.nan),
                         "GL": rng.uniform(0, 360, n),
                         "GB": np.degrees(np.arcsin(rng.uniform(-1, 1, n))) * 0.5,
                         "BINCOMP": np.where(ns, "NS", np.where(m2 < 0.45, "He", "CO"))})
