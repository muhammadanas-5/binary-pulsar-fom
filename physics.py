"""Kinematic and radiative orbital-decay terms and the figure of merit F (paper Eqs. 1-6 and 9).

All P-dot quantities are dimensionless (s/s). Distances in kpc, proper motions in mas/yr,
orbital periods in seconds.
"""
import numpy as np
from scipy.optimize import brentq

C = 2.99792458e8
T_SUN = 4.925490947e-6
KPC = 3.0856775814913673e19
DAY = 86400.0
MAS_YR = (np.pi / 6.48e8) / (365.25 * 86400.0)
# Solar Galactocentric parameters (paper Sect. 2.2; GRAVITY Collaboration 2021).
R0, TH0 = 8.275, 240.5e3   # kpc, m/s


def f_ecc(e):  # Peters & Mathews 1963
    e = np.asarray(e, float)
    return (1 + 73 / 24 * e**2 + 37 / 96 * e**4) * (1 - e**2) ** -3.5


def pbdot_gr(pb, e, m1, m2):  # paper Eq. (1)
    pb, m1, m2 = map(lambda x: np.asarray(x, float), (pb, m1, m2))
    return (-(192 * np.pi / 5) * (2 * np.pi * T_SUN / pb) ** (5 / 3)
            * m1 * m2 * (m1 + m2) ** (-1 / 3) * f_ecc(e))


def pbdot_shk(pb, mu, d):  # paper Eq. (4); Shklovskii 1970
    return np.asarray(pb, float) * (np.asarray(mu, float) * MAS_YR) ** 2 * \
        np.asarray(d, float) * KPC / C


def pbdot_gal(pb, l, b, d):
    """Galactic term, paper Eqs. (5)-(6) (Damour & Taylor 1991; Nice & Taylor 1995;
    Lazaridis et al. 2009), with the vertical acceleration K_z of Holmberg & Flynn (2004)
    in the form used by Lazaridis et al. (2009): K_z/(1e-9 cm s^-2) = 2.27 z + 3.68(1-e^{-4.31 z}).

    Sign convention: Pbdot_Gal/Pb = n . (a_psr - a_sun)/c with n the line-of-sight unit vector,
    i.e. positive for a pulsar between the Sun and the Galactic centre (l = 0, b = 0).
    """
    l, b = np.radians(np.asarray(l, float)), np.radians(np.asarray(b, float))
    d = np.asarray(d, float)
    beta = d / R0 * np.cos(b) - np.cos(l)
    rot = -(TH0**2 / (C * R0 * KPC)) * (np.cos(l) + beta / (np.sin(l)**2 + beta**2)) * np.cos(b)
    z = np.abs(d * np.sin(b))
    kz = (2.27 * z + 3.68 * (1 - np.exp(-4.31 * z))) * 1e-11          # m s^-2
    vert = -kz * np.abs(np.sin(b)) / C
    return np.asarray(pb, float) * (rot + vert)


def sigma_kin(pb, mu, d, sd_frac, smu_frac, l=None, b=None, gal=False):
    """Uncertainty of the kinematic correction.

    gal=False is the simplified form, paper Eq. (9):
        sigma = |Pbdot_Shk| sqrt((sd/d)^2 + (2 smu/mu)^2).
    gal=True propagates the distance error through Shklovskii + Galactic terms together
    (they are fully correlated through d): sigma^2 = (d(Shk+Gal)/dd * sd)^2 + (2 Shk smu/mu)^2.
    """
    pb, mu, d = (np.asarray(x, float) for x in (pb, mu, d))
    shk = pbdot_shk(pb, mu, d)
    dkin_dd = shk / d
    if gal:
        h = 1e-4 * d
        dkin_dd = dkin_dd + (pbdot_gal(pb, l, b, d + h) - pbdot_gal(pb, l, b, d - h)) / (2 * h)
    sd = np.asarray(sd_frac, float) * d
    return np.sqrt((dkin_dd * sd) ** 2 + (2 * shk * np.asarray(smu_frac, float)) ** 2)


def figure_of_merit(pb, mu, d, sd_frac, smu_frac, sig_obs, l=None, b=None, gal=False):
    """Paper Eq. (7): F = sigma(Pbdot_kin) / sigma(Pbdot_obs), first order and WITHOUT the
    Galactic-model term sigma_mod. The full figure of merit (Eq. 8 with sigma_mod, Monte Carlo)
    is in budget.py; gal=False gives F_Shk of Eq. (9)."""
    return sigma_kin(pb, mu, d, sd_frac, smu_frac, l, b, gal) / np.asarray(sig_obs, float)


def logF_terms(pb, mu, d, sd_frac, smu_frac, sig_obs):
    """Additive decomposition of log10 F for the Shklovskii-only F (gal=False). Exact."""
    sd_frac, smu_frac = np.asarray(sd_frac, float), np.asarray(smu_frac, float)
    return {"T_Pb": np.log10(np.asarray(pb, float)),
            "T_mu": 2 * np.log10(np.asarray(mu, float) * MAS_YR),
            "T_d": np.log10(np.asarray(d, float) * KPC),
            "T_err": 0.5 * np.log10(sd_frac**2 + 4 * smu_frac**2),
            "T_sig": -np.log10(np.asarray(sig_obs, float)),
            "const": np.full(np.shape(pb), -np.log10(C))}


def mass_function(pb, x):
    return 4 * np.pi**2 * np.asarray(x, float) ** 3 / (T_SUN * np.asarray(pb, float) ** 2)


def m2_solve(f, m1, sini, m2max=100.0):
    f, m1, sini = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, float))
                                        for v in (f, m1, sini)])
    out = np.full(f.shape, np.nan)
    for i in range(f.size):
        f_, m_, s_ = f.flat[i], m1.flat[i], sini.flat[i]
        if not (np.isfinite(f_) and np.isfinite(m_) and s_ > 0 and f_ > 0):
            continue
        g = lambda m2: (m2 * s_) ** 3 - f_ * (m_ + m2) ** 2
        if g(1e-6) * g(m2max) > 0:
            continue
        out.flat[i] = brentq(g, 1e-6, m2max, xtol=1e-10, rtol=1e-10)
    return out
