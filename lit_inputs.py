"""Inputs for the sixteen literature systems of Table 1.

All numbers are transcribed from Table 1 of the paper (which in turn cites the timing papers
listed in its note). Pbdot and its uncertainty are in 1e-12 s/s in the table and converted here.

Galactic coordinates are derived from the J2000 positions encoded in the pulsar names
(hhmm, ddmm), which agree with catalogue coordinates to better than 0.1 deg. B1913+16 is
J1915+1606 and B1534+12 is J1537+1155.

sigma_mu/mu: for J1909-3744 it is the ATNF v2.8.1 value; for the others it is the value
implied by the proper-motion term of the simplified F (Eq. 9), as stated in the Table 1 note.
J2145-0750 has a negligible proper-motion term, so 0 is used.
"""
import re
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from physics import DAY

# label, J-name, comp, Pb [d], mu [mas/yr], sigma_mu/mu, d [kpc], sigma_d [kpc],
# Pbdot_obs [1e-12], sigma_Pbdot [1e-12], dist_type, reference number (Table 1 note)
_ROWS = [
    ("J0348+0432",    "J0348+0432", "WD",  0.1024,  5.35,   0.0215, 2.10,  0.20,  -0.273,     0.045,     "photometric", 9),
    ("J2234+0611",    "J2234+0611", "WD",  32.001,  27.1,   0.0009, 0.95,  0.04,  +3.1,       2.5,       "parallax",    15),
    ("J1913+1102",    "J1913+1102", "NS",  0.2063,  9.3,    0.0971, 7.14,  1.5,   -0.48,      0.03,      "DM",          7),
    ("J2222-0137",    "J2222-0137", "WD",  2.4458,  45.06,  0.0011, 0.267, 0.001, +0.251,     0.008,     "VLBI",        13),
    ("J1946+2052",    "J1946+2052", "NS",  0.0785,  4.5,    0.0446, 3.5,   1.0,   -1.8296,    0.0015,    "DM",          4),
    ("J1756-2251",    "J1756-2251", "NS",  0.3196,  2.42,   0.0525, 0.73,  0.40,  -0.229,     0.005,     "parallax",    5),
    ("J0737-3039A/B", "J0737-3039A", "NS", 0.1023,  3.31,   0.0151, 0.735, 0.060, -1.247920,  0.000078,  "parallax",    2),
    ("J1829+2456",    "J1829+2456", "NS",  1.176,   9.52,   0.0065, 1.2,   0.3,   -0.029,     0.012,     "DM",          16),
    ("J1738+0333",    "J1738+0333", "WD",  0.3548,  8.67,   0.0023, 1.47,  0.10,  -0.0170,    0.0031,    "parallax",    8),
    ("J1012+5307",    "J1012+5307", "WD",  0.6047,  25.6,   0.0040, 0.83,  0.05,  +0.061,     0.005,     "parallax",    11),
    ("J1933-6211",    "J1933-6211", "WD",  12.819,  12.42,  0.0083, 1.6,   0.3,   +0.7,       0.1,       "parallax",    17),
    ("J2145-0750",    "J2145-0750", "WD",  6.8389,  13.15,  0.0,    0.50,  0.15,  +0.13,      0.02,      "parallax",    12),
    ("J1757-1854",    "J1757-1854", "NS",  0.1835,  4.4,    0.0684, 11.5,  3.0,   -5.294,     0.006,     "unknown",     6),
    ("B1913+16",      "J1915+1606", "NS",  0.323,   1.48,   0.0276, 9.0,   2.0,   -2.423,     0.001,     "DM",          1),
    ("J1909-3744",    "J1909-3744", "WD",  1.5334,  37.02,  0.0048, 1.158, 0.003, +0.5093,    0.0007,    "Pbdot",       10),
    ("B1534+12",      "J1537+1155", "NS",  0.4207,  25.29,  0.0022, 0.94,  0.07,  -0.1366,    0.0003,    "parallax",    3),
]
_COLS = ["label", "jname", "comp", "pb_d", "mu", "sigma_mu_frac", "d", "sigma_d",
         "pbdot_obs_e12", "sig_obs_e12", "dist_type", "ref"]


def _lb_from_name(jname):
    m = re.match(r"J(\d{2})(\d{2})([+-])(\d{2})(\d{2})", jname)
    ra = (int(m.group(1)) + int(m.group(2)) / 60.0) * 15.0
    dec = (1 if m.group(3) == "+" else -1) * (int(m.group(4)) + int(m.group(5)) / 60.0)
    g = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").galactic
    return g.l.deg, g.b.deg


def literature_sample():
    t = pd.DataFrame(_ROWS, columns=_COLS)
    lb = np.array([_lb_from_name(j) for j in t.jname])
    t["gl"], t["gb"] = lb[:, 0], lb[:, 1]
    t["pb_s"] = t.pb_d * DAY
    t["sigma_d_frac"] = t.sigma_d / t.d
    t["pbdot_obs"] = t.pbdot_obs_e12 * 1e-12
    t["sig_obs"] = t.sig_obs_e12 * 1e-12
    # parallax / VLBI distances are drawn by inverting a Gaussian parallax (Sect. 2.3)
    t["dist_method"] = np.where(t.dist_type.isin(["parallax", "VLBI"]), "parallax", "DM/other")
    return t
