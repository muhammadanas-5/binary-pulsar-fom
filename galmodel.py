"""Line-of-sight Galactic acceleration in the McMillan (2017) potential (via galpy).

Used only for the model-uncertainty term sigma_mod of the figure of merit (paper Sect. 2.2-2.3):
    sigma_mod = | Pbdot_Gal(reference, Eqs. 5-6) - Pbdot_Gal(McMillan17) |   at the nominal distance.

Geometry: Galactocentric Cartesian frame with the Galactic centre at the origin and the Sun in
the plane at x = -R0 (so l = 0 points towards the centre). A pulsar at (l, b, d) sits at
    x = -R0 + d cos b cos l,  y = d cos b sin l,  z = d sin b,
and the line-of-sight unit vector is n = (cos b cos l, cos b sin l, sin b).
The orbital-period derivative induced by the differential acceleration is
    Pbdot_Gal = Pb * n . (a_psr - a_sun) / c .
The sign convention matches Eqs. (5)-(6) (checked in tests/test_reproduce.py).

The potential is evaluated with the Sun at the paper's R0 (physics.R0), not galpy's own
R0 = 8.21 kpc; the potential's own length/velocity scale is used only for unit conversion.
"""
import numpy as np
from functools import lru_cache

from physics import C, KPC, R0

_POT = None
_RO = _VO = None


def _potential():
    global _POT, _RO, _VO
    if _POT is None:
        from galpy.potential.mwpotentials import McMillan17
        from galpy.util import conversion
        ph = conversion.get_physical(McMillan17)
        _POT, _RO, _VO = McMillan17, ph["ro"], ph["vo"]
    return _POT, _RO, _VO


def accel(x, y, z):
    """Galactocentric acceleration vector (m/s^2) at Cartesian position (kpc)."""
    from galpy.potential import evaluateRforces, evaluatezforces
    pot, ro, vo = _potential()
    x, y, z = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, z))
    R = np.hypot(x, y)
    fR = evaluateRforces(pot, R / ro, z / ro, use_physical=False)
    fz = evaluatezforces(pot, R / ro, z / ro, use_physical=False)
    scale = (vo * 1e3) ** 2 / (ro * KPC)                   # natural force -> m/s^2
    Rsafe = np.where(R > 0, R, 1.0)
    return fR * scale * x / Rsafe, fR * scale * y / Rsafe, fz * scale


def pbdot_gal_mcmillan(pb, l, b, d):
    """Galactic contribution to Pbdot (s/s) from the full McMillan17 potential."""
    pb, d = np.asarray(pb, float), np.asarray(d, float)
    lr, br = np.radians(l), np.radians(b)
    nx, ny, nz = np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)
    ax, ay, az = accel(-R0 + d * nx, d * ny, d * nz)
    sx, sy, sz = accel(-R0, 0.0, 0.0)
    los = nx * (ax - sx) + ny * (ay - sy) + nz * (az - sz)
    return pb * los / C


class SightlineTable:
    """McMillan17 term tabulated on a logarithmic distance grid along one sightline and
    interpolated (the full-potential evaluation is slow)."""

    def __init__(self, l, b, dmin=0.01, dmax=60.0, n=80):
        self.l, self.b = float(l), float(b)
        self.grid = np.geomspace(dmin, dmax, n)
        # per unit period, so the table is independent of Pb
        self.vals = pbdot_gal_mcmillan(1.0, self.l, self.b, self.grid)

    def __call__(self, pb, d):
        d = np.clip(np.asarray(d, float), self.grid[0], self.grid[-1])
        return np.asarray(pb, float) * np.interp(np.log(d), np.log(self.grid), self.vals)


@lru_cache(maxsize=None)
def table(l, b):
    return SightlineTable(round(float(l), 3), round(float(b), 3))
