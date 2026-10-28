"""Regression tests. Run from the repository root:  python -m pytest -q tests"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import physics, galmodel, lit_inputs, budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shklovskii_double_pulsar():
    # Paper Sect. 3.3: +1.73e-16 for J0737-3039A/B
    v = physics.pbdot_shk(0.1023 * 86400, 3.31, 0.735)
    assert abs(v / 1.73e-16 - 1) < 0.01


def test_galactic_sign_and_model_agreement():
    # Reference model (Eqs. 5-6) and McMillan17 share sign and agree to ~10% near the Sun
    for l, b, d in [(0, 0, 0.5), (90, 0, 1.0), (180, 0, 1.0), (250, -5, 1.0), (100, 60, 0.9)]:
        ref = float(physics.pbdot_gal(86400.0, l, b, d))
        mcm = float(galmodel.pbdot_gal_mcmillan(86400.0, l, b, d)[0])
        assert np.sign(ref) == np.sign(mcm) and abs(mcm / ref - 1) < 0.10


def test_table1_reproduced():
    lit = budget.evaluate(lit_inputs.literature_sample(), seed=0)
    pap = pd.read_csv(os.path.join(ROOT, "paper_table1.csv"))
    m = lit.merge(pap, on="label")
    assert len(m) == 16
    assert (m.cls == m.class_paper).all()
    assert (m.lever == m.lever_paper).all()
    assert (np.abs(m.F / m.F_paper - 1) < 0.05).all()


def test_catalogue_sample_matches_paper():
    from run_analysis import build_catalogue_sample
    kept, excluded, _, info = build_catalogue_sample(
        parent_csv=os.path.join(ROOT, "data", "atnf_v2.8.1_parent_sample.csv"))
    assert info["n_final"] == 28 and info["n_excluded"] == 32
    assert info["excluded_by_reason"] and sorted(info["excluded_by_reason"].values()) == [1, 11, 20]
    cat = budget.evaluate(kept, n_mc=20000, n_true=2000, seed=0)
    counts = cat.cls.value_counts().to_dict()
    assert counts == {"D": 11, "M": 10, "B": 7}
    assert abs(cat.F.median() - 0.50) < 0.02
