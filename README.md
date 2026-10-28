# Figure of merit for kinematic limitation of binary-pulsar tests of orbital decay

Code accompanying: M. Anas, *A figure of merit for kinematic limitation of binary-pulsar tests of
orbital decay*. It computes

    F = sigma(Pbdot_kin) / sigma(Pbdot_obs),   Pbdot_kin = Pbdot_Shk + Pbdot_Gal

for sixteen literature binary pulsars (Table 1), for field binaries in the ATNF Pulsar Catalogue
(Table 2) and for a smoothed-bootstrap synthetic population (Sect. 3.6).

## Install and test

    pip install -r requirements.txt
    python -m pytest -q tests          # Table 1 (classes, levers, F within 5%) and the 28-system catalogue sample

## Reproduce the results

    python run_upgrade.py --lit-only   # Table 1 and Fig. 1 (no network needed)
    python run_upgrade.py              # + Table 2, synthetic population, Figs. 2-4

The catalogue stage needs the ATNF Pulsar Catalogue (paper: v2.8.1). The 60-system parent sample
produced from that query is included in `data/atnf_v2.8.1_parent_sample.csv`, so the paper's Table 2
and Sect. 3.6 can be reproduced without network access:

    python run_upgrade.py --parent-csv data/atnf_v2.8.1_parent_sample.csv --outdir results
    python make_tables.py --results results --out tables    # LaTeX tables of the paper

Without `--parent-csv` the pipeline queries ATNF through `psrqpy` (cached in `atnf_binaries.csv`);
a newer catalogue release may change the sample slightly. `python run_analysis.py` runs the catalogue stage alone (parent sample, exclusions,
cross-check with Table 1). Random seeds: 0 (per-system Monte Carlo), 42 (synthetic population).
`--mock` uses a fake catalogue to test the pipeline only; never use its output.

## Modules

| file | role |
|---|---|
| `physics.py` | Eqs. 1-6, 9: GR decay, Shklovskii and reference Galactic terms, simplified F |
| `galmodel.py` | line-of-sight Galactic acceleration in McMillan (2017) via galpy |
| `budget.py` | Eq. 8: components, Monte Carlo F, lever, class probabilities |
| `lit_inputs.py` | Table 1 inputs |
| `catalogue.py` | ATNF query, parent sample, exclusions (Sect. 3.5) |
| `population.py` | synthetic population (Sect. 3.6) |
| `analysis.py` | mixture models, KDE, misclassification, correlations |
| `run_analysis.py` | catalogue stage and Table 1 cross-check |
| `run_upgrade.py` | full analysis; writes result tables, `fig1.pdf`-`fig4.pdf`, `report.json` |
| `make_tables.py` | typesets the paper's LaTeX tables from the result files |
