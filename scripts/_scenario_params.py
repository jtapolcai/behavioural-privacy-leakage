"""
_scenario_params.py
===================
Single source of truth for the declared scenario parameters that appear in
Table 2 of the paper (tab:calibration).

These are NOT estimated from data — they are chosen to reflect hodler
under-representation in young pools (Assumption: hodler under-representation).
The FIFO/Little's-law comparison in the paper is post-hoc validation, not a
calibration criterion.

To update the entropy results:
  1. Edit the constants below (RG_PI / RG_MU or PP_PI / PP_MU).
  2. Update the LaTeX table manually:
       6aa410a17e14d2dde604af68/behavioural_privacy_leakage_fc27.tex
       near \\label{tab:calibration}
  3. Re-run the pipeline (or just the affected scripts):
       python scripts/compute_m2_canonical.py   --rg-data ...
       python scripts/compute_m3_canonical.py   --rg-data ...
       python scripts/compute_pp_m2_addr.py     --source  ...
       python scripts/compute_pp_m3.py          --source  ...
       python scripts/compute_pp_m4.py          --source  ...
       python scripts/compute_table2.py         --output-dir Figures
  4. Copy Figures/macros_generated.tex to the paper repo.
"""

import numpy as np

# ── Railgun scenario parameters ───────────────────────────────────────────────
# Components: sprinter / regular / hodler
# Paper notation: π^{est}, μ^{est}

RG_PI  = np.array([0.085, 0.461, 0.454])   # mixing weights (sum = 1)
RG_MU  = np.array([0.09,  8.79,  403.0])   # mean holding times (days)

# ── Privacy Pools scenario parameters ─────────────────────────────────────────
PP_PI  = np.array([0.065, 0.349, 0.586])   # mixing weights (sum = 1)
PP_MU  = np.array([0.09,  8.79,  246.0])   # mean holding times (days)

# Shared observed means from joint CDF fit (μ^{obs})
MU_OBS = np.array([0.09,  8.79,   45.0])   # sprinter / regular / hodler (days)
