#!/usr/bin/env python3
"""Build the four empirical CDF series displayed in Figure 4.

The observed Railgun CDF is read from the event-level export. The observed
Privacy Pools CDF is rebuilt from the H1-linked pairs. No density estimates or
random-pairing baselines are plotted.
"""

from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

import numpy as np
import pandas as pd


BASE = _FIGURES.parent
FIGURES = _FIGURES
MAX_PLOT_POINTS = 1000


def thin_cdf(data: pd.DataFrame, max_points: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    """Keep evenly spaced empirical quantiles to stay within TeX memory limits."""
    if len(data) <= max_points:
        return data
    indices = np.unique(np.linspace(0, len(data) - 1, max_points).astype(int))
    return data.iloc[indices].reset_index(drop=True)


def save_observed_railgun() -> None:
    data = pd.read_csv(FIGURES / "cdf_deltas_data_full.csv")
    data = data.loc[data["days"] > 0, ["days", "cdf"]]
    data = thin_cdf(data)
    data.to_csv(FIGURES / "fig4_cdf_railgun_observed.csv", index=False)


def save_observed_privacy_pools() -> None:
    pairs = pd.read_csv(FIGURES / "pp_h1_pairs.csv")
    days = np.sort(pairs.loc[pairs["dt_days"] > 0, "dt_days"].to_numpy())
    cdf = np.arange(1, len(days) + 1) / len(days)
    pd.DataFrame({"days": days, "cdf": cdf}).to_csv(
        FIGURES / "fig4_cdf_privacypools_observed.csv", index=False
    )


save_observed_railgun()
save_observed_privacy_pools()
print("Wrote the two observed Figure 4 CDF series to Figures/.")
