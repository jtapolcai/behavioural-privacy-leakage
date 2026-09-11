#!/usr/bin/env python3
"""
Compute FIFO flow lag and Little's-law residence-time estimates
for the same-address (H1) subset of Railgun and Privacy Pools,
then compare to the full-population values already in
  Figures/figure_ch4_03_little_law_timeseries.csv
  Figures/figure_ch4_03_pp_retained_fifo.csv

Approach
--------
Rather than filtering shield/unshield events by address (which captures
unrelated events landing on H1 addresses), we build the cumulative curves
directly from the matched pairs:
  - Railgun: h1_pairs_with_ppoi_tags.csv  (sh_tx, delay_h, sh_amt, un_amt)
             joined with eth_shield.csv   to recover the shield timestamp.
  - PP:      pp_h1_pairs.csv              already contains dep_time / with_time.

This ensures that the H1 cumulative curves only count ETH that is genuinely
part of a same-address round trip.

Run from BehaviouralPrivacyLeakageFC/:
    python3 scripts/compute_h1_subset_fifo.py

Outputs
-------
Figures/figure_h1_subset_cumulative_railgun.csv
Figures/figure_h1_subset_cumulative_pp.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

RG_DATA = _RG_DATA  # via _paths
PP_DATA = Path("../privacypools-deanonymization-main/data/processed")
FIG     = Path("Figures")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fifo_lag_series(cum_shield: np.ndarray,
                    cum_unshield: np.ndarray,
                    dates: pd.DatetimeIndex) -> np.ndarray:
    """
    For each day t where cum_unshield(t) = V > 0, find the first day t' where
    cum_shield(t') >= V; FIFO lag = (t − t').days.
    Returns float array with NaN where undefined.
    """
    lags = np.full(len(dates), np.nan)
    for i, v in enumerate(cum_unshield):
        if v <= 0:
            continue
        idx = np.searchsorted(cum_shield, v, side='left')
        if idx < len(dates):
            lags[i] = (dates[i] - dates[idx]).days
    return lags


def little_law_scalar(retained: np.ndarray,
                      cum_unshield: np.ndarray,
                      dates: pd.DatetimeIndex) -> float:
    """
    T_LL = mean(retained_balance) / mean_daily_outflow
    where mean_daily_outflow = total_outflow / total_days.
    """
    total_days    = (dates[-1] - dates[0]).days
    total_outflow = float(cum_unshield[-1])
    if total_days <= 0 or total_outflow <= 0:
        return np.nan
    return float(np.mean(retained)) / (total_outflow / total_days)


def pairs_to_daily_curves(dep_times: pd.Series,
                          dep_amounts: pd.Series,
                          wdw_times: pd.Series,
                          wdw_amounts: pd.Series,
                          label: str):
    """
    Build aligned daily cumulative curves from matched (deposit, withdrawal) pairs.
    Returns (date_grid, cum_shield, cum_unshield, retained).
    """
    dep_daily = (dep_amounts
                 .groupby(dep_times.dt.normalize()).sum()
                 .cumsum())
    wdw_daily = (wdw_amounts
                 .groupby(wdw_times.dt.normalize()).sum()
                 .cumsum())

    date_idx = pd.date_range(
        start=min(dep_daily.index.min(), wdw_daily.index.min()),
        end  =max(dep_daily.index.max(), wdw_daily.index.max()),
        freq ='D',
    )
    cs = dep_daily.reindex(date_idx).ffill().fillna(0).values.astype(float)
    cu = wdw_daily.reindex(date_idx).ffill().fillna(0).values.astype(float)
    retained = np.clip(cs - cu, 0, None)

    n_pairs = len(dep_times)
    print(f"  {label}: {n_pairs:,} pairs, "
          f"total deposited {cs[-1]:.2f} ETH, "
          f"total withdrawn {cu[-1]:.2f} ETH")
    return date_idx, cs, cu, retained


def report_and_save(dates, cs, cu, retained, fifo, tll, outfile):
    fifo_valid = fifo[~np.isnan(fifo)]
    print(f"  Little's law T_LL   : {tll:.1f} days")
    if len(fifo_valid):
        print(f"  FIFO lag — median   : {np.nanmedian(fifo):.1f} days")
        print(f"  FIFO lag — last     : {fifo_valid[-1]:.1f} days")
    df = pd.DataFrame({
        "date"         : dates.strftime("%Y-%m-%d"),
        "cum_shield"   : cs,
        "cum_unshield" : cu,
        "retained"     : retained,
        "fifo_lag_days": fifo,
    })
    df.to_csv(outfile, index=False)
    print(f"  Wrote {outfile}")
    return fifo_valid


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Railgun H1 subset  (join pairs → eth_shield timestamps)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Railgun — same-address (H1) subset")
print("=" * 60)

pairs_rg = pd.read_csv(RG_DATA / "h1_pairs_with_ppoi_tags.csv")
pairs_rg  = pairs_rg[pairs_rg["tok"] == "WETH"].copy()
print(f"  WETH H1 pairs: {len(pairs_rg):,}")

# Recover shield timestamp from eth_shield.csv
shields = pd.read_csv(RG_DATA / "transactions/eth_shield.csv",
                      usecols=["tx_hash", "time"])
shields["tx_hash"] = shields["tx_hash"].str.lower()
pairs_rg["sh_tx"]  = pairs_rg["sh_tx"].str.lower()

pairs_rg = pairs_rg.merge(shields.rename(columns={"tx_hash": "sh_tx",
                                                    "time":   "sh_time"}),
                           on="sh_tx", how="inner")
print(f"  Pairs matched with shield timestamp: {len(pairs_rg):,}")

pairs_rg["sh_time"]  = pd.to_datetime(pairs_rg["sh_time"], utc=True)
pairs_rg["un_time"]  = pairs_rg["sh_time"] + pd.to_timedelta(
                            pairs_rg["delay_h"], unit="h")

dates_rg, cs_rg, cu_rg, ret_rg = pairs_to_daily_curves(
    pairs_rg["sh_time"], pairs_rg["sh_amt"],
    pairs_rg["un_time"], pairs_rg["un_amt"],
    "Railgun H1")

fifo_rg = fifo_lag_series(cs_rg, cu_rg, dates_rg)
tll_rg  = little_law_scalar(ret_rg, cu_rg, dates_rg)

# Also report mean observed delay directly
mean_delay_rg = pairs_rg["delay_h"].mean() / 24
print(f"  Mean observed pair delay: {mean_delay_rg:.1f} days")

fv_rg = report_and_save(dates_rg, cs_rg, cu_rg, ret_rg, fifo_rg, tll_rg,
                         FIG / "figure_h1_subset_cumulative_railgun.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Privacy Pools H1 subset  (pp_h1_pairs already has both timestamps)
# ─────────────────────────────────────────────────────────────────────────────

_pp_pairs_csv = FIG / "pp_h1_pairs.csv"
if not _pp_pairs_csv.exists():
    print("\n⚠  pp_h1_pairs.csv not found — skipping PP H1 subset (run without --skip-pp)")
    import sys; sys.exit(0)

print()
print("=" * 60)
print("Privacy Pools — same-address (H1) subset")
print("=" * 60)

pairs_pp = pd.read_csv(_pp_pairs_csv)
print(f"  PP H1 pairs: {len(pairs_pp):,}")

pairs_pp["dep_time"]  = pd.to_datetime(pairs_pp["dep_time"],  utc=True)
pairs_pp["with_time"] = pd.to_datetime(pairs_pp["with_time"], utc=True)

dates_pp, cs_pp, cu_pp, ret_pp = pairs_to_daily_curves(
    pairs_pp["dep_time"],  pairs_pp["dep_amount"],
    pairs_pp["with_time"], pairs_pp["with_amount"],
    "PP H1")

fifo_pp = fifo_lag_series(cs_pp, cu_pp, dates_pp)
tll_pp  = little_law_scalar(ret_pp, cu_pp, dates_pp)

mean_delay_pp = pairs_pp["dt_days"].mean()
print(f"  Mean observed pair delay: {mean_delay_pp:.1f} days")

fv_pp = report_and_save(dates_pp, cs_pp, cu_pp, ret_pp, fifo_pp, tll_pp,
                         FIG / "figure_h1_subset_cumulative_pp.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Full-population comparison
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("Comparison: full population vs. H1 subset")
print("=" * 60)

# Railgun full population
full_rg = pd.read_csv(FIG / "figure_ch4_03_little_law_timeseries.csv")
full_rg["date"] = pd.to_datetime(full_rg["date"])
tll_full_rg  = full_rg["little_days_global"].dropna().iloc[-60:].mean()
fifo_full_rg = full_rg["fifo_horizontal_lag_days"].dropna().iloc[-1]

print(f"\n  Railgun")
print(f"    Full pop. T_LL  (60-day avg)   : {tll_full_rg:.1f} days")
print(f"    Full pop. FIFO lag (last date) : {fifo_full_rg:.1f} days")
print(f"    H1 subset T_LL                 : {tll_rg:.1f} days")
if len(fv_rg):
    print(f"    H1 subset FIFO lag (last)      : {fv_rg[-1]:.1f} days")
print(f"    H1 subset mean pair delay       : {mean_delay_rg:.1f} days")
if tll_rg > 0:
    print(f"    Ratio full/H1  (T_LL)          : {tll_full_rg/tll_rg:.1f}×")

# PP full population
full_pp_df = pd.read_csv(FIG / "figure_ch4_03_pp_retained_fifo.csv")
fifo_full_pp = full_pp_df["fifo_horizontal_lag_days"].dropna().iloc[-1]

print(f"\n  Privacy Pools")
print(f"    Full pop. FIFO lag (last date) : {fifo_full_pp:.1f} days")
print(f"    H1 subset T_LL                 : {tll_pp:.1f} days")
if len(fv_pp):
    print(f"    H1 subset FIFO lag (last)      : {fv_pp[-1]:.1f} days")
print(f"    H1 subset mean pair delay       : {mean_delay_pp:.1f} days")

print("\nDone.")
