#!/usr/bin/env python3
"""generate_rg_raw.py
=============================================================
Stage-0 script: generates pre-processed Railgun CSV files that
all downstream scripts depend on.  Must run before any other
compute_* or fit_* scripts.

Reads from:
  <RG_DATA>/data/transactions/eth_shield.csv
  <RG_DATA>/data/transactions/eth_unshields.csv
  <RG_DATA>/data/h1_pairs_with_ppoi_tags.csv
  <RG_DATA>/data/aggregated/eth_shield_aggregated.csv
  <RG_DATA>/data/aggregated/eth_unshields_aggregated.csv

Writes to <FIGURES>/:
  cdf_deltas_data_full.csv           — empirical CDF of H1 holding times
  cdf_deltas_data_sampled_y_001.csv  — thinned CDF (Δy ≥ 0.001)
  figure_ch4_02_weekly_boundary_counts.csv
  figure_ch4_03_cumulative_pool_flow.csv
  figure_ch4_03_little_law_timeseries.csv
  figure_ch4_06_h1_coverage.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path config ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, FIGURES as _FIGURES

PROTOCOL_ADDRS = frozenset({
    "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9",
    "0xac9f360ae85469b27aeddeafc579ef2d052ad405",
    "0x4025ee6512dbbda97049bcf5aa5d38c54af6be8a",
    "0xe8a8b458bcd1ececc6b6b58f80929b29ccecff40",
    "0x22af4edbea3de885dda8f0a0653e6209e44e5b84",
    "0xc3f2c8f9d5f0705de706b1302b7a039e1e11ac88",
    "0x0000000000000000000000000000000000000000",
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rg-data",   type=Path, default=_RG_DATA,
                   help="railgun data/ directory")
    p.add_argument("--output-dir", type=Path, default=_FIGURES,
                   help="output directory")
    return p.parse_args()


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def generate_cdf_deltas(h1_csv: Path, out_dir: Path) -> None:
    """Empirical CDF of H1 holding times in days."""
    print("  Loading H1 pairs …")
    h1 = pd.read_csv(h1_csv)

    # delay column may be delay_h (hours) or delay_days
    if "delay_h" in h1.columns:
        days = h1["delay_h"].dropna() / 24.0
    elif "delay_days" in h1.columns:
        days = h1["delay_days"].dropna()
    elif "days" in h1.columns:
        days = h1["days"].dropna()
    else:
        # Try to find any numeric delay column
        for col in h1.columns:
            if "delay" in col.lower() or "delta" in col.lower():
                days = h1[col].dropna()
                print(f"  Using column '{col}' as delay")
                break
        else:
            raise KeyError(f"No delay column found in {h1_csv.name}. "
                           f"Columns: {list(h1.columns)}")

    days = days[days >= 0].sort_values().reset_index(drop=True)
    n = len(days)
    cdf_vals = (np.arange(1, n + 1)) / n

    full = pd.DataFrame({"days": days.values, "cdf": cdf_vals})
    full.to_csv(out_dir / "cdf_deltas_data_full.csv", index=False)
    print(f"  Written: cdf_deltas_data_full.csv  ({n:,} rows)")

    # Thinned version: keep rows where Δcdf ≥ 0.001
    thin_rows = [0]
    for i in range(1, n):
        if cdf_vals[i] - cdf_vals[thin_rows[-1]] >= 0.001:
            thin_rows.append(i)
    if thin_rows[-1] != n - 1:
        thin_rows.append(n - 1)
    sampled = full.iloc[thin_rows]
    sampled.to_csv(out_dir / "cdf_deltas_data_sampled_y_001.csv", index=False)
    print(f"  Written: cdf_deltas_data_sampled_y_001.csv  ({len(sampled):,} rows)")


def generate_weekly_counts(sh: pd.DataFrame, un: pd.DataFrame, out_dir: Path) -> None:
    """Weekly shield / unshield counts."""
    sh2 = sh.copy()
    un2 = un.copy()
    sh2["week"] = sh2["time"].dt.to_period("W").dt.start_time.dt.date.astype(str)
    un2["week"] = un2["time"].dt.to_period("W").dt.start_time.dt.date.astype(str)

    wks_sh = sh2.groupby("week").size().rename("shields")
    wks_un = un2.groupby("week").size().rename("unshields")
    wks = pd.concat([wks_sh, wks_un], axis=1).fillna(0).astype(int).reset_index()
    wks.columns = ["week_start", "shields", "unshields"]
    wks = wks.sort_values("week_start")
    wks.to_csv(out_dir / "figure_ch4_02_weekly_boundary_counts.csv", index=False)
    print(f"  Written: figure_ch4_02_weekly_boundary_counts.csv  ({len(wks):,} weeks)")


def generate_cumulative_flow(sh: pd.DataFrame, un: pd.DataFrame, out_dir: Path) -> None:
    """Cumulative ETH flow timeseries for FIFO / Little's law analysis."""
    sh2 = sh[["time", "amount_eth"]].copy().sort_values("time")
    un2 = un[["time", "amount_eth"]].copy().sort_values("time")

    sh2["cum_in"]  = sh2["amount_eth"].cumsum()
    un2["cum_out"] = un2["amount_eth"].cumsum()

    # Merge on time grid (daily)
    sh2["date"] = sh2["time"].dt.date
    un2["date"] = un2["time"].dt.date
    daily_in  = sh2.groupby("date")["amount_eth"].sum().cumsum().rename("cum_in")
    daily_out = un2.groupby("date")["amount_eth"].sum().cumsum().rename("cum_out")
    flow = pd.concat([daily_in, daily_out], axis=1).ffill().reset_index()
    flow["date"] = flow["date"].astype(str)
    flow.to_csv(out_dir / "figure_ch4_03_cumulative_pool_flow.csv", index=False)
    print(f"  Written: figure_ch4_03_cumulative_pool_flow.csv  ({len(flow):,} days)")

    # Little's law timeseries: rolling 90d mean balance / mean exit rate
    flow["balance"] = flow["cum_in"].fillna(0) - flow["cum_out"].fillna(0)
    flow.to_csv(out_dir / "figure_ch4_03_little_law_timeseries.csv", index=False)
    print(f"  Written: figure_ch4_03_little_law_timeseries.csv")


def generate_h1_coverage(h1: pd.DataFrame, un: pd.DataFrame, out_dir: Path) -> None:
    """H1 coverage: fraction of withdrawals covered by address reuse."""
    h1_addrs = set(_norm(h1.get("from_addr", h1.get("to_address", pd.Series(dtype=str)))).dropna())
    un2 = un.copy()
    un2["to_l"] = _norm(un2["to_address"])
    un2["h1"] = un2["to_l"].isin(h1_addrs).astype(int)

    # Cumulative coverage over time
    un2 = un2.sort_values("time")
    un2["cum_total"] = range(1, len(un2) + 1)
    un2["cum_h1"]    = un2["h1"].cumsum()
    un2["coverage"]  = un2["cum_h1"] / un2["cum_total"]
    un2["date"]      = un2["time"].dt.date.astype(str)

    cov = un2.groupby("date")[["cum_total", "cum_h1", "coverage"]].last().reset_index()
    cov.to_csv(out_dir / "figure_ch4_06_h1_coverage.csv", index=False)
    print(f"  Written: figure_ch4_06_h1_coverage.csv  ({len(cov):,} days)")


def generate_dataset_inventory(rg: "Path", sh_raw: "pd.DataFrame",
                               un_raw: "pd.DataFrame", out_dir: "Path") -> None:
    """Dataset inventory CSV used by generate_dataset_table.py.

    Reads aggregated (all-asset) shield/unshield CSVs to get total row counts
    and timestamp ranges for the full Railgun dataset (not WETH-only).
    """
    agg_dir = rg / "aggregated"
    sh_agg_csv = agg_dir / "eth_shield_aggregated.csv"
    un_agg_csv = agg_dir / "eth_unshields_aggregated.csv"

    rows = []
    if sh_agg_csv.exists():
        sh_agg = pd.read_csv(sh_agg_csv)
        n = len(sh_agg)
        # timestamps stored as nanoseconds integers
        t_col = "last_time_ns" if "last_time_ns" in sh_agg.columns else sh_agg.columns[0]
        if t_col in sh_agg.columns:
            ts = sh_agg[t_col].dropna().astype(float)
            t_min = pd.to_datetime(ts.min() / 1e9, unit="s", utc=True).date()
            t_max = pd.to_datetime(ts.max() / 1e9, unit="s", utc=True).date()
        else:
            t_min = t_max = ""
        rows.append({"label": "RG Shields (all assets)", "color": "colorRailgun",
                     "rows": n, "start": str(t_min), "end": str(t_max)})
    elif sh_raw is not None:
        # Fallback: use WETH transaction data
        n = len(sh_raw)
        t_min = sh_raw["time"].min().date()
        t_max = sh_raw["time"].max().date()
        rows.append({"label": "RG Shields (WETH)", "color": "colorRailgun",
                     "rows": n, "start": str(t_min), "end": str(t_max)})

    if un_agg_csv.exists():
        un_agg = pd.read_csv(un_agg_csv)
        n = len(un_agg)
        t_col = "first_time_ns" if "first_time_ns" in un_agg.columns else un_agg.columns[0]
        if t_col in un_agg.columns:
            ts = un_agg[t_col].dropna().astype(float)
            t_min = pd.to_datetime(ts.min() / 1e9, unit="s", utc=True).date()
            t_max = pd.to_datetime(ts.max() / 1e9, unit="s", utc=True).date()
        else:
            t_min = t_max = ""
        rows.append({"label": "RG Unshields (all assets)", "color": "colorRailgun",
                     "rows": n, "start": str(t_min), "end": str(t_max)})
    elif un_raw is not None:
        n = len(un_raw)
        t_min = un_raw["time"].min().date()
        t_max = un_raw["time"].max().date()
        rows.append({"label": "RG Unshields (WETH)", "color": "colorRailgun",
                     "rows": n, "start": str(t_min), "end": str(t_max)})

    if rows:
        import csv as _csv
        out = out_dir / "figure_ch4_01_dataset_inventory.csv"
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["label", "color", "rows", "start", "end"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Written: figure_ch4_01_dataset_inventory.csv  ({len(rows)} rows)")
    else:
        print("  ⚠  No aggregated CSV found — figure_ch4_01_dataset_inventory.csv not written")


def main() -> None:
    args = parse_args()
    rg   = args.rg_data
    out  = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    tx_dir  = rg / "transactions"
    agg_dir = rg / "aggregated"
    h1_csv  = rg / "h1_pairs_with_ppoi_tags.csv"

    # ── H1 pairs → CDF ───────────────────────────────────────────────────────
    if h1_csv.exists():
        generate_cdf_deltas(h1_csv, out)
    else:
        print(f"  ⚠  H1 pairs not found at {h1_csv} — skipping CDF deltas")

    # ── Transaction CSVs ─────────────────────────────────────────────────────
    sh_csv = tx_dir / "eth_shield.csv"
    un_csv = tx_dir / "eth_unshields.csv"

    sh_raw = un_raw = None
    if not sh_csv.exists() or not un_csv.exists():
        print(f"  ⚠  Transaction CSVs not found at {tx_dir} — skipping weekly/flow")
    else:
        print("  Loading transaction CSVs …")
        sh_raw = pd.read_csv(sh_csv)
        un_raw = pd.read_csv(un_csv)
        sh_raw["time"] = pd.to_datetime(sh_raw["time"], utc=True, errors="coerce")
        un_raw["time"] = pd.to_datetime(un_raw["time"], utc=True, errors="coerce")
        sh_raw = sh_raw[sh_raw["time"].notna()].copy()
        un_raw = un_raw[un_raw["time"].notna()].copy()

        # Amount column normalisation
        for df in (sh_raw, un_raw):
            if "amount_eth" not in df.columns:
                for col in ("value_eth", "eth_amount", "amount"):
                    if col in df.columns:
                        df.rename(columns={col: "amount_eth"}, inplace=True)
                        break
                else:
                    df["amount_eth"] = 0.0

        # Filter protocol addresses and WETH only
        sh = sh_raw.copy(); un = un_raw.copy()
        if "token_symbol" in sh.columns:
            sh = sh[sh["token_symbol"].str.upper() == "WETH"]
            un = un[un["token_symbol"].str.upper() == "WETH"]
        if "from_address" in sh.columns:
            sh = sh[~_norm(sh["from_address"]).isin(PROTOCOL_ADDRS)]
        if "to_address" in un.columns:
            un = un[~_norm(un["to_address"]).isin(PROTOCOL_ADDRS)]

        generate_weekly_counts(sh, un, out)
        generate_cumulative_flow(sh, un, out)

        if h1_csv.exists():
            h1 = pd.read_csv(h1_csv)
            generate_h1_coverage(h1, un, out)

    # ── Dataset inventory (uses aggregated CSVs when available) ──────────────
    generate_dataset_inventory(rg, sh_raw, un_raw, out)

    print("  generate_rg_raw.py done.")


if __name__ == "__main__":
    main()
