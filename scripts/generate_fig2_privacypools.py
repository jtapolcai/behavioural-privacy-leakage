#!/usr/bin/env python3
"""Generate the Privacy Pools weekly event-count series used in Figure 2.

The output is read directly by the PGFPlots source, so refreshing Figure 2 only
requires rerunning this script and recompiling the paper.
"""

import argparse
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
DEFAULT_PP_DATA = (
    BASE.parent / "privacypools-deanonymization-main" / "data" / "processed"
)
DEFAULT_OUTPUT = BASE / "Figures" / "figure_ch4_02_weekly_boundary_counts_pp.csv"


def parse_utc(series: pd.Series) -> pd.Series:
    """Parse both dot- and comma-millisecond timestamps in the exports."""
    normalized = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_datetime(
        normalized, format="%Y-%m-%d %H:%M:%S.%f UTC", utc=True
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pp-data", type=Path, default=DEFAULT_PP_DATA)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument(
    "--last-complete-week",
    type=str,
    help="Optional Sunday cutoff in YYYY-MM-DD format (default: inferred).",
)
args = parser.parse_args()

deposits = pd.read_csv(args.pp_data / "processed_privacypools_eth_pool_deposits.csv")
withdrawals = pd.read_csv(args.pp_data / "processed_privacypools_data_withdraws.csv")

deposit_times = parse_utc(deposits["time"])
withdrawal_times = parse_utc(withdrawals["evt_block_time"])

if args.last_complete_week:
    last_complete_week = pd.Timestamp(args.last_complete_week, tz="UTC")
else:
    # The week containing the older export's final event may be partial.  Keep
    # the Sunday immediately before that Monday-to-Sunday observation week.
    common_export_end = min(deposit_times.max(), withdrawal_times.max())
    week_start = common_export_end.tz_localize(None).to_period("W-SUN").start_time
    last_complete_week = pd.Timestamp(week_start - pd.Timedelta(days=1), tz="UTC")

deposit_counts = (
    pd.Series(1, index=deposit_times)
    .resample("W-SUN")
    .sum()
    .rename("deposits")
)
withdrawal_counts = (
    pd.Series(1, index=withdrawal_times)
    .resample("W-SUN")
    .sum()
    .rename("withdrawals")
)

weekly = (
    pd.concat([deposit_counts, withdrawal_counts], axis=1)
    .fillna(0)
    .astype(int)
    .loc[:last_complete_week]
)
weekly.index = weekly.index.strftime("%Y-%m-%d")
weekly.index.name = "week_end"
args.output.parent.mkdir(parents=True, exist_ok=True)
weekly.to_csv(args.output)

print(
    f"Wrote {len(weekly)} complete weeks through "
    f"{last_complete_week:%Y-%m-%d} to {args.output}"
)
