#!/usr/bin/env python3
"""Generate Privacy Pools cumulative-flow and FIFO-lag data for Figure 5."""

from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

import numpy as np
import pandas as pd


BASE = _FIGURES.parent
PP_DATA = _PP_DATA
FIGURES = _FIGURES


def weekly_plot_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Keep weekly points plus the final observation for efficient PGFPlots."""
    sampled = data.iloc[::7].copy()
    if sampled.index[-1] != data.index[-1]:
        sampled = pd.concat([sampled, data.iloc[[-1]]])
    return sampled


def parse_time(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_datetime(normalized, format="%Y-%m-%d %H:%M:%S.%f UTC", utc=True)


def parse_amount(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False))


deposits = pd.read_csv(PP_DATA / "processed_privacypools_eth_pool_deposits.csv")
withdrawals = pd.read_csv(PP_DATA / "processed_privacypools_data_withdraws.csv")
deposits["timestamp"] = parse_time(deposits["time"])
withdrawals["timestamp"] = parse_time(withdrawals["evt_block_time"])
deposits["amount"] = parse_amount(deposits["amount_eth"])
withdrawals["amount"] = parse_amount(withdrawals["eth_amount"])

# Stop at the end of the shorter export so both cumulative curves have equal
# temporal coverage and no artificial zero-deposit tail is introduced.
common_end = min(deposits["timestamp"].max(), withdrawals["timestamp"].max())
deposits = deposits.loc[deposits["timestamp"] <= common_end]
withdrawals = withdrawals.loc[withdrawals["timestamp"] <= common_end]

daily_deposits = deposits.set_index("timestamp")["amount"].resample("D").sum()
daily_withdrawals = withdrawals.set_index("timestamp")["amount"].resample("D").sum()
dates = pd.date_range(
    min(daily_deposits.index.min(), daily_withdrawals.index.min()),
    common_end.normalize(),
    freq="D",
    tz="UTC",
)
flows = pd.DataFrame(index=dates)
flows["deposit"] = daily_deposits.reindex(dates, fill_value=0)
flows["withdraw"] = daily_withdrawals.reindex(dates, fill_value=0)
flows["cum_deposit"] = flows["deposit"].cumsum()
flows["cum_withdraw"] = flows["withdraw"].cumsum()
flows["retained"] = flows["cum_deposit"] - flows["cum_withdraw"]

# FIFO horizontal lag: at date t, invert the cumulative-deposit curve at the
# cumulative amount withdrawn by t, then measure the horizontal distance to t.
deposit_curve = flows["cum_deposit"].to_numpy()
withdraw_curve = flows["cum_withdraw"].to_numpy()
fifo_days = np.full(len(flows), np.nan)
for i, withdrawn in enumerate(withdraw_curve):
    if withdrawn <= 0:
        continue
    source_index = np.searchsorted(deposit_curve[: i + 1], withdrawn, side="left")
    if source_index <= i:
        fifo_days[i] = i - source_index
flows["fifo_horizontal_lag_days"] = fifo_days

output = flows.reset_index(names="date")
output["date"] = output["date"].dt.strftime("%Y-%m-%d")
output[["date", "cum_deposit", "cum_withdraw", "retained"]].to_csv(
    FIGURES / "figure_ch4_03_pp_cumulative_pool_flow.csv", index=False
)
output[["date", "retained", "fifo_horizontal_lag_days"]].to_csv(
    FIGURES / "figure_ch4_03_pp_retained_fifo.csv", index=False
)

# Plot-only weekly samples. The full daily files above remain the authoritative
# analysis data; these smaller files only reduce pdfLaTeX/PGFPlots workload.
weekly_plot_sample(output)[
    ["date", "cum_deposit", "cum_withdraw", "retained"]
].to_csv(FIGURES / "figure_ch4_03_pp_cumulative_pool_flow_plot.csv", index=False)
weekly_plot_sample(output)[
    ["date", "retained", "fifo_horizontal_lag_days"]
].to_csv(FIGURES / "figure_ch4_03_pp_retained_fifo_plot.csv", index=False)

# Railgun's authoritative daily files are already part of the paper artifact.
# Create equivalent weekly plot samples without modifying those source files.
railgun_flow = pd.read_csv(FIGURES / "figure_ch4_03_cumulative_pool_flow.csv")
weekly_plot_sample(railgun_flow).to_csv(
    FIGURES / "figure_ch4_03_cumulative_pool_flow_plot.csv", index=False
)
railgun_fifo = pd.read_csv(FIGURES / "figure_ch4_03_little_law_timeseries.csv")
weekly_plot_sample(railgun_fifo).to_csv(
    FIGURES / "figure_ch4_03_little_law_timeseries_plot.csv", index=False
)

print(
    f"Wrote {len(output)} daily Privacy Pools observations through "
    f"{output.date.iloc[-1]} and weekly plot samples for both systems."
)
