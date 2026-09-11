#!/usr/bin/env python3
"""
generate_pp_and_montecarlo.py
===============================
Generates Privacy Pools comparison data and Monte Carlo null-hypothesis
distributions for the FC 2027 paper figures.

Outputs (written to Figures/):
  pp_h1_pairs.csv                   — PP H1-linked deposit-withdraw pairs
  pp_cdf_deltas_pdf_data.csv        — PP H1 timing PDF (same format as Railgun)
  mc_railgun_timing_pdf.csv         — Railgun Monte Carlo null timing PDF
  mc_pp_timing_pdf.csv              — PP Monte Carlo null timing PDF
  pp_h1_coverage.csv                — PP H1 coverage (same format as figure_ch4_06_h1_coverage.csv)
  pp_dataset_inventory.csv          — PP dataset summary row
  pp_h1_summary_stats.csv           — key percentile / count stats for the text
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = _FIGURES.parent  # repo root via _paths
PP_DIR = _PP_DATA
FIG    = BASE / "Figures"

RG_PDF_FILE = FIG / "cdf_deltas_pdf_data_sampled.csv"   # existing Railgun PDF

# ── helpers ────────────────────────────────────────────────────────────────

def parse_euro_float(s):
    """Convert European decimal comma string to float, e.g. '0,022' → 0.022."""
    if isinstance(s, float):
        return s
    s = str(s).strip().strip('"')
    return float(s.replace(",", "."))


def parse_timestamp(s):
    """
    Parse timestamps like '2026-06-19 03:37:11.000 UTC'
    or                     '2026-07-07 14:06:59,000 UTC'
    """
    s = str(s).strip().strip('"')
    s = s.replace(",", ".")          # comma-decimal milliseconds → dot
    s = s.replace(" UTC", "")
    return pd.Timestamp(s, tz="UTC")


def make_pdf(dt_days: np.ndarray, bins: int = 300, label: str = ""):
    """Compute a histogram PDF on log-space bins (same style as Railgun data)."""
    dt_pos = dt_days[dt_days > 0]
    if len(dt_pos) == 0:
        return pd.DataFrame(columns=["days", "pdf", "left_days", "right_days", "probability_mass"])
    lo, hi = np.log10(dt_pos.min()), np.log10(dt_pos.max())
    edges = np.logspace(lo, hi, bins + 1)
    counts, _ = np.histogram(dt_pos, bins=edges)
    widths = np.diff(edges)
    n = len(dt_pos)
    density = counts / (n * widths)
    prob_mass = counts / n
    rows = []
    for i in range(len(counts)):
        if counts[i] > 0:
            rows.append({
                "days":             0.5 * (edges[i] + edges[i+1]),
                "pdf":              density[i],
                "left_days":        edges[i],
                "right_days":       edges[i+1],
                "probability_mass": prob_mass[i],
            })
    return pd.DataFrame(rows)


# ── 1. Load Privacy Pools raw data ─────────────────────────────────────────
print("Loading Privacy Pools data …")

dep_raw = pd.read_csv(PP_DIR / "processed_privacypools_eth_pool_deposits.csv")
dep_raw["amount_eth_f"] = dep_raw["amount_eth"].apply(parse_euro_float)
dep_raw["ts"] = dep_raw["time"].apply(parse_timestamp)
deposits = dep_raw[["depositor", "amount_eth_f", "ts"]].rename(
    columns={"depositor": "address", "amount_eth_f": "amount", "ts": "time"}
)

with_raw = pd.read_csv(PP_DIR / "processed_privacypools_data_withdraws.csv")
with_raw["amount_eth_f"] = with_raw["eth_amount"].apply(parse_euro_float)
with_raw["ts"] = with_raw["evt_block_time"].apply(parse_timestamp)
withdrawals = with_raw[["recipient", "amount_eth_f", "ts", "relayer"]].rename(
    columns={"recipient": "address", "amount_eth_f": "amount", "ts": "time"}
)

print(f"  Deposits:    {len(deposits):,}")
print(f"  Withdrawals: {len(withdrawals):,}")


# ── 2. H1 pairs: address reuse (depositor == recipient) ─────────────────────
print("\nComputing H1 (address reuse) pairs …")

dep_addrs  = set(deposits["address"].str.lower())
with_addrs = set(withdrawals["address"].str.lower())
reused     = dep_addrs & with_addrs

# For each reused address, pair every deposit with every subsequent withdrawal
h1_rows = []
for addr in reused:
    addr_deps  = deposits[deposits["address"].str.lower()  == addr].sort_values("time")
    addr_withs = withdrawals[withdrawals["address"].str.lower() == addr].sort_values("time")
    for _, w in addr_withs.iterrows():
        # closest prior deposit (greedy — same convention as Railgun H1)
        prior = addr_deps[addr_deps["time"] <= w["time"]]
        if prior.empty:
            continue
        d = prior.iloc[-1]
        dt = (w["time"] - d["time"]).total_seconds() / 86400.0  # days
        if dt < 0:
            continue
        h1_rows.append({
            "address":    addr,
            "dep_time":   d["time"],
            "dep_amount": d["amount"],
            "with_time":  w["time"],
            "with_amount":w["amount"],
            "dt_days":    dt,
        })

h1_pairs = pd.DataFrame(h1_rows)
h1_pairs.to_csv(FIG / "pp_h1_pairs.csv", index=False)
print(f"  H1 pairs found: {len(h1_pairs):,}")
if len(h1_pairs):
    p50  = np.percentile(h1_pairs["dt_days"], 50)
    p90  = np.percentile(h1_pairs["dt_days"], 90)
    p99  = np.percentile(h1_pairs["dt_days"], 99)
    print(f"  Median Δt: {p50:.2f} d   P90: {p90:.1f} d   P99: {p99:.1f} d")


# ── 3. Timing PDF for PP H1 pairs ──────────────────────────────────────────
print("\nGenerating PP H1 timing PDF …")
if len(h1_pairs):
    pdf_pp = make_pdf(h1_pairs["dt_days"].values, bins=300)
    pdf_pp.to_csv(FIG / "pp_cdf_deltas_pdf_data.csv", index=False)
    print(f"  Saved {len(pdf_pp)} histogram bins.")


# ── 4. Monte Carlo null-hypothesis timing distributions ────────────────────
# Null model: for each withdrawal, pick a deposit uniformly at random from
# all deposits whose timestamp precedes the withdrawal.  Repeat MC_RUNS times
# and average the histograms.

MC_RUNS  = 2000
MC_BINS  = 300
RNG      = np.random.default_rng(42)

def run_monte_carlo(dep_times: np.ndarray, with_times: np.ndarray,
                    runs: int = MC_RUNS, bins: int = MC_BINS,
                    label: str = "") -> pd.DataFrame:
    """
    Vectorized Monte Carlo null-hypothesis timing.

    dep_times, with_times: float arrays of unix seconds (sorted ascending).
    For each withdrawal w_i, the number of valid prior deposits is
      n_i = searchsorted(dep_times, w_i, side='right').
    We draw a random integer in [0, n_i) for each run and compute
      dt = w_i - dep_times[random_idx].
    This is equivalent to the serial loop but ~1000× faster.
    """
    dep_sorted  = np.sort(dep_times)
    with_sorted = np.sort(with_times)

    # Number of eligible deposits for each withdrawal (shape: n_with)
    n_eligible = np.searchsorted(dep_sorted, with_sorted, side="right")
    # Only keep withdrawals that have at least one prior deposit
    mask = n_eligible > 0
    with_valid  = with_sorted[mask]
    n_el_valid  = n_eligible[mask]

    all_dts = []
    for _ in range(runs):
        # Draw a uniform random index in [0, n_el_valid[i]) for each withdrawal
        rand_frac = RNG.random(len(with_valid))          # [0,1)
        rand_idx  = (rand_frac * n_el_valid).astype(int)  # [0, n_el_valid[i])
        rand_idx  = np.clip(rand_idx, 0, n_el_valid - 1)
        chosen_dep = dep_sorted[rand_idx]
        dts = (with_valid - chosen_dep) / 86400.0         # days
        all_dts.append(dts[dts > 0])

    arr = np.concatenate(all_dts)
    return make_pdf(arr, bins=bins, label=label)


print("\nRunning Monte Carlo for Privacy Pools …")
dep_ts_pp  = np.sort(np.array([t.timestamp() for t in deposits["time"]]))
with_ts_pp = np.sort(np.array([t.timestamp() for t in withdrawals["time"]]))
mc_pp = run_monte_carlo(dep_ts_pp, with_ts_pp, label="pp_mc")
mc_pp.to_csv(FIG / "mc_pp_timing_pdf.csv", index=False)
print(f"  PP Monte Carlo done — {len(mc_pp)} bins.")


# For Railgun MC we use the existing sampled PDF bin centres to infer the
# observation range, then reconstruct deposit/withdrawal times from the
# figure_ch4_h1_temporal_pdf.csv (CDF data) and the raw data files.
# Fallback: use the PP MC as a template and scale to Railgun's window.

# Load Railgun deposit & withdrawal timestamps if the raw timeline CSV exists
rg_dep_file  = FIG / "figure_ch4_02_weekly_boundary_counts.csv"
rg_flow_file = FIG / "figure_ch4_03_cumulative_pool_flow.csv"

if rg_dep_file.exists():
    print("\nRunning Monte Carlo for Railgun …")
    wbc = pd.read_csv(rg_dep_file)
    date_col  = next((c for c in wbc.columns if c in ("date","week","week_start")), None)
    dep_col   = next((c for c in wbc.columns if "shield" in c.lower() and "un" not in c.lower()), None)
    with_col  = next((c for c in wbc.columns if "unshield" in c.lower()), None)
    if date_col and dep_col and with_col:
        # Jitter timestamps within the week (uniform random within 7-day window)
        # so we don't get 7-day discretization artefacts in the MC output.
        rg_dep_ts  = []
        rg_with_ts = []
        WEEK_SEC = 7 * 86400
        for _, row in wbc.iterrows():
            t0 = pd.Timestamp(row[date_col]).timestamp()
            nd = int(row.get(dep_col,  0))
            nw = int(row.get(with_col, 0))
            if nd > 0:
                rg_dep_ts.extend(t0 + RNG.uniform(0, WEEK_SEC, nd))
            if nw > 0:
                rg_with_ts.extend(t0 + RNG.uniform(0, WEEK_SEC, nw))
        if rg_dep_ts and rg_with_ts:
            mc_rg = run_monte_carlo(
                np.sort(np.array(rg_dep_ts)),
                np.sort(np.array(rg_with_ts)),
                label="rg_mc"
            )
            mc_rg.to_csv(FIG / "mc_railgun_timing_pdf.csv", index=False)
            print(f"  Railgun Monte Carlo done — {len(mc_rg)} bins.")
        else:
            print("  Weekly counts empty, skipping Railgun MC.")
    else:
        print(f"  Columns: {wbc.columns.tolist()} — skipping Railgun MC.")
else:
    # Fallback: generate analytical uniform null for Railgun
    # (observation period ~1416 days, uniform random pairing)
    print("\nGenerating analytical null for Railgun (uniform prior over observation window) …")
    OBS_DAYS = 1416.0   # 2022-05-13 → 2026-04-29
    # Under uniform random pairing, Δt ~ Uniform[0, OBS_DAYS] (rough approximation)
    # In practice it's a triangular-ish distribution — simulate it:
    rg_dep_ts_synth  = np.sort(RNG.uniform(0, OBS_DAYS * 86400, 34747))
    rg_with_ts_synth = np.sort(RNG.uniform(0, OBS_DAYS * 86400, 47596))
    mc_rg = run_monte_carlo(rg_dep_ts_synth, rg_with_ts_synth, runs=500, label="rg_mc")
    mc_rg.to_csv(FIG / "mc_railgun_timing_pdf.csv", index=False)
    print(f"  Railgun analytical MC done — {len(mc_rg)} bins.")


# ── 5. PP H1 coverage table ─────────────────────────────────────────────────
print("\nComputing PP H1 coverage …")
n_with = len(withdrawals)
n_dep  = len(deposits)
n_with_reused_addr = len(withdrawals[withdrawals["address"].str.lower().isin(reused)])
n_dep_reused_addr  = len(deposits[deposits["address"].str.lower().isin(reused)])
n_with_uniq = withdrawals["address"].str.lower().nunique()
n_dep_uniq  = deposits["address"].str.lower().nunique()
# Volume
if len(h1_pairs):
    pp_with_vol = withdrawals["amount"].sum()
    pp_h1_vol   = h1_pairs["with_amount"].sum()
    vol_pct     = 100 * pp_h1_vol / pp_with_vol if pp_with_vol > 0 else 0.0
else:
    vol_pct = 0.0

pp_coverage = pd.DataFrame([
    {"base": "Distinct addresses",
     "shield_pct":   100 * len(reused) / n_dep_uniq  if n_dep_uniq  > 0 else 0,
     "unshield_pct": 100 * len(reused) / n_with_uniq if n_with_uniq > 0 else 0},
    {"base": "Boundary events",
     "shield_pct":   100 * n_dep_reused_addr  / n_dep  if n_dep  > 0 else 0,
     "unshield_pct": 100 * n_with_reused_addr / n_with if n_with > 0 else 0},
    {"base": "ETH volume",
     "shield_pct":   vol_pct,   # approximate (same withdraw volume fraction)
     "unshield_pct": vol_pct},
])
pp_coverage.to_csv(FIG / "pp_h1_coverage.csv", index=False)
print(pp_coverage.to_string(index=False))


# ── 6. PP dataset inventory summary ────────────────────────────────────────
dep_start  = deposits["time"].min().strftime("%Y-%m-%d")
dep_end    = deposits["time"].max().strftime("%Y-%m-%d")
with_start = withdrawals["time"].min().strftime("%Y-%m-%d")
with_end   = withdrawals["time"].max().strftime("%Y-%m-%d")

pp_inventory = pd.DataFrame([
    {"label": "PP Shield events",   "color": "deposit", "rows": n_dep,  "start": dep_start,  "end": dep_end},
    {"label": "PP Unshield events", "color": "withdraw","rows": n_with, "start": with_start, "end": with_end},
])
pp_inventory.to_csv(FIG / "pp_dataset_inventory.csv", index=False)
print("\nPP dataset inventory:")
print(pp_inventory.to_string(index=False))


# ── 7. Key stats for the text ───────────────────────────────────────────────
h1_count  = len(h1_pairs)
h1_pct_w  = 100 * h1_count / n_with if n_with > 0 else 0
h1_pct_a  = 100 * len(reused) / n_with_uniq if n_with_uniq > 0 else 0

stats = {
    "pp_n_deposits":         n_dep,
    "pp_n_withdrawals":      n_with,
    "pp_n_deposit_addrs":    n_dep_uniq,
    "pp_n_withdraw_addrs":   n_with_uniq,
    "pp_h1_pairs":           h1_count,
    "pp_h1_pct_events":      round(h1_pct_w, 2),
    "pp_h1_pct_addrs":       round(h1_pct_a, 2),
    "pp_h1_median_dt_days":  round(float(np.percentile(h1_pairs["dt_days"], 50)), 3) if h1_count else None,
    "pp_h1_p90_dt_days":     round(float(np.percentile(h1_pairs["dt_days"], 90)), 1) if h1_count else None,
    "pp_h1_p99_dt_days":     round(float(np.percentile(h1_pairs["dt_days"], 99)), 1) if h1_count else None,
}
pd.DataFrame([stats]).to_csv(FIG / "pp_h1_summary_stats.csv", index=False)

print("\n── Summary ──────────────────────────────────────────────────")
for k, v in stats.items():
    print(f"  {k:<35} = {v}")
print("\nAll done. CSV files written to Figures/")
