#!/usr/bin/env python3
"""compute_rg_entropy_models.py
=============================================================
Computes M0, M1, M2 entropy models for Railgun using raw event data
from railgun_deanonymization-48A0.

M0: H_naive_item = log2(pool_bucketed) — already in witness files
M1: temporal prior entropy, computed from deposit timestamps + reuse_mix model
M2: H_knap_item = witness participation entropy — already in witness files

Outputs (written to ../Figures/entropy_models/):
  rg_entropy_per_withdrawal.csv  — per-withdrawal M0, M1, M2 for each horizon
  rg_entropy_summary.csv         — median/quantiles per horizon per model
  rg_m1_params.json              — fitted temporal model parameters
  rg_entropy_vs_T_tikz.tex       — TikZ figure comparing Railgun and PP
"""

from __future__ import annotations
import csv
import json
import math
import warnings
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

import numpy as np

# ── paths ───────────────────────────────────────────────────────────────────
PAPER   = Path(__file__).resolve().parents[1]
RG_DATA = _RG_DATA.parent  # via _paths (points to railgun_deanonymization/)
OUT     = PAPER / "Figures" / "entropy_models"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

HORIZONS_DAYS = [3, 7, 30, 180, None]   # None = all-time
HORIZON_LABELS = {3: "3d", 7: "7d", 30: "30d", 180: "180d", None: "all"}

WITNESS_STEP = "0.0001"   # finest bucket granularity available

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_ts(s: str) -> float:
    """Parse '2026-04-29 20:23:59.000 UTC' → Unix timestamp in seconds."""
    s = s.strip().rstrip(" UTC").replace(",", ".")
    import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse timestamp: {s!r}")

def entropy_bits(weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return math.nan
    p = weights[weights > 0] / total
    return float(-np.sum(p * np.log2(p)))

def quantiles(arr, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"p{int(q*100)}": math.nan for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# ── 1. Load raw events ───────────────────────────────────────────────────────
print("Loading aggregated shield events …")
# For kto1 direction, the POOL is shields, indexed by last_time_ns.
# The witness_entropy_pass.py uses eth_shield_aggregated.csv as pool for kto1.
shield_agg = list(csv.DictReader(open(RG_DATA / "data/aggregated/eth_shield_aggregated.csv")))
dep_times   = np.array([int(r["last_time_ns"]) / 1e9 for r in shield_agg])
dep_amounts = np.array([int(r["note_wei"]) / 1e18 for r in shield_agg])  # wei → ETH

# Amount buckets — same bucketing used by witness_entropy_pass.py
# step_wei = 0.0001 ETH = 1e14 wei (WITNESS_STEP = "0.0001")
STEP_WEI = int(float(WITNESS_STEP) * 1e18)   # 100_000_000_000_000
# Deposit note_wei as Python ints (can exceed int64; values up to ~100 ETH = 1e20 wei)
dep_amounts_wei_py = [int(r["note_wei"]) for r in shield_agg]
# Convert to numpy float64 for fast comparison (precision is fine for ≤ comparisons
# at ~1e14 resolution; ETH amounts are well within float64's 53-bit mantissa for
# values ≤ 100 ETH = 1e20, where ULP ≈ 2^4 ≈ 16 wei << step_wei = 1e14)
dep_note_wei_f64 = np.array(dep_amounts_wei_py, dtype=np.float64)

# Sort by time
sort_idx         = np.argsort(dep_times)
dep_times        = dep_times[sort_idx]
dep_amounts      = dep_amounts[sort_idx]
dep_note_wei_f64 = dep_note_wei_f64[sort_idx]
# Also keep address and agg_size arrays in the same sorted order (needed for M4)
dep_addrs     = [shield_agg[i]["address"].lower() for i in sort_idx]
dep_agg_sizes = np.array([int(shield_agg[i]["agg_size"]) for i in sort_idx], dtype=np.int32)
print(f"  {len(dep_times):,} agg shields, {dep_times[0]:.0f} → {dep_times[-1]:.0f}")

print("Loading aggregated unshield events …")
# For kto1 direction, TARGETS are unshields, indexed by first_time_ns.
# ti in witness file = agg_id = row index in eth_unshields_aggregated.csv
unshield_agg = list(csv.DictReader(open(RG_DATA / "data/aggregated/eth_unshields_aggregated.csv")))
# Build lookup: agg_id → timestamp  (agg_id is just the CSV row index)
with_times_by_ti = {i: int(r["first_time_ns"]) / 1e9
                    for i, r in enumerate(unshield_agg)}
with_amounts_by_ti = {i: int(r["note_wei"]) / 1e18
                      for i, r in enumerate(unshield_agg)}
# Withdrawal note_wei as float64 for fast numpy masking
# (pool filter: deposit note_wei <= withdrawal note_wei, same as witness script's note[j] > T check)
with_note_wei_f64_by_ti = {i: float(r["note_wei"])
                            for i, r in enumerate(unshield_agg)}
# Address and reuse flag for each unshield (needed for M4)
un_addrs  = [r["address"].lower() for r in unshield_agg]
un_reuse  = [r["tag_reuse"] == "True" for r in unshield_agg]
un_time_ns_arr = np.array([int(r["first_time_ns"]) for r in unshield_agg])

# Also build a sorted array for reference
with_times_sorted = np.array(sorted(with_times_by_ti.values()))
print(f"  {len(unshield_agg):,} agg unshields, "
      f"{with_times_sorted[0]:.0f} → {with_times_sorted[-1]:.0f}")

# ── Precompute cumulative ETH flows for retained-balance estimate ─────────────
# dep_times and dep_note_wei_f64 are already sorted by time.
cum_shield_eth = np.cumsum(dep_note_wei_f64 / 1e18)   # (n_shields,)

# Sort unshields by time for cumulative sum
_un_t   = np.array([with_times_by_ti[i]         for i in range(len(unshield_agg))])
_un_wei = np.array([with_note_wei_f64_by_ti[i]  for i in range(len(unshield_agg))])
_un_ord = np.argsort(_un_t)
_un_t_sorted   = _un_t[_un_ord]
_un_wei_sorted = _un_wei[_un_ord]
cum_unshield_eth = np.cumsum(_un_wei_sorted / 1e18)   # (n_unshields,)

def get_rho_rem(t_w: float) -> float:
    """P(retained balance) at withdrawal time t_w.
    = (cumulative_shield_ETH - cumulative_unshield_ETH) / cumulative_shield_ETH
    Clipped to [0, 1].
    """
    n_sh = int(np.searchsorted(dep_times, t_w, side="left"))
    if n_sh == 0:
        return 0.0
    total_in  = float(cum_shield_eth[n_sh - 1])
    n_un = int(np.searchsorted(_un_t_sorted, t_w, side="left"))
    total_out = float(cum_unshield_eth[n_un - 1]) if n_un > 0 else 0.0
    b_ret = max(0.0, total_in - total_out)
    return min(1.0, b_ret / total_in) if total_in > 0 else 0.0

def h_bin(p: float) -> float:
    """Binary entropy."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

# ── 2. Fit temporal model (reuse_mix) from Railgun H1 pairs ─────────────────
print("\nFitting temporal model from Railgun H1 pairs …")
h1_rows = list(csv.DictReader(open(RG_DATA / "data/h1_pairs_with_ppoi_tags.csv")))
# delay_h = delay in hours
h1_delays_days = np.array([float(r["delay_h"]) / 24.0 for r in h1_rows if r.get("delay_h")])
h1_delays_days = h1_delays_days[h1_delays_days > 0]
print(f"  {len(h1_delays_days):,} H1 pairs, median delay: {np.median(h1_delays_days):.2f} days")

# 2-exp EM fit
def fit_2exp_em(data: np.ndarray, n_iter: int = 200, seed: int = 0):
    """EM for mixture of 2 exponentials."""
    rng = np.random.default_rng(seed)
    # Init: π = [0.5, 0.5], means from k-means style
    mu = np.array([np.percentile(data, 25), np.percentile(data, 75)])
    pi = np.array([0.5, 0.5])
    for _ in range(n_iter):
        # E-step: responsibility of each component for each datum
        lam = 1.0 / mu
        r = pi[:, None] * (lam[:, None] * np.exp(-lam[:, None] * data[None, :]))  # (2, N)
        r /= r.sum(axis=0, keepdims=True)
        # M-step
        Nk = r.sum(axis=1)  # (2,)
        pi = Nk / Nk.sum()
        mu = (r * data[None, :]).sum(axis=1) / Nk
    return pi.tolist(), (1.0 / mu).tolist(), mu.tolist()

pi_rg, lam_rg, mu_rg = fit_2exp_em(h1_delays_days)
print(f"  Reuse-mix fit: π={[round(p,3) for p in pi_rg]}, "
      f"μ=[{mu_rg[0]:.2f}d, {mu_rg[1]:.2f}d]")

# Also compute Little's Law mean from the Little timeseries
ll_path = PAPER.parent / "RailGunDeanonymizationFC27" / "Figures" / "figure_ch4_03_little_law_timeseries.csv"
if ll_path.exists():
    ll_rows = list(csv.DictReader(open(ll_path)))
    little_vals = [float(r["little_days_global"]) for r in ll_rows
                   if r["little_days_global"] and float(r["little_days_global"]) > 1]
    little_mean = np.median(little_vals)  # median of time-series values
    fifo_vals = [float(r["fifo_horizontal_lag_days"]) for r in ll_rows
                 if r.get("fifo_horizontal_lag_days") and r["fifo_horizontal_lag_days"]]
    fifo_mean = np.median([v for v in fifo_vals if v > 0])
    print(f"  Little median: {little_mean:.2f} days, FIFO median: {fifo_mean:.2f} days")
else:
    little_mean = 254.0   # from last row
    fifo_mean = 187.0
    print(f"  Little: {little_mean} d (from CSV last row), FIFO: {fifo_mean} d")

model_params = {
    "reuse_mix": {"type": "mixture", "pi": pi_rg, "lam": lam_rg, "mu_days": mu_rg},
    "fifo_exp":  {"type": "exp", "mu_days": float(fifo_mean)},
    "little_exp":{"type": "exp", "mu_days": float(little_mean)},
}

def temporal_weights(dep_t: np.ndarray, w_time: float, model_name: str) -> np.ndarray:
    """Compute unnormalized temporal weights f_θ(Δt) for preceding deposits."""
    dt_days = (w_time - dep_t) / 86400.0
    dt_days = np.clip(dt_days, 1e-9, None)
    p = model_params[model_name]
    if p["type"] == "exp":
        lam = 1.0 / p["mu_days"]
        return lam * np.exp(-lam * dt_days)
    # mixture
    pis  = np.array(p["pi"])
    lams = np.array(p["lam"])
    return (pis[:, None] * lams[:, None] * np.exp(-lams[:, None] * dt_days[None, :])).sum(axis=0)

# ── 3. Load witness files (M0 and M2 already computed) ──────────────────────
print("\nLoading witness files …")
witness_data = {}   # horizon_label -> {ti -> row}
for T in HORIZONS_DAYS:
    label = HORIZON_LABELS[T]
    fname = f"witness_kto1_{label}_step{WITNESS_STEP}.csv"
    fpath = RG_DATA / "results" / "witness" / fname
    if not fpath.exists():
        print(f"  WARNING: {fname} not found, skipping")
        continue
    rows = list(csv.DictReader(open(fpath)))
    witness_data[label] = {int(r["ti"]): r for r in rows}
    print(f"  [{label}]: {len(rows):,} rows, "
          f"saturated {sum(1 for r in rows if r['saturated']=='True'):,}")

# ── 4. Compute M1 per withdrawal using chunked numpy ─────────────────────────
# Match withdrawals to their sorted index in with_times.
# The witness 'ti' value is the index in the sorted withdrawal array.
# Verify: range of ti matches len(with_times)
all_ti = sorted(witness_data.get("all", {}).keys())
print(f"\nM1 computation: {len(unshield_agg):,} withdrawals, {len(dep_times):,} deposits")
print(f"  ti range in witness 'all': {min(all_ti)} – {max(all_ti)}")

# Compute M1 for all withdrawals (those with a witness row at 'all' horizon)
TIME_MODELS = ["reuse_mix", "fifo_exp", "little_exp"]

CHUNK = 200  # process N withdrawals at a time to limit memory
ti_list = sorted(witness_data.get("all", {}).keys())
n_total = len(ti_list)
print(f"  Computing M1 for {n_total:,} withdrawals in chunks of {CHUNK} …")

# ti maps directly to agg_id in eth_unshields_aggregated.csv
# Get withdrawal time for each ti from the lookup dict
m1_results     = {m: {} for m in TIME_MODELS}   # model -> {ti -> H1}
m1_ext_results = {m: {} for m in TIME_MODELS}   # model -> {ti -> H1_ext (retained-balance)}
rho_rem_cache  = {}   # ti -> ρ_rem

for chunk_start in range(0, n_total, CHUNK):
    chunk_ti = ti_list[chunk_start: chunk_start + CHUNK]
    w_times_chunk = np.array([with_times_by_ti[ti] for ti in chunk_ti])  # shape (C,)

    # For each withdrawal in this chunk, find how many deposits precede it
    n_preceding = np.searchsorted(dep_times, w_times_chunk, side="left")  # (C,)

    for model_name in TIME_MODELS:
        H1_chunk     = np.full(len(chunk_ti), math.nan)
        H1_ext_chunk = np.full(len(chunk_ti), math.nan)

        for ci, (ti_val, w_t, n_dep) in enumerate(
                zip(chunk_ti, w_times_chunk, n_preceding)):
            w_amount = with_note_wei_f64_by_ti[ti_val]

            if n_dep == 0:
                H1_chunk[ci] = 0.0
                H1_ext_chunk[ci] = 0.0
                rho_rem_cache[ti_val] = 0.0
                continue

            dep_t_sub_all = dep_times[:n_dep]
            dep_note_sub  = dep_note_wei_f64[:n_dep]

            # ── Pool_small: note_wei ≤ W (current M1 pool) ──
            mask_small = dep_note_sub <= w_amount
            dep_t_small = dep_t_sub_all[mask_small]

            if len(dep_t_small) == 0:
                H1_small = 0.0
            else:
                w_small  = temporal_weights(dep_t_small, w_t, model_name)
                H1_small = entropy_bits(w_small)
            H1_chunk[ci] = H1_small

            # ── Extended M1: mix in Pool_large (note_wei > W) ──
            rho = get_rho_rem(w_t)
            rho_rem_cache[ti_val] = rho

            if rho > 0:
                mask_large = dep_note_sub > w_amount
                dep_t_large = dep_t_sub_all[mask_large]
                if len(dep_t_large) > 0:
                    w_large  = temporal_weights(dep_t_large, w_t, model_name)
                    H1_large = entropy_bits(w_large)
                    H1_ext_chunk[ci] = h_bin(rho) + rho * H1_large + (1.0 - rho) * H1_small
                else:
                    H1_ext_chunk[ci] = H1_small
            else:
                H1_ext_chunk[ci] = H1_small

        for ci, ti_val in enumerate(chunk_ti):
            m1_results[model_name][ti_val]     = float(H1_chunk[ci])
            m1_ext_results[model_name][ti_val] = float(H1_ext_chunk[ci])

    if (chunk_start // CHUNK) % 20 == 0:
        pct = 100.0 * min(chunk_start + CHUNK, n_total) / n_total
        latest_h1     = m1_results["reuse_mix"].get(chunk_ti[-1], math.nan)
        latest_h1_ext = m1_ext_results["reuse_mix"].get(chunk_ti[-1], math.nan)
        print(f"  {pct:5.1f}%  H1_reuse={latest_h1:.3f}  H1_ext={latest_h1_ext:.3f}")

print("  Done.")

# ── 4b. Compute M4 (address-reuse graph conditioning, H1 heuristic) ──────────
# M4 assigns weight γ₁ ≫ 1 to deposits from the same address as the withdrawal.
# In the hard-conditioning limit (γ₁ → ∞):
#   H₄ = log₂(m)  where m = number of same-address amount-compatible deposits in
#                       D≤(w) that precede w in time.
# If m=0 (no same-address deposit found in D≤(w)): fall back to H₃ (M3/knapsack).
# If m=1: H₄=0 (unambiguous identification from address alone).
# For withdrawals without address-reuse tag: H₄ = H₃.
print("\nComputing M4 (H1 address-reuse graph conditioning) …")

# Build address → sorted list of deposit indices (into the time-sorted shield array)
from collections import defaultdict
addr_to_dep_idxs: dict[str, list[int]] = defaultdict(list)
for i, addr in enumerate(dep_addrs):
    addr_to_dep_idxs[addr].append(i)
# Each list is already in time order (we sorted dep_addrs by dep_times above)

# Load M3 (H_knap_item) for the all-horizon witness rows as the fallback
h3_by_ti_all: dict[int, float] = {}
for ti_val, w_row in witness_data.get("all", {}).items():
    try:
        h3_by_ti_all[ti_val] = float(w_row["H_knap_item"])
    except (ValueError, KeyError):
        h3_by_ti_all[ti_val] = math.nan

n_reuse_found    = 0
n_reuse_fallback = 0
m4_results: dict[int, float] = {}   # ti → H4
m4_m_counts: dict[int, int]  = {}   # ti → m (same-address deposit count)

for ti_val in ti_list:
    h3 = h3_by_ti_all.get(ti_val, math.nan)

    if not un_reuse[ti_val]:
        # No address evidence → M4 = M3
        m4_results[ti_val]  = h3
        m4_m_counts[ti_val] = 0
        continue

    addr   = un_addrs[ti_val]
    w_note = with_note_wei_f64_by_ti[ti_val]
    w_time_ns = int(un_time_ns_arr[ti_val])

    # Count deposits from same address with note_wei ≤ w_note AND time before w
    candidate_idxs = addr_to_dep_idxs.get(addr, [])
    m = 0
    for i in candidate_idxs:
        # dep_times is in seconds; w_time_ns is nanoseconds; convert
        if int(dep_times[i] * 1e9) < w_time_ns and dep_note_wei_f64[i] <= w_note:
            m += int(dep_agg_sizes[i])

    m4_m_counts[ti_val] = m
    if m >= 1:
        m4_results[ti_val] = math.log2(m) if m > 1 else 0.0
        n_reuse_found += 1
    else:
        # Same-address deposit not in D≤(w) → fallback to M3
        m4_results[ti_val] = h3
        n_reuse_fallback += 1

n_reuse_total = sum(1 for ti in ti_list if un_reuse[ti])
n_h4_zero     = sum(1 for ti in ti_list if m4_results.get(ti, math.nan) == 0.0
                    and un_reuse[ti])
print(f"  {n_reuse_total:,} reuse withdrawals in ti_list")
print(f"    → {n_reuse_found:,} with same-address D≤(w) deposit (H4=log2(m))")
print(f"    → {n_h4_zero:,} with m=1  (H4=0, singleton model support)")
print(f"    → {n_reuse_fallback:,} fallback to H3 (no D≤(w) match)")

# ── 5. Assemble per-withdrawal output ────────────────────────────────────────
print("\nAssembling per-withdrawal results …")
per_with_rows = []
for T in HORIZONS_DAYS:
    label = HORIZON_LABELS[T]
    wd = witness_data.get(label, {})
    for ti_val in sorted(wd.keys()):
        w_row = wd[ti_val]
        H0 = float(w_row["H_naive_item"])
        H2 = float(w_row["H_knap_item"])
        sat = w_row["saturated"] == "True"
        skp = w_row["skipped"] == "True"
        row = {
            "horizon": label,
            "ti": ti_val,
            "H0": round(H0, 6),
            "H2_knap": round(H2, 6),
            "saturated": int(sat),
            "skipped": int(skp),
        }
        for model_name in TIME_MODELS:
            h1     = m1_results[model_name].get(ti_val, math.nan)
            h1_ext = m1_ext_results[model_name].get(ti_val, math.nan)
            row[f"H1_{model_name}"]     = round(h1,     6) if math.isfinite(h1)     else ""
            row[f"H1_ext_{model_name}"] = round(h1_ext, 6) if math.isfinite(h1_ext) else ""
        row["rho_rem"]  = round(rho_rem_cache.get(ti_val, 0.0), 4)
        # M4 columns (all-horizon; for horizon-specific, M4 = M3 outside 'all')
        h4 = m4_results.get(ti_val, math.nan)
        row["H4_h1"]    = round(h4, 6) if math.isfinite(h4) else ""
        row["m_h1"]     = m4_m_counts.get(ti_val, 0)
        row["tag_reuse"]= int(un_reuse[ti_val])
        per_with_rows.append(row)

write_csv(OUT / "rg_entropy_per_withdrawal.csv", per_with_rows)
print(f"  Written {len(per_with_rows):,} rows to rg_entropy_per_withdrawal.csv")

# ── 6. Summary statistics ─────────────────────────────────────────────────────
print("\nSummary statistics …")
summary_rows = []
for T in HORIZONS_DAYS:
    label = HORIZON_LABELS[T]
    wd = witness_data.get(label, {})
    if not wd:
        continue
    H0_vals  = [float(r["H_naive_item"]) for r in wd.values() if r["skipped"] != "True"]
    H2_vals  = [float(r["H_knap_item"])  for r in wd.values() if r["skipped"] != "True"]
    n_sat    = sum(1 for r in wd.values() if r["saturated"] == "True")
    n_skip   = sum(1 for r in wd.values() if r["skipped"]   == "True")

    row_base = {
        "horizon": label,
        "n_total": len(wd),
        "n_saturated": n_sat,
        "n_skipped": n_skip,
    }
    for m, vals in [("H0", H0_vals), ("H2_knap", H2_vals)]:
        q = quantiles(vals)
        row_base[f"{m}_mean"]   = round(float(np.mean(vals)) if vals else math.nan, 4)
        for k, v in q.items():
            row_base[f"{m}_{k}"] = round(v, 4)

    for model_name in TIME_MODELS:
        h1_vals     = [m1_results[model_name][ti]     for ti in wd
                       if math.isfinite(m1_results[model_name].get(ti, math.nan))]
        h1_ext_vals = [m1_ext_results[model_name][ti] for ti in wd
                       if math.isfinite(m1_ext_results[model_name].get(ti, math.nan))]
        for tag, vals in [("H1", h1_vals), ("H1_ext", h1_ext_vals)]:
            q = quantiles(vals)
            row_base[f"{tag}_{model_name}_mean"] = round(float(np.mean(vals)) if vals else math.nan, 4)
            for k, v in q.items():
                row_base[f"{tag}_{model_name}_{k}"] = round(v, 4)

    # M4 summary (only meaningful at 'all' horizon; for other horizons copy M3)
    h4_all_vals = [m4_results.get(ti, math.nan) for ti in wd if math.isfinite(m4_results.get(ti, math.nan))]
    h4_reuse_vals = [m4_results.get(ti, math.nan) for ti in wd
                     if un_reuse[ti] and math.isfinite(m4_results.get(ti, math.nan))]
    q_h4 = quantiles(h4_all_vals)
    row_base["H4_h1_mean"] = round(float(np.mean(h4_all_vals)) if h4_all_vals else math.nan, 4)
    for k, v in q_h4.items():
        row_base[f"H4_h1_{k}"] = round(v, 4)
    row_base["H4_h1_reuse_p50"] = round(float(np.median(h4_reuse_vals)) if h4_reuse_vals else math.nan, 4)
    row_base["n_reuse"] = sum(1 for ti in wd if un_reuse[ti])
    row_base["n_h4_zero"] = sum(1 for ti in wd if m4_results.get(ti, math.nan) == 0.0 and un_reuse[ti])

    summary_rows.append(row_base)
    print(f"  [{label:6s}]  H0={row_base.get('H0_p50','?'):.3f}  "
          f"H1_reuse={row_base.get('H1_reuse_mix_p50','?'):.3f}  "
          f"H1_ext={row_base.get('H1_ext_reuse_mix_p50','?'):.3f}  "
          f"H2={row_base.get('H2_knap_p50','?'):.3f}  "
          f"H4={row_base.get('H4_h1_p50','?'):.3f}  "
          f"(n={len(wd)}, sat={n_sat}, skip={n_skip})")

write_csv(OUT / "rg_entropy_summary.csv", summary_rows)
rg_summary_by_horizon = {r["horizon"]: r for r in summary_rows}

# ── 6b. Auto-generate LaTeX table (tab:entropy_model_numeric) ────────────────
# Values from the 'all' horizon (widest window / all-time M1, M2) and
# 180d horizon for M3 (knapsack witness).  M4 from all-horizon computation.
print("\nGenerating LaTeX entropy table …")

def _f(x, decimals=2):
    """Format float for table; NaN → '---'."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    return f"{x:.{decimals}f}"

rg_all = rg_summary_by_horizon.get("all", {})
rg_180 = rg_summary_by_horizon.get("180d", {})

# Load PP summary (entropy_vs_T.csv written by estimate_pp_entropy.py)
pp_all_row: dict = {}
pp_180_row: dict = {}
if (OUT / "entropy_vs_T.csv").exists():
    for r in csv.DictReader(open(OUT / "entropy_vs_T.csv")):
        t = int(r.get("T_days", 0))
        if t == 180:
            pp_180_row = r
        if t >= 9999:   # sentinel for all-time in PP
            pp_all_row = r
    if not pp_all_row and pp_180_row:
        pp_all_row = pp_180_row  # PP all-time not separately computed

def _pp(key, row=pp_all_row):
    v = row.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return math.nan

# Paper numbers (medians):
rg_M1   = float(rg_all.get("H0_p50",         math.nan))   # uniform baseline
rg_M2_r = float(rg_all.get("H1_ext_reuse_mix_p50", math.nan))  # temporal, reuse-mix (M2 canonical)
rg_M2_f = float(rg_all.get("H1_ext_fifo_exp_p50",  math.nan))
rg_M2_l = float(rg_all.get("H1_ext_little_exp_p50",math.nan))
rg_M3   = float(rg_180.get("H2_knap_p50",    math.nan))   # knapsack @ 180d
rg_M4_all   = float(rg_all.get("H4_h1_p50",  math.nan))   # M4 all targets
rg_M4_reuse = float(rg_all.get("H4_h1_reuse_p50", math.nan))
rg_n_reuse  = int(rg_all.get("n_reuse",  0))
rg_n_total  = int(rg_all.get("n_total",  0))
rg_n_h4zero = int(rg_all.get("n_h4_zero", 0))

pp_M1   = _pp("H0", pp_all_row) if pp_all_row else math.nan
pp_M2_r = _pp("H1_reuse_mix", pp_180_row) if pp_180_row else math.nan
pp_M2_f = _pp("H1_fifo_exp",  pp_180_row) if pp_180_row else math.nan
pp_M2_l = _pp("H1_little_exp",pp_180_row) if pp_180_row else math.nan
pp_M3   = _pp("H2_reuse_mix", pp_180_row) if pp_180_row else math.nan

# ── PP M4: read from per_withdrawal.csv (H_direct_strict column) ─────────────
pp_per_with_path = PAPER / "analysis" / "pp_origin_entropy" / "per_withdrawal.csv"
pp_M4_all = math.nan
pp_M4_reuse = math.nan
pp_n_reuse = 0
pp_n_total_m4 = 0
pp_n_h4zero = 0
pp_reuse_pct = math.nan

if pp_per_with_path.exists():
    pp_pw = list(csv.DictReader(open(pp_per_with_path)))
    pp_canon = [r for r in pp_pw
                if r["model"] == "reuse_mix" and r["scenario"] == "exact_rows_collapsed"]
    pp_n_total_m4 = len(pp_canon)

    def _fpp(v):
        try: x = float(v); return x if math.isfinite(x) else math.nan
        except: return math.nan

    pp_n_dir  = np.array([int(r["n_direct_origins"]) for r in pp_canon])
    pp_H4s    = np.array([_fpp(r["H_direct_strict"]) for r in pp_canon])
    pp_H3r    = np.array([_fpp(r["H_knapsack_rho50"]) for r in pp_canon])

    # Fallback: no-reuse and abstained → M3
    no_reuse_pp = pp_n_dir == 0
    abstained_pp = np.array([int(r["abstained"]) for r in pp_canon], dtype=bool)
    pp_H4_corr = pp_H4s.copy()
    pp_H4_corr[no_reuse_pp]  = pp_H3r[no_reuse_pp]
    pp_H4_corr[abstained_pp] = pp_H3r[abstained_pp]

    pp_n_reuse   = int((pp_n_dir > 0).sum())
    pp_n_h4zero  = int((pp_H4_corr == 0).sum())
    pp_reuse_pct = 100.0 * pp_n_reuse / pp_n_total_m4
    h4zero_pp_pct = 100.0 * pp_n_h4zero / pp_n_total_m4

    finite_pp = pp_H4_corr[np.isfinite(pp_H4_corr)]
    pp_M4_all   = float(np.median(finite_pp)) if len(finite_pp) > 0 else math.nan
    reuse_finite = pp_H4_corr[pp_n_dir > 0]
    reuse_finite = reuse_finite[np.isfinite(reuse_finite)]
    pp_M4_reuse = float(np.median(reuse_finite)) if len(reuse_finite) > 0 else math.nan
    print(f"  PP M4: n_total={pp_n_total_m4}, n_reuse={pp_n_reuse} ({pp_reuse_pct:.1f}%), "
          f"H4=0: {pp_n_h4zero} ({h4zero_pp_pct:.1f}%)")
    print(f"  PP M4 median (all)={_f(pp_M4_all)}  (reuse-only)={_f(pp_M4_reuse)}")
else:
    print("  WARNING: PP per_withdrawal.csv not found, M4 skipped for PP")
    pp_reuse_pct = math.nan
    h4zero_pp_pct = math.nan

reuse_pct  = 100.0 * rg_n_reuse / rg_n_total if rg_n_total > 0 else math.nan
h4zero_pct = 100.0 * rg_n_h4zero / rg_n_total if rg_n_total > 0 else math.nan

table_tex = rf"""% Auto-generated by compute_rg_entropy_models.py — DO NOT EDIT MANUALLY.
% Source: rg_entropy_summary.csv + entropy_vs_T.csv
\begin{{table}}[t]
\centering\small
\begin{{tabular}}{{@{{}}lp{{5.5cm}}cc@{{}}}}
\toprule
Model & Configuration & Railgun & PP \\
      &               & ($n{{=}}38\,472$) & ($n{{=}}4\,206$) \\
\midrule
M1  & Uniform over amount-compatible pool
    & {_f(rg_M1)} & {_f(pp_M1)} \\
\midrule
M2  & Temporal prior, reuse-mix
    & {_f(rg_M2_r)} & {_f(pp_M2_r)} \\
M2  & Temporal prior, FIFO exp.
    & {_f(rg_M2_f)} & {_f(pp_M2_f)} \\
M2  & Temporal prior, Little exp.
    & {_f(rg_M2_l)} & {_f(pp_M2_l)} \\
\midrule
M3  & Knapsack witnesses, $T{{=}}180$d
    & {_f(rg_M3)} & {_f(pp_M3)} \\
\midrule
M4  & Address-selected proxy, all targets
    & {_f(rg_M4_all)} & {_f(pp_M4_all)} \\
M4  & Address-selected proxy, reuse targets only
    & {_f(rg_M4_reuse)} ({reuse_pct:.1f}\%) & {_f(pp_M4_reuse)} ({pp_reuse_pct:.1f}\%) \\
\bottomrule
\end{{tabular}}
\caption{{Archived median model statistics in bits; cohorts and estimands are
not aligned across rows or systems. M1 is the uniform amount-compatible
baseline. M2 uses alternative temporal scenarios and, for Railgun, a heuristic
larger-deposit mixture. Railgun M3 is capped witness-participation entropy
at 180 days, not a bound or an evaluation of the full M3 mixture (97\% cap
saturation). Railgun M4 uses an all-history fallback. PP M4 comes from a
separate exact-row-collapsed origin analysis with a different fallback from
the PP M3 summary. Address-selected uniform weights are not generally M3
conditioning. Zero model entropy does not validate identification.}}

\label{{tab:entropy_model_numeric}}
\end{{table}}
"""
(OUT / "entropy_model_table.tex").write_text(table_tex)
print(f"  Written entropy_model_table.tex")
print(f"  RG  M1={_f(rg_M1)}  M2={_f(rg_M2_r)}  M3={_f(rg_M3)}  M4(all)={_f(rg_M4_all)}  M4(reuse)={_f(rg_M4_reuse)}")
print(f"  PP  M1={_f(pp_M1)}  M2={_f(pp_M2_r)}  M3={_f(pp_M3)}  M4(all)={_f(pp_M4_all)}  M4(reuse)={_f(pp_M4_reuse)}")

# ── 7. Save model parameters ──────────────────────────────────────────────────
params_out = {
    "script": "compute_rg_entropy_models.py",
    "dataset": str(RG_DATA),
    "n_shields": len(dep_times),
    "n_unshields": len(unshield_agg),
    "h1_pairs_n": len(h1_delays_days),
    "h1_pairs_median_days": float(np.median(h1_delays_days)),
    "model_params": model_params,
    "witness_step": WITNESS_STEP,
    "note_saturated": (
        "H_knap_item is a sample-dependent statistic when saturated=True "
        "(witness cap at 2000 reached). "
        f"{sum(1 for r in (list(witness_data.get('all',{}).values())) if r['saturated']=='True')} "
        "of all-horizon rows are saturated."
    ),
}
(OUT / "rg_m1_params.json").write_text(json.dumps(params_out, indent=2))

# ── 8. Combined RG + PP TikZ comparison figure ────────────────────────────────
print("\nGenerating combined RG + PP TikZ figure …")

# Load PP entropy-vs-T CSV for comparison
pp_csv = OUT / "entropy_vs_T.csv"
pp_data = {}
if pp_csv.exists():
    for row in csv.DictReader(open(pp_csv)):
        pp_data[int(row["T_days"])] = row

# Build combined CSV for pgfplots
rg_summary_by_horizon = {r["horizon"]: r for r in summary_rows}
T_labels_plot = ["3d", "7d", "30d", "180d", "all"]
T_days_plot   = [3, 7, 30, 180, 1400]  # "all" ≈ 3.8 years ≈ 1387 days

combined_rows = []
for T_day, label in zip(T_days_plot, T_labels_plot):
    rg = rg_summary_by_horizon.get(label, {})
    pp = pp_data.get(T_day if T_day < 1000 else 180, {})  # PP only up to 180d
    combined_rows.append({
        "T_plot": T_day,
        "label": label,
        "rg_H0_p50":   rg.get("H0_p50", ""),
        "rg_H1_reuse_p50": rg.get("H1_reuse_mix_p50", ""),
        "rg_H1_fifo_p50":  rg.get("H1_fifo_exp_p50", ""),
        "rg_H2_p50":   rg.get("H2_knap_p50", ""),
        "pp_H0":       pp.get("H0", ""),
        "pp_H1_reuse": pp.get("H1_reuse_mix", ""),
        "pp_H2_reuse": pp.get("H2_reuse_mix", ""),
    })
write_csv(OUT / "rg_pp_entropy_comparison.csv", combined_rows)

tikz = r"""% Auto-generated by compute_rg_entropy_models.py
% Combined Railgun + PP entropy model comparison
% Note: Railgun H2 (H_knap_item) is a sample-dependent statistic when saturated=True.
%       RG x-axis uses T_plot with "all" at 1400 d (log scale).
\begin{tikzpicture}
\begin{axis}[
  figurestyle,
  xlabel={Search horizon $T$ [days]},
  ylabel={Median entropy [bits]},
  xmode=log,
  xtick={3,7,30,180,1400},
  xticklabels={3,7,30,180,all},
  xticklabel style={font=\scriptsize},
  ymin=0, ymax=16,
  xmajorgrids, ymajorgrids,
  legend style={at={(0.97,0.97)}, anchor=north east, font=\scriptsize,
                fill=white, fill opacity=0.85, draw=black!40,
                cells={align=left}},
  legend cell align=left,
]

%% ── Railgun ────────────────────────────────────────────────────────────────
\addplot[thick, cblue, dotted]
  table[x=T_plot, y=rg_H0_p50, col sep=comma]
  {Figures/entropy_models/rg_pp_entropy_comparison.csv};
\addlegendentry{RG M1 (uniform)}

\addplot[thick, cblue]
  table[x=T_plot, y=rg_H1_reuse_p50, col sep=comma]
  {Figures/entropy_models/rg_pp_entropy_comparison.csv};
\addlegendentry{RG M2 (temporal, reuse mix)}

\addplot[thick, cblue, dashed, mark=*, mark size=2pt]
  table[x=T_plot, y=rg_H2_p50, col sep=comma]
  {Figures/entropy_models/rg_pp_entropy_comparison.csv};
\addlegendentry{RG M3 (knapsack, capped proxy)}

%% ── Privacy Pools ──────────────────────────────────────────────────────────
\addplot[thick, cwithdraw, dotted]
  table[x=T_plot, y=pp_H0, col sep=comma]
  {Figures/entropy_models/rg_pp_entropy_comparison.csv};
\addlegendentry{PP M1 (uniform)}

\addplot[thick, cwithdraw]
  table[x=T_plot, y=pp_H1_reuse, col sep=comma]
  {Figures/entropy_models/rg_pp_entropy_comparison.csv};
\addlegendentry{PP M2 (temporal, reuse mix)}

\addplot[thick, cwithdraw, dashed, mark=square*, mark size=2pt]
  table[x=T_plot, y=pp_H2_reuse, col sep=comma]
  {Figures/entropy_models/rg_pp_entropy_comparison.csv};
\addlegendentry{PP M3 (knapsack)}

\end{axis}
\end{tikzpicture}
"""
(OUT / "rg_pp_entropy_comparison_tikz.tex").write_text(tikz)

print("\nAll outputs written to", OUT)
print("Done.")
