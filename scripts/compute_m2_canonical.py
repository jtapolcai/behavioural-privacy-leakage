#!/usr/bin/env python3
"""
compute_m2_canonical.py
=======================
Computes the canonical M2 temporal origin entropy for Railgun using the
calibrated three-component exponential mixture parameters from §3.2.

Model definition (Definition 3 / eq:h2 in the paper):

  f_θ(Δt) = Σ_k (π_k^corr / μ_k^true) · exp(−Δt / μ_k^true)

  p_2(d | w; θ) = mixture of:
    (1 − ρ_rem) · f_θ(Δt_{d,w}) / Z^≤   for d ∈ D^≤(w)
    ρ_rem       · f_θ(Δt_{d,w}) / Z^>   for d ∈ D^>(w)

  H_2(w) = −Σ_d p_2(d|w) log_2 p_2(d|w)

Canonical Railgun parameters (§3.2, corrected for FIFO and selection bias):
  pi_corr = [0.085, 0.461, 0.454]   # sprinter / regular / hodler
  mu_days = [0.09,  8.79,  403.0]   # calibrated μ_k^true (days)

Outputs:
  Prints median, mean, and percentile summary.
  Writes per-withdrawal H2 to Figures/entropy_models/rg_m2_canonical.csv
"""

from __future__ import annotations
import csv
import math
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

import numpy as np

# ── Canonical model parameters (§3.2) ─────────────────────────────────────
PI_CORR = np.array([0.085, 0.461, 0.454])  # corrected mixing weights
MU_DAYS = np.array([0.09,  8.79,  403.0])  # calibrated μ_k^true (days)

SECS_PER_DAY = 86_400.0

# ── Paths ─────────────────────────────────────────────────────────────────
PAPER    = Path(__file__).resolve().parents[1]
RG_DATA  = _RG_DATA.parent  # via _paths
AGG      = RG_DATA / "data" / "aggregated"
OUT_CSV  = PAPER / "Figures" / "entropy_models" / "rg_m2_canonical.csv"


def mixture_density(dt_days: np.ndarray) -> np.ndarray:
    """f_θ(Δt) = Σ_k (π_k / μ_k) exp(−Δt / μ_k)"""
    out = np.zeros(len(dt_days), dtype=np.float64)
    for pi, mu in zip(PI_CORR, MU_DAYS):
        out += (pi / mu) * np.exp(-dt_days / mu)
    return out


def entropy_bits(weights: np.ndarray) -> float:
    s = weights.sum()
    if s <= 0:
        return math.nan
    p = weights[weights > 0] / s
    return float(-np.dot(p, np.log2(p)))


def main() -> None:
    # ── Load deposits ──────────────────────────────────────────────────────
    print("Loading deposits …")
    shields = list(csv.DictReader(open(AGG / "eth_shield_aggregated.csv")))
    dep_t_s  = np.array([int(r["last_time_ns"]) / 1e9 for r in shields])
    dep_wei  = np.array([int(r["note_wei"])          for r in shields], dtype=np.float64)
    sort_d   = np.argsort(dep_t_s)
    dep_t_s  = dep_t_s[sort_d];  dep_wei = dep_wei[sort_d]
    dep_eth  = dep_wei / 1e18
    cum_shield_eth = np.cumsum(dep_eth)
    print(f"  {len(dep_t_s):,} deposits")

    # ── Load withdrawals ───────────────────────────────────────────────────
    print("Loading withdrawals …")
    unshields = list(csv.DictReader(open(AGG / "eth_unshields_aggregated.csv")))
    wit_t_s   = np.array([int(r["first_time_ns"]) / 1e9 for r in unshields])
    wit_wei   = np.array([int(r["note_wei"])            for r in unshields], dtype=np.float64)
    sort_u    = np.argsort(wit_t_s)
    wit_t_s_sorted = wit_t_s[sort_u];  wit_wei_sorted = wit_wei[sort_u]
    cum_unshield_eth = np.cumsum(wit_wei_sorted / 1e18)
    print(f"  {len(wit_t_s):,} withdrawals")

    # ── Per-withdrawal M2 entropy ─────────────────────────────────────────
    print("Computing M2 entropy (all-time, calibrated parameters) …")
    results = []
    n = len(wit_t_s)

    for idx in range(n):
        if idx % 5000 == 0:
            print(f"  {idx:,}/{n:,}")
        tw = wit_t_s[idx]
        ww = wit_wei[idx]

        # All deposits before this withdrawal
        n_dep_before = int(np.searchsorted(dep_t_s, tw, side="left"))
        if n_dep_before == 0:
            continue

        dt_days = (tw - dep_t_s[:n_dep_before]) / SECS_PER_DAY
        dw      = dep_wei[:n_dep_before]

        mask_le = dw <= ww
        mask_gt = ~mask_le

        # Retained-balance mixing weight ρ_rem
        sigma_in  = float(cum_shield_eth[n_dep_before - 1])
        n_un_before = int(np.searchsorted(wit_t_s_sorted, tw, side="left"))
        sigma_out = float(cum_unshield_eth[n_un_before - 1]) if n_un_before > 0 else 0.0
        rho = max(0.0, min(1.0, (sigma_in - sigma_out) / sigma_in)) if sigma_in > 0 else 0.0

        # Component weights
        all_probs = []

        w_le = mixture_density(dt_days[mask_le])
        z_le = w_le.sum()
        if z_le > 0 and mask_le.any():
            all_probs.append((1.0 - rho) * w_le / z_le)

        w_gt = mixture_density(dt_days[mask_gt])
        z_gt = w_gt.sum()
        if z_gt > 0 and mask_gt.any():
            all_probs.append(rho * w_gt / z_gt)

        if not all_probs:
            continue

        p_all = np.concatenate(all_probs)
        p_all /= p_all.sum()  # normalise
        h2 = entropy_bits(p_all)
        results.append({"agg_id": idx, "H2_canonical": h2, "rho_rem": rho,
                        "n_le": int(mask_le.sum()), "n_gt": int(mask_gt.sum())})

    H2 = np.array([r["H2_canonical"] for r in results])
    H2 = H2[np.isfinite(H2)]

    print(f"\nRailgun M2 canonical (n={len(H2):,}):")
    print(f"  Median : {np.median(H2):.4f} bits")
    print(f"  Mean   : {np.mean(H2):.4f} bits")
    print(f"  p10    : {np.percentile(H2, 10):.4f} bits")
    print(f"  p25    : {np.percentile(H2, 25):.4f} bits")
    print(f"  p75    : {np.percentile(H2, 75):.4f} bits")
    print(f"  p90    : {np.percentile(H2, 90):.4f} bits")

    # Write results
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agg_id", "H2_canonical", "rho_rem",
                                          "n_le", "n_gt"])
        w.writeheader()
        w.writerows(results)
    print(f"\nPer-withdrawal results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
