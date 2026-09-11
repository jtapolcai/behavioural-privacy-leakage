#!/usr/bin/env python3
"""
compute_m3_stochastic.py
========================
Computes the stochastic M3 origin entropy for Railgun (Definition 4 / eq:m3 in the paper).

Model (eq:m3):
    p_3(d_i | w; θ) = f_θ(Δt_i) · c_i  /  Σ_j f_θ(Δt_j) · c_j

where
  f_θ(Δt)  — calibrated holding-time density from §3.2 (three-component exponential mixture)
  c_i       — witness-participation count of deposit d_i in the knapsack enumeration

This softens the hard search window T: instead of excluding deposits beyond T, each
deposit is re-weighted by how probable its holding time is under the prior.

Data requirement
----------------
This script requires per-deposit participation counts c_i, i.e. for each withdrawal w
a file (or in-memory structure) mapping deposit index → c_i count.
The archived witness files (results/witness/witness_kto1_*.csv) contain only aggregated
per-withdrawal statistics (H_knap_item, item_n, …), NOT individual c_i values.

To run the full stochastic M3 computation the knapsack solver must be re-run with
item-level output. Until that export is available the archived H_knap_item (4.67 bits
at 180 days) remains the only available proxy; it equals H(c_i/Σc_j) without temporal
re-weighting.

Expected input CSV (per-withdrawal, per-deposit row):
  withdrawal_agg_id, deposit_agg_id, c_i, deposit_time_ns, withdrawal_time_ns

Usage (once the per-deposit export is available):
  python compute_m3_stochastic.py --input <per_deposit_counts.csv> [--output <out.csv>]
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Canonical model parameters (§3.2) ─────────────────────────────────────
PI_CORR = np.array([0.085, 0.461, 0.454])   # corrected mixing weights
MU_DAYS = np.array([0.09,  8.79,  403.0])   # calibrated μ_k^true (days)

SECS_PER_DAY = 86_400.0

PAPER = Path(__file__).resolve().parents[1]
OUT_CSV = PAPER / "Figures" / "entropy_models" / "rg_m3_stochastic.csv"


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


def compute_m3_for_withdrawal(
    dt_days: np.ndarray,   # Δt_i = (t_w - t_d_i) / SECS_PER_DAY, shape (N,)
    c_i: np.ndarray,       # witness participation counts, shape (N,); 0 for non-witnesses
) -> float:
    """
    p_3(d_i | w) ∝ f_θ(Δt_i) · c_i
    Returns H_3 in bits; nan if no witnesses.
    """
    w = mixture_density(dt_days) * c_i
    return entropy_bits(w)


def main(input_csv: Path) -> None:
    """Load per-deposit counts and compute stochastic M3 per withdrawal."""
    print(f"Loading per-deposit counts from {input_csv} …")

    # Group rows by withdrawal id
    by_withdrawal: dict[str, list[dict]] = defaultdict(list)
    with open(input_csv, newline="") as f:
        for row in csv.DictReader(f):
            by_withdrawal[row["withdrawal_agg_id"]].append(row)

    print(f"  {len(by_withdrawal):,} withdrawals")

    results = []
    for wid, rows in by_withdrawal.items():
        dt_s    = np.array([float(rows[0]["withdrawal_time_ns"]) / 1e9
                            - float(r["deposit_time_ns"]) / 1e9
                            for r in rows])
        dt_days = dt_s / SECS_PER_DAY
        c       = np.array([float(r["c_i"]) for r in rows])

        h3 = compute_m3_for_withdrawal(dt_days, c)
        results.append({"withdrawal_agg_id": wid, "H3_stochastic": h3,
                        "n_witnesses": int((c > 0).sum()), "n_total": len(rows)})

    H3 = np.array([r["H3_stochastic"] for r in results])
    H3 = H3[np.isfinite(H3)]

    print(f"\nRailgun M3 stochastic (n={len(H3):,}):")
    print(f"  Median : {np.median(H3):.4f} bits")
    print(f"  Mean   : {np.mean(H3):.4f} bits")
    print(f"  p10    : {np.percentile(H3, 10):.4f} bits")
    print(f"  p90    : {np.percentile(H3, 90):.4f} bits")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["withdrawal_agg_id", "H3_stochastic",
                                          "n_witnesses", "n_total"])
        w.writeheader()
        w.writerows(results)
    print(f"\nPer-withdrawal results written to {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic M3 computation.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Per-deposit participation CSV "
                             "(columns: withdrawal_agg_id, deposit_agg_id, c_i, "
                             "deposit_time_ns, withdrawal_time_ns)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found.\n\n"
            "The stochastic M3 requires per-deposit participation counts c_i that are "
            "not in the archived witness summary files. Re-run the knapsack solver with "
            "item-level output to produce this file.\n"
            "Until then, the archived H_knap_item (4.67 bits at 180 days, witness_kto1_180d_step0.001.csv) "
            "is the available proxy: it equals H(c_i/Σc_j) without the f_θ re-weighting."
        )

    main(args.input)
