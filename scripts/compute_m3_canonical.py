#!/usr/bin/env python3
"""
compute_m3_canonical.py
=======================
Computes the canonical M3 origin entropy for Railgun and writes the
unified per-withdrawal CSV used by compute_table2.py.

Evidence signals (on top of M2 temporal prior):
  AR  – address reuse:   deposit address == withdrawal address (inline)
  GR  – gas-payer reuse: deposit address == relayer address (from rg_pt_gr_counts.csv)
  PT  – public transfer: on-chain link deposit → withdrawal (from rg_pt_gr_counts.csv)

Output: Figures/entropy_models/rg_m3_canonical.csv
Columns (unified format, identical to pp_m3_canonical.csv):
  withdrawal_id, n_ar, n_gr, n_pt,
  H2, H3_ar, H3_ar_gr, H3_ar_gr_pt10, H3_ar_gr_ptinf
"""

from __future__ import annotations
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, FIGURES as _FIGURES
from _scenario_params import RG_PI, RG_MU
from models import (DepositData, WithdrawalData, Evidence,
                    compute_m2, compute_m3, mean_finite)

ALPHA_P = 10.0
K_REL   = 10     # relayer exclusion threshold

PAPER   = Path(__file__).resolve().parents[1]
RG_DATA = _RG_DATA.parent
AGG     = RG_DATA / "data" / "aggregated"
PTGR    = PAPER / "Figures" / "entropy_models" / "rg_pt_gr_counts.csv"
OUT_CSV = PAPER / "Figures" / "entropy_models" / "rg_m3_canonical.csv"


def load_rg_deposits() -> DepositData:
    print("Loading RG deposits …")
    rows = list(csv.DictReader(open(AGG / "eth_shield_aggregated.csv")))
    agg_id  = np.array([int(r["agg_id"])              for r in rows])
    t_s     = np.array([int(r["last_time_ns"])  / 1e9 for r in rows])
    addr    = np.array([r["address"].lower().strip()   for r in rows])
    amt_eth = np.array([int(r["note_wei"])      / 1e18 for r in rows])
    print(f"  {len(rows):,} deposits")
    return DepositData.from_arrays(agg_id, t_s, addr, amt_eth)


def load_rg_withdrawals() -> WithdrawalData:
    print("Loading RG withdrawals …")
    rows = list(csv.DictReader(open(AGG / "eth_unshields_aggregated.csv")))
    agg_id  = np.array([int(r["agg_id"])               for r in rows])
    t_s     = np.array([int(r["first_time_ns"]) / 1e9  for r in rows])
    addr    = np.array([r["address"].lower().strip()    for r in rows])
    amt_eth = np.array([int(r["note_wei"])       / 1e18 for r in rows])
    print(f"  {len(rows):,} withdrawals")
    return WithdrawalData.from_arrays(agg_id, t_s, addr, amt_eth)


def load_rg_evidence() -> Evidence:
    """Load GR and PT evidence from rg_pt_gr_counts.csv."""
    if not PTGR.exists():
        print(f"  WARNING: {PTGR.name} not found — GR and PT evidence will be empty.")
        return {}
    evidence: Evidence = {}
    for row in csv.DictReader(open(PTGR)):
        wid = int(row["withdrawal_agg_id"])
        def _parse(col: str):
            raw = str(row.get(col, "")) if row.get(col) else ""
            return {int(x) for x in raw.split("|") if x and x != "nan"}
        evidence[wid] = {
            "gr": _parse("deposit_agg_ids_gr"),
            "pt": _parse("deposit_agg_ids_pt"),
        }
    n_gr = sum(1 for v in evidence.values() if v["gr"])
    n_pt = sum(1 for v in evidence.values() if v["pt"])
    print(f"  PT/GR evidence: {len(evidence):,} withdrawals, "
          f"{n_gr:,} with GR, {n_pt:,} with PT")
    return evidence


def main() -> None:
    deps = load_rg_deposits()
    wits = load_rg_withdrawals()
    evidence = load_rg_evidence()

    print("Computing M2 (calibrated 3-component mixture) …")
    _, m2_dict = compute_m2(deps, wits, RG_PI, RG_MU, use_rho_rem=True)
    print(f"  M2 mean: {mean_finite(list(m2_dict.values())):.4f} bits")

    print("Computing M3 (AR / GR / PT variants) …")
    m3_df = compute_m3(deps, wits, m2_dict, evidence,
                       RG_PI, RG_MU, alpha_p=ALPHA_P, use_rho_rem=True)

    n = len(m3_df)
    for col in ["H3_ar", "H3_ar_gr", "H3_ar_gr_pt10", "H3_ar_gr_ptinf"]:
        print(f"  {col:<22}: {mean_finite(m3_df[col]):.4f}  "
              f"(AR cov {(m3_df['n_ar']>0).mean()*100:.1f}%)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    m3_df.to_csv(OUT_CSV, index=False)
    print(f"\nWritten {n:,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
