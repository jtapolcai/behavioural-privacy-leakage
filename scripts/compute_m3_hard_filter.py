#!/usr/bin/env python3
"""
compute_m3_hard_filter.py
=========================
New M3 model definition:

  1. AR / GR (hard filter): if the withdrawal has address-reuse (AR) or
     gas-payer-reuse (GR) evidence, the anonymity set is reduced to exactly
     those matched deposits.  Uniform distribution over the hard-filtered set:

       H3 = log2(|AR ∪ GR matched deposits|)

  2. PT (soft boost): if no AR/GR evidence but a public-transfer link exists,
     multiply that deposit's M2 weight by α_p and renormalise.  The PT-boosted
     entropy is computed analytically given α_p, n_PT (# PT-matched deposits)
     and n_total (total amount-compatible candidates) and their aggregate M2
     probability mass.

  3. Fallback: if no AR/GR/PT evidence → H3 = H2_canonical.

Data availability
-----------------
Current exports contain:
  - m_h1        : # AR-matched deposits (within horizon window)     [AVAILABLE]
  - tag_reuse   : broader AR flag (may include multiple horizons)   [AVAILABLE]
  - n_pt        : # PT-matched deposits                             [MISSING - not in export]
  - n_gr        : # GR-matched deposits                             [MISSING - not in export]
  - H2_canonical: M2 entropy for fallback                           [AVAILABLE]

Until PT and GR per-withdrawal counts are exported, only the AR hard filter
is computed.  PT and GR columns are accepted as optional inputs (defaulting
to zero) so the script can be extended without code changes.

Input columns (rg_m2_canonical.csv joined with rg_entropy_per_withdrawal.csv):
  horizon, ti (=agg_id), H2_canonical, n_le, m_h1, [n_pt=0], [n_gr=0]

Output:
  Figures/entropy_models/rg_m3_hard_filter.csv
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

PAPER = Path(__file__).resolve().parents[1]

# ── Parameters ──────────────────────────────────────────────────────────────
ALPHA_P = 10.0          # PT soft-boost factor (modelling parameter)
HORIZON = "180d"        # which horizon row to use from rg_entropy_per_withdrawal

# ── Input files ─────────────────────────────────────────────────────────────
RG_M2   = PAPER / "Figures/entropy_models/rg_m2_canonical.csv"
RG_EV   = PAPER / "Figures/entropy_models/rg_entropy_per_withdrawal.csv"


# ── Core computation ─────────────────────────────────────────────────────────

def h3_ar_gr_hard_filter(n_ar_gr: int) -> float:
    """H3 when AR or GR evidence is present: uniform over the hard-filtered set."""
    if n_ar_gr <= 0:
        return math.nan
    if n_ar_gr == 1:
        return 0.0          # certain identification
    return math.log2(n_ar_gr)


def h3_pt_soft_boost(H2: float, n_pt: int, n_total: int, alpha_p: float) -> float:
    """
    H3 for PT-only evidence.

    Approximation: treat M2 as uniform over n_total candidates (conservative),
    then boost n_pt of them by alpha_p.

    Weight of a PT candidate    : alpha_p
    Weight of a non-PT candidate: 1
    Normaliser Z = n_pt * alpha_p + (n_total - n_pt)

    This is an approximation; the exact computation requires per-deposit M2 weights.
    """
    if n_pt <= 0 or n_total <= 0:
        return H2     # no PT evidence → fallback
    Z = n_pt * alpha_p + (n_total - n_pt)
    p_pt    = alpha_p / Z
    p_rest  = 1.0 / Z if n_total > n_pt else 0.0
    H = 0.0
    if p_pt > 0 and n_pt > 0:
        H -= n_pt * p_pt * math.log2(p_pt)
    if p_rest > 0 and (n_total - n_pt) > 0:
        H -= (n_total - n_pt) * p_rest * math.log2(p_rest)
    return H


def compute_m3(row: pd.Series, alpha_p: float = ALPHA_P) -> tuple[float, str]:
    """
    Returns (H3, case) where case ∈ {'ar_gr', 'pt', 'fallback'}.
    """
    n_ar  = int(row.get("m_h1", 0) or 0)
    n_gr  = int(row.get("n_gr",  0) or 0)
    n_pt  = int(row.get("n_pt",  0) or 0)
    n_ar_gr = n_ar + n_gr   # union (upper bound; exact union needs deposit-level data)

    H2 = float(row["H2_canonical"])
    n_total = int(row.get("n_le", 0) or 0)

    if n_ar_gr > 0:
        return h3_ar_gr_hard_filter(n_ar_gr), "ar_gr"
    elif n_pt > 0:
        return h3_pt_soft_boost(H2, n_pt, n_total, alpha_p), "pt"
    else:
        return H2, "fallback"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load M2 canonical results (one row per withdrawal, all horizons merged)
    m2 = pd.read_csv(RG_M2)                  # columns: agg_id, H2_canonical, rho_rem, n_le, n_gt
    m2 = m2.rename(columns={"agg_id": "ti"})

    # Load evidence file, filter to chosen horizon
    ev = pd.read_csv(RG_EV)
    ev = ev[ev["horizon"] == HORIZON][["ti", "m_h1", "tag_reuse"]].copy()

    # Add missing PT / GR columns (not yet in export)
    ev["n_pt"] = 0
    ev["n_gr"] = 0

    df = m2.merge(ev, on="ti", how="left")
    df["m_h1"]     = df["m_h1"].fillna(0).astype(int)
    df["tag_reuse"] = df["tag_reuse"].fillna(0).astype(int)
    df["n_pt"]      = df["n_pt"].fillna(0).astype(int)
    df["n_gr"]      = df["n_gr"].fillna(0).astype(int)

    results = df.apply(lambda r: pd.Series(compute_m3(r), index=["H3", "case"]), axis=1)
    df = pd.concat([df, results], axis=1)

    # Summary
    H3 = df["H3"].dropna()
    print(f"Railgun M3 hard-filter (horizon={HORIZON}, n={len(H3):,})")
    print(f"  Median : {H3.median():.4f} bits")
    print(f"  Mean   : {H3.mean():.4f} bits")
    print(f"  p10/p90: {H3.quantile(.1):.4f} / {H3.quantile(.9):.4f} bits")
    print()
    print("Case breakdown:")
    print(df["case"].value_counts().to_string())
    print()
    print("Note: n_pt=0 and n_gr=0 because PT/GR per-withdrawal counts")
    print("      are not yet in the export. Add those columns to unlock.")

    out = PAPER / "Figures/entropy_models/rg_m3_hard_filter.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["ti", "H2_canonical", "H3", "case", "m_h1", "n_pt", "n_gr", "n_le"]].to_csv(
        out, index=False, float_format="%.6f"
    )
    print(f"\nWritten → {out}")


if __name__ == "__main__":
    main()
