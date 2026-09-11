#!/usr/bin/env python3
"""
compute_m3_canonical.py
=======================
Computes the canonical M3 origin entropy for Railgun.

M3 applies three evidence signals on top of M2:
  AR (address reuse)   — hard filter: anonymity set = AR-matched deposits
  GR (gas-payer reuse) — hard filter: anonymity set = GR-matched deposits
                          (relayers excluded: gas payers with >= K_REL tx)
  PT (public transfer) — soft boost: multiply M2 weight by alpha_p, renormalise

Priority: AR/GR hard filter > PT soft boost > M2 fallback.

Outputs:
  Figures/entropy_models/rg_m3_canonical.csv
  columns: agg_id, H2_canonical, H3, case, n_ar_gr, n_pt
"""

from __future__ import annotations
import csv
import math
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

import numpy as np
import pandas as pd

# ── Parameters ───────────────────────────────────────────────────────────────
PI_CORR   = np.array([0.085, 0.461, 0.454])   # declared scenario weights
MU_DAYS   = np.array([0.09,  8.79,  403.0])   # declared scenario scales (days)
ALPHA_P   = 10.0    # PT soft-boost factor
K_REL     = 10      # relayer exclusion threshold (gas payers with >= K_REL tx)

SECS_PER_DAY = 86_400.0

# ── Paths ────────────────────────────────────────────────────────────────────
PAPER   = Path(__file__).resolve().parents[1]
RG_DATA = _RG_DATA.parent  # via _paths
AGG     = RG_DATA / "data" / "aggregated"
OUT_CSV = PAPER / "Figures" / "entropy_models" / "rg_m3_canonical.csv"

# Evidence files (produced by export_pt_gr_counts.py)
PTGR_CSV = PAPER / "Figures" / "entropy_models" / "rg_pt_gr_counts.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def mixture_density(dt_days: np.ndarray) -> np.ndarray:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load deposits ─────────────────────────────────────────────────────
    print("Loading deposits …")
    shields = list(csv.DictReader(open(AGG / "eth_shield_aggregated.csv")))
    dep_agg_id = np.array([int(r["agg_id"])           for r in shields])
    dep_t_s    = np.array([int(r["last_time_ns"]) / 1e9 for r in shields])
    dep_wei    = np.array([int(r["note_wei"])           for r in shields], dtype=object)
    dep_addr   = np.array([r["address"].lower().strip() for r in shields])
    sort_d = np.argsort(dep_t_s)
    dep_agg_id = dep_agg_id[sort_d]
    dep_t_s    = dep_t_s[sort_d]
    dep_wei    = dep_wei[sort_d]
    dep_addr   = dep_addr[sort_d]
    print(f"  {len(dep_t_s):,} deposits")

    # ── Load withdrawals ───────────────────────────────────────────────────
    print("Loading withdrawals …")
    unshields = list(csv.DictReader(open(AGG / "eth_unshields_aggregated.csv")))
    wit_agg_id = np.array([int(r["agg_id"])            for r in unshields])
    wit_t_s    = np.array([int(r["first_time_ns"]) / 1e9 for r in unshields])
    wit_wei    = np.array([int(r["note_wei"])            for r in unshields], dtype=object)
    # Keep exact integers for amount eligibility; floating point only for flows.
    dep_eth    = np.asarray(dep_wei, dtype=float) / 1e18
    cum_shield_eth = np.cumsum(dep_eth)
    wit_eth_sorted = (np.asarray(wit_wei, dtype=float) / 1e18)[np.argsort(wit_t_s)]
    cum_unshield_eth = np.cumsum(wit_eth_sorted)
    wit_t_s_sorted = np.sort(wit_t_s)
    print(f"  {len(wit_t_s):,} withdrawals")

    # ── Load PT/GR evidence ─────────────────────────────────────────────
    pt_map: dict[int, set[int]] = {}
    gr_map: dict[int, set[int]] = {}
    if PTGR_CSV.exists():
        print("Loading PT/GR evidence …")
        ptgr = pd.read_csv(PTGR_CSV)
        for _, row in ptgr.iterrows():
            wid = int(row["withdrawal_agg_id"])
            raw_pt = str(row["deposit_agg_ids_pt"]) if pd.notna(row["deposit_agg_ids_pt"]) else ""
            raw_gr = str(row["deposit_agg_ids_gr"]) if pd.notna(row["deposit_agg_ids_gr"]) else ""
            pt_ids = {int(x) for x in raw_pt.split("|") if x and x != "nan"}
            gr_ids = {int(x) for x in raw_gr.split("|") if x and x != "nan"}
            if pt_ids: pt_map[wid] = pt_ids
            if gr_ids: gr_map[wid] = gr_ids
    else:
        import warnings
        warnings.warn(f"PT/GR evidence not found at {PTGR_CSV} — "
                      "running AR-only (GR and PT evidence unavailable). "
                      "Run export_pt_gr_counts.py to generate it.")

    # Build address → set of deposit agg_ids (for AR lookup)
    dep_addr_to_ids: dict[str, set[int]] = {}
    for aid, addr in zip(dep_agg_id, dep_addr):
        dep_addr_to_ids.setdefault(addr, set()).add(int(aid))

    # Load withdrawal addresses for AR matching
    wit_addr = np.array([r["address"].lower().strip() for r in unshields])

    # ── Per-withdrawal M3 entropy ──────────────────────────────────────────
    print("Computing M3 entropy …")
    results = []
    n = len(wit_t_s)

    for idx in range(n):
        if idx % 5000 == 0:
            print(f"  {idx:,}/{n:,}")

        tw  = wit_t_s[idx]
        ww  = wit_wei[idx]
        wid = int(wit_agg_id[idx])

        # All deposits strictly before this withdrawal
        n_dep_before = int(np.searchsorted(dep_t_s, tw, side="left"))
        if n_dep_before == 0:
            continue

        dt_days      = (tw - dep_t_s[:n_dep_before]) / SECS_PER_DAY
        dw           = dep_wei[:n_dep_before]
        d_agg_ids    = dep_agg_id[:n_dep_before]

        mask_le = dw <= ww
        mask_gt = ~mask_le

        # Retained-balance mixing weight ρ_rem (same as M2)
        sigma_in    = float(cum_shield_eth[n_dep_before - 1])
        n_un_before = int(np.searchsorted(wit_t_s_sorted, tw, side="left"))
        sigma_out   = float(cum_unshield_eth[n_un_before - 1]) if n_un_before > 0 else 0.0
        rho = max(0.0, min(1.0, (sigma_in - sigma_out) / sigma_in)) if sigma_in > 0 else 0.0

        # ── Determine evidence case ────────────────────────────────────
        # AR: deposits whose address equals the withdrawal address
        ar_ids = dep_addr_to_ids.get(wit_addr[idx], set())
        # GR: gas-payer-matched deposits (relayer-excluded, from export)
        gr_ids_w = gr_map.get(wid, set())
        ar_gr_ids = ar_ids | gr_ids_w

        # Intersect with amount-compatible candidates before this withdrawal
        candidate_ids = set(d_agg_ids[mask_le])
        ar_gr_candidates = ar_gr_ids & candidate_ids

        pt_ids      = pt_map.get(wid, set())
        pt_candidates = pt_ids & candidate_ids

        if ar_gr_candidates:
            # Hard filter: uniform over matched deposits
            n_match = len(ar_gr_candidates)
            h3 = 0.0 if n_match == 1 else math.log2(n_match)
            case = "ar_gr"

        elif pt_candidates:
            # Soft boost: multiply M2 weight of PT-matched deposits by alpha_p
            w_le = mixture_density(dt_days[mask_le])
            # Apply boost to PT-matched positions
            boost = np.ones(mask_le.sum(), dtype=np.float64)
            for i, aid in enumerate(d_agg_ids[mask_le]):
                if aid in pt_candidates:
                    boost[i] = ALPHA_P
            w_le_boosted = w_le * boost

            # Combine with ρ_rem (larger deposits unchanged)
            z_le = w_le_boosted.sum()
            all_probs = []
            if z_le > 0:
                all_probs.append((1.0 - rho) * w_le_boosted / z_le)
            w_gt = mixture_density(dt_days[mask_gt])
            z_gt = w_gt.sum()
            if z_gt > 0 and mask_gt.any():
                all_probs.append(rho * w_gt / z_gt)
            if not all_probs:
                continue
            p_all = np.concatenate(all_probs)
            p_all /= p_all.sum()
            h3   = entropy_bits(p_all)
            case = "pt"

        else:
            # Fallback: M2 (recompute for consistency)
            w_le = mixture_density(dt_days[mask_le])
            z_le = w_le.sum()
            all_probs = []
            if z_le > 0 and mask_le.any():
                all_probs.append((1.0 - rho) * w_le / z_le)
            w_gt = mixture_density(dt_days[mask_gt])
            z_gt = w_gt.sum()
            if z_gt > 0 and mask_gt.any():
                all_probs.append(rho * w_gt / z_gt)
            if not all_probs:
                continue
            p_all = np.concatenate(all_probs)
            p_all /= p_all.sum()
            h3   = entropy_bits(p_all)
            case = "fallback"

        results.append({
            "agg_id":    wid,
            "H3":        h3,
            "case":      case,
            "n_ar_gr":   len(ar_gr_candidates),
            "n_pt":      len(pt_candidates),
        })

    H3    = np.array([r["H3"] for r in results])
    cases = np.array([r["case"] for r in results])
    valid = np.isfinite(H3)

    print(f"\nRailgun M3 canonical (n={valid.sum():,}):")
    print(f"  Median : {np.median(H3[valid]):.4f} bits")
    print(f"  Mean   : {np.mean(H3[valid]):.4f} bits")
    print(f"  p10/p90: {np.percentile(H3[valid], 10):.4f} / "
          f"{np.percentile(H3[valid], 90):.4f} bits")
    print()
    for c in ["ar_gr", "pt", "fallback"]:
        mask = (cases == c) & valid
        if mask.sum():
            print(f"  {c:10s}: n={mask.sum():,}, median={np.median(H3[mask]):.4f} bits")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["agg_id", "H3", "case", "n_ar_gr", "n_pt"])
        wr.writeheader()
        wr.writerows(results)
    print(f"\nWritten → {OUT_CSV}")


if __name__ == "__main__":
    main()
