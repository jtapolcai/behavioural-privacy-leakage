#!/usr/bin/env python3
"""
compute_table2.py
=================
Recomputes supported values and marks unavailable canonical results in Table 2 (entropy_model_table.tex) and regenerates
the LaTeX table.

Models:
  M1  – uniform over all-time amount-compatible pool (log2 N_le)
  M2  – declared temporal scenario (three-component mixture)
  M3  – graph-conditioned M2
        · AR        – address-reuse hard filter only
        · AR+GR     – AR + gas-payer reuse (relayer-excluded, K_rel=10)
        · AR+GR+PT (α_p=10)   – soft boost on PT-matched deposits
        · AR+GR+PT (α_p→∞)   – PT treated as hard filter
  M4  – unavailable until per-candidate component distributions are supplied

Outputs:
  Figures/entropy_models/table2_values.json   – all computed medians
  Figures/entropy_models/entropy_model_table.tex  – regenerated LaTeX table
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

# ── Paths ────────────────────────────────────────────────────────────────────
import argparse
PAPER        = Path(__file__).resolve().parents[1]
SOURCE_PAPER = PAPER
FIGS         = _FIGURES / "entropy_models"   # outputs from compute_rg_entropy_models.py
RG_DATA      = _RG_DATA                      # railgun data/ directory
AGG          = RG_DATA / "aggregated"
WITNESS_CSV  = _RG_DATA.parent / "results" / "witness" / "witness_kto1_all_step0.0001.csv"
PP_EV        = SOURCE_PAPER / "analysis" / "pp_origin_entropy" / "per_withdrawal.csv"

# ── Parameters ───────────────────────────────────────────────────────────────
K_REL   = 10      # relayer exclusion threshold
ALPHA_P = 10.0    # PT soft-boost factor for α_p=10 variant

# ── Helpers ──────────────────────────────────────────────────────────────────

def median_finite(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else math.nan


def h_hard(n: int) -> float:
    """Entropy of uniform distribution over n items."""
    return 0.0 if n <= 1 else math.log2(n)


# ════════════════════════════════════════════════════════════════════════════
# 1.  RAILGUN values
# ════════════════════════════════════════════════════════════════════════════

def compute_rg(vals: dict) -> None:
    print("=== Railgun ===")

    # ── M1 (address-level uniform baseline: log2 unique depositor addresses) ─
    rg_ev = pd.read_csv(FIGS / "rg_entropy_per_withdrawal.csv")
    witness_df = pd.read_csv(WITNESS_CSV)
    h_all_df = rg_ev[rg_ev["horizon"] == "all"].copy()
    h_all_df = h_all_df.merge(
        witness_df[["ti", "H_naive_addr", "H_knap_addr"]], on="ti", how="left"
    )
    vals["rg_m1"] = median_finite(h_all_df["H_naive_addr"])
    print(f"  M1 (addr-level) : {vals['rg_m1']:.4f}")

    # ── M2 (address-level calibrated) ─────────────────────────────────────
    rg_m2 = pd.read_csv(FIGS / "rg_m2_canonical.csv")
    # column name may be H2_canonical (from compute_m2_canonical.py) or H2_addr
    _m2_col = "H2_canonical" if "H2_canonical" in rg_m2.columns else "H2_addr"
    vals["rg_m2"] = median_finite(rg_m2[_m2_col])
    print(f"  M2              : {vals['rg_m2']:.4f}")
    m2_dict = dict(zip(rg_m2["agg_id"].astype(int),
                        rg_m2[_m2_col].astype(float)))

    # ── Load deposit addresses for AR ─────────────────────────────────────
    print("  Loading deposit data for M3-AR …")
    shields = list(csv.DictReader(open(AGG / "eth_shield_aggregated.csv")))
    dep_agg_id = np.array([int(r["agg_id"]) for r in shields])
    dep_t_s    = np.array([int(r["last_time_ns"]) / 1e9 for r in shields])
    dep_addr   = np.array([r["address"].lower().strip() for r in shields])
    dep_wei    = np.array([int(r["note_wei"]) for r in shields], dtype=object)
    sort_d = np.argsort(dep_t_s)
    dep_agg_id, dep_t_s, dep_addr, dep_wei = (
        dep_agg_id[sort_d], dep_t_s[sort_d],
        dep_addr[sort_d],   dep_wei[sort_d])

    # address → set of deposit agg_ids
    dep_addr_to_ids: dict[str, set[int]] = {}
    for aid, addr in zip(dep_agg_id, dep_addr):
        dep_addr_to_ids.setdefault(addr, set()).add(int(aid))

    # ── Load withdrawals ───────────────────────────────────────────────────
    unshields = list(csv.DictReader(
        open(AGG / "eth_unshields_aggregated.csv")))
    wit_agg_id = np.array([int(r["agg_id"]) for r in unshields])
    wit_t_s    = np.array([int(r["first_time_ns"]) / 1e9 for r in unshields])
    wit_wei    = np.array([int(r["note_wei"]) for r in unshields], dtype=np.float64)
    wit_addr   = np.array([r["address"].lower().strip() for r in unshields])

    # ── M3-AR only (fast pass – no temporal prior needed for hard filter) ──
    print("  Computing M3-AR …")
    h3_ar = []
    ar_cov = 0
    for idx in range(len(wit_t_s)):
        tw  = wit_t_s[idx]
        ww  = wit_wei[idx]
        wid = int(wit_agg_id[idx])

        n_before = int(np.searchsorted(dep_t_s, tw, side="left"))
        if n_before == 0:
            h3_ar.append(m2_dict.get(wid, math.nan))
            continue

        candidate_ids = set(dep_agg_id[:n_before][dep_wei[:n_before] <= ww].tolist())
        ar_ids        = dep_addr_to_ids.get(wit_addr[idx], set())
        ar_cands      = ar_ids & candidate_ids

        if ar_cands:
            ar_cov += 1
            h3_ar.append(h_hard(len(ar_cands)))
        else:
            h3_ar.append(m2_dict.get(wid, math.nan))

    vals["rg_m3_ar"] = median_finite(h3_ar)
    vals["rg_ar_cov"] = ar_cov / len(h3_ar)
    print(f"  M3-AR           : {vals['rg_m3_ar']:.4f}  "
          f"(AR hard-filter applied to {ar_cov:,} / {len(h3_ar):,})")

    # ── M3 AR+GR+PT from canonical CSV ────────────────────────────────────
    m3 = pd.read_csv(FIGS / "rg_m3_canonical.csv")
    # addr-level M2 fallback for all non-graph cases
    m3["H2"] = m3["agg_id"].map(m2_dict)
    fallback_mask = m3["case"] == "fallback"
    pt_mask       = m3["case"] == "pt"

    # AR+GR (no PT): 'fallback' and 'pt' cases → addr-level M2
    h3_ar_gr = m3["H3"].copy()
    h3_ar_gr[fallback_mask] = m3.loc[fallback_mask, "H2"]
    h3_ar_gr[pt_mask]       = m3.loc[pt_mask,       "H2"]
    vals["rg_m3_ar_gr"] = median_finite(h3_ar_gr)
    print(f"  M3-AR+GR        : {vals['rg_m3_ar_gr']:.4f}")

    # AR+GR+PT α=10: 'fallback' cases → addr-level M2; 'pt' keeps PT-boosted H3
    h3_pt10 = m3["H3"].copy()
    h3_pt10[fallback_mask] = m3.loc[fallback_mask, "H2"]
    vals["rg_m3_ar_gr_pt_10"] = median_finite(h3_pt10)
    print(f"  M3-AR+GR+PT α=10: {vals['rg_m3_ar_gr_pt_10']:.4f}")

    # AR+GR+PT α→∞: 'fallback' → addr-level M2; 'pt' → log2(n_pt) hard filter
    h3_inf = m3["H3"].copy()
    h3_inf[fallback_mask] = m3.loc[fallback_mask, "H2"]
    h3_inf[pt_mask]       = m3.loc[pt_mask, "n_pt"].apply(h_hard)
    vals["rg_m3_ar_gr_pt_inf"] = median_finite(h3_inf)
    print(f"  M3-AR+GR+PT α→∞ : {vals['rg_m3_ar_gr_pt_inf']:.4f}")

    h180 = rg_ev[rg_ev["horizon"] == "180d"]
    vals["rg_p_feas"] = float(1 - h180["skipped"].mean())
    vals["rg_sat_frac"] = float(h180["saturated"].mean())

    # ── M4 (address-level knapsack participation entropy, horizon=all) ───────
    # H_knap_addr from witness_kto1_all_step0.0001.csv (per-address participation
    # entropy; slightly lower than item-level H_knap_item).
    # Skipped targets (2.5% with no knapsack data) fall back to addr-level M2.
    h4_rg = h_all_df["H_knap_addr"].values.copy().astype(float)
    ti_all = h_all_df["ti"].values.astype(int)
    skip_mask = h_all_df["skipped"].values == 1
    for i, (is_skip, ti) in enumerate(zip(skip_mask, ti_all)):
        if is_skip:
            h4_rg[i] = m2_dict.get(int(ti), math.nan)
    vals["rg_m4_knap"] = median_finite(h4_rg)
    n_skip_all = int(skip_mask.sum())
    n_sat_all = int((h_all_df["saturated"].values == 1).sum())

    # p_cov=0.25 for RG: per-target mixing weight α = p_cov / p_feas
    # (so that p_feas * α = p_cov overall).
    # H4_025(w) ≈ α*H_knap(w) + (1-α)*H2(w)  for hit targets (concavity bound)
    #           = H2(w)                         for skip targets
    _alpha_025 = 0.25 / vals["rg_p_feas"]   # ≈ 0.2565 for p_feas≈0.9746
    h4_rg_pcov025 = np.empty(len(h_all_df), dtype=float)
    for i, (is_skip, ti_val) in enumerate(zip(skip_mask, ti_all)):
        h2_i = m2_dict.get(int(ti_val), math.nan)
        if is_skip:
            h4_rg_pcov025[i] = h2_i
        else:
            h4_rg_pcov025[i] = _alpha_025 * h4_rg[i] + (1 - _alpha_025) * h2_i
    vals["rg_m4_pcov025"] = median_finite(h4_rg_pcov025)
    vals["rg_m4_pcov05"] = None   # unavailable: need raw participation weights
    vals["rg_h_knap"] = vals["rg_m4_knap"]
    vals["rg_m4_status"] = (
        f"Computed from witness_kto1_all_step0.0001 (step=0.0001 ETH, addr-level); "
        f"{n_skip_all}/{len(h_all_df)} skipped targets fall back to addr-level M2; "
        f"{n_sat_all}/{len(h_all_df)} saturated (K≥500). "
        f"p_cov=0.5 mixing unavailable (raw per-deposit participation counts not archived)."
    )
    print(f"  M4 (knapsack)   : {vals['rg_m4_knap']:.4f}  "
          f"(skipped={n_skip_all:,}, saturated={n_sat_all:,})")


# ════════════════════════════════════════════════════════════════════════════
# 2.  PRIVACY POOLS values
# ════════════════════════════════════════════════════════════════════════════

def compute_pp(vals: dict) -> None:
    print("\n=== Privacy Pools ===")

    pp_ev = pd.read_csv(PP_EV)
    sc = pp_ev[
        (pp_ev["model"] == "reuse_mix") &
        (pp_ev["scenario"] == "exact_rows_collapsed")
    ].copy()
    n_pp = len(sc)
    print(f"  PP withdrawals  : {n_pp:,}")

    # ── M1 (canonical: log2 n_amount_origins = H0_amount) ─────────────────
    # H0_amount = log2(n_amount_origins), the Hartley entropy over D^≤(w).
    # Targets with n_amount_origins=0 (72 rows) have no compatible deposits
    # and are excluded from the median by median_finite().
    vals["pp_m1"] = median_finite(sc["H0_amount"])
    n_m1_valid = int((sc["H0_amount"] > 0).sum())
    print(f"  M1 (H0_amount)  : {vals['pp_m1']:.4f}  (n_valid={n_m1_valid:,})")

    # Archived all-earlier baseline (H_uniform = log2 N^all_earlier):
    # different support (all strictly earlier, not amount-filtered D^≤).
    vals["pp_all_earlier_baseline"] = median_finite(sc["H_uniform"])
    print(f"  All-earlier base: {vals['pp_all_earlier_baseline']:.4f}  (archived H_uniform)")

    # ── M2 (canonical JSON) ───────────────────────────────────────────────
    pp_m2_json = json.load(open(FIGS / "pp_m2_canonical.json"))
    vals["pp_m2"] = pp_m2_json["median"]
    print(f"  M2 (canonical)  : {vals['pp_m2']:.4f}")

    # ── M3 (address evidence from pp_m3_canonical.csv) ───────────────────
    pp_m3_csv = FIGS / "pp_m3_canonical.csv"
    if pp_m3_csv.exists():
        pp_m3 = pd.read_csv(pp_m3_csv)
        vals["pp_m3_ar"]          = median_finite(pp_m3["H3_ar"])
        vals["pp_m3_ar_gr"]       = median_finite(pp_m3["H3_ar_gr"])
        vals["pp_m3_ar_gr_pt_10"] = median_finite(pp_m3["H3_ar_gr_pt10"])
        vals["pp_m3_ar_gr_pt_inf"]= median_finite(pp_m3["H3_ar_gr_ptinf"])
        n_pp_m3 = len(pp_m3)
        vals["pp_ar_cov"]  = float((pp_m3["n_ar"] > 0).sum() / n_pp_m3)
        vals["pp_gr_cov"]  = float((pp_m3["n_gr"] > 0).sum() / n_pp_m3)
        vals["pp_pt_cov"]  = float(((pp_m3["n_pt"] > 0) & (pp_m3["n_ar"] == 0)).sum() / n_pp_m3)
        vals["pp_m3_status"] = "Computed from deposit/withdrawal address join on exact_rows_collapsed cohort."
        print(f"  M3-AR           : {vals['pp_m3_ar']:.4f}  (AR cov {vals['pp_ar_cov']*100:.1f}%)")
        print(f"  M3-AR+GR        : {vals['pp_m3_ar_gr']:.4f}")
        print(f"  M3-AR+GR+PT α=10: {vals['pp_m3_ar_gr_pt_10']:.4f}")
        print(f"  M3-AR+GR+PT α→∞ : {vals['pp_m3_ar_gr_pt_inf']:.4f}")
    else:
        for key in ("m3_ar", "m3_ar_gr", "m3_ar_gr_pt_10", "m3_ar_gr_pt_inf"):
            vals["pp_" + key] = None
        vals["pp_ar_cov"] = None
        vals["pp_m3_status"] = "pp_m3_canonical.csv not found; run compute_pp_m3.py."
        print("  PP M3: pp_m3_canonical.csv not found.")

    # ── M4 (from pp_m4_canonical.csv) ────────────────────────────────────
    pp_m4_csv = FIGS / "pp_m4_canonical.csv"
    if pp_m4_csv.exists():
        pp_m4 = pd.read_csv(pp_m4_csv)
        # H4_pcov025: linear approx p4(0.25) = 0.5*p2 + 0.5*p4(0.5)
        # => H4_025 ≈ 0.5*H2 + 0.5*H4_pcov05  (lower bound by concavity)
        if "H4_pcov025" not in pp_m4.columns:
            pp_m4["H4_pcov025"] = 0.5 * pp_m4["H2"] + 0.5 * pp_m4["H4_pcov05"]
        vals["pp_m4_pcov025"]    = median_finite(pp_m4["H4_pcov025"])
        vals["pp_m4_pcov_pfeas"] = median_finite(pp_m4["H4_pcov_pfeas"])
        vals["pp_m4"]            = None   # generic m4 slot unused
        n_hit_w_data = int((pp_m4["n_knap_deps"] > 0).sum())
        vals["pp_m4_status"] = (
            f"Computed from k=1,2,3 knapsack CSV exports covering "
            f"{n_hit_w_data}/{len(pp_m4)} canonical targets with deposit-level data; "
            f"remaining hits fall back to M2. "
            f"p_cov=0.25 uses linear distribution interpolation (H4_025=0.5*H2+0.5*H4_05, lower bound)."
        )
        print(f"  M4 (p_cov=0.25) : {vals['pp_m4_pcov025']:.4f}")
        print(f"  M4 (p_cov=p_feas): {vals['pp_m4_pcov_pfeas']:.4f}")
    else:
        for key in ("m4", "m4_pcov05", "m4_pcov_pfeas"):
            vals["pp_" + key] = None
        vals["pp_m4_status"] = "pp_m4_canonical.csv not found; run compute_pp_m4.py."
        print("  PP M4: pp_m4_canonical.csv not found.")

    vals["pp_h_knap"] = None
    vals["pp_one_to_one_amount_support_fraction"] = float((sc["n_direct_origins"] > 0).mean())
    vals["pp_p_feas"] = float(1 - sc["abstained"].mean())
    vals["pp_abs_frac"] = float(sc["abstained"].mean())
    vals["pp_n"] = n_pp


# ════════════════════════════════════════════════════════════════════════════
# 3.  Coverage statistics
# ════════════════════════════════════════════════════════════════════════════

def compute_coverage(vals: dict) -> None:
    print("\n=== Coverage statistics ===")

    _ptgr_path = FIGS / "rg_pt_gr_counts.csv"
    if not _ptgr_path.exists():
        print(f"  ⚠  {_ptgr_path.name} not found — skipping coverage statistics")
        return

    m3 = pd.read_csv(FIGS / "rg_m3_canonical.csv")
    ptgr = pd.read_csv(_ptgr_path)
    rg_ev = pd.read_csv(FIGS / "rg_entropy_per_withdrawal.csv")
    n_rg  = len(rg_ev[rg_ev["horizon"] == "all"])

    n_ar_gr_cases = (m3["case"] == "ar_gr").sum()
    n_pt_cases    = (m3["case"] == "pt").sum()

    # GR coverage (from ptgr file)
    n_gr_raw = (ptgr["n_gr"] > 0).sum()

    vals["rg_n"]          = n_rg
    vals["rg_ar_gr_cov"]  = float(n_ar_gr_cases / n_rg)   # 4.1%
    vals["rg_pt_only_cov"]= float(n_pt_cases    / n_rg)   # 2.3%
    vals["rg_gr_raw_cov"] = float(n_gr_raw      / n_rg)   # ~5.8%

    # AR coverage (from M3-AR computation run above)
    # Approximate from M3-AR hard-filter cases vs total
    print(f"  RG AR+GR coverage: {vals['rg_ar_gr_cov']*100:.1f}%")
    print(f"  RG PT-only coverage: {vals['rg_pt_only_cov']*100:.1f}%")
    if vals.get("pp_ar_cov") is not None:
        print(f"  PP AR coverage: {vals['pp_ar_cov']*100:.1f}%  (from pp_m3_canonical.csv)")
    else:
        print("  PP AR coverage: unavailable.")


# ════════════════════════════════════════════════════════════════════════════
# 4.  Generate LaTeX macros + table
# ════════════════════════════════════════════════════════════════════════════

def fmt(v, decimals=2) -> str:
    """Format a float value or return '---' for None/nan."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    if isinstance(v, str):
        return v   # already formatted (e.g. percentage strings)
    return f"{v:.{decimals}f}"


def write_macros(vals: dict, output_dir: Path) -> Path:
    """Generate macros_generated.tex with \\newcommand for every key result.

    Include in the LaTeX preamble with:
        \\input{Figures/macros_generated}
    """

    def ent(key, d=2):
        v = vals.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return r"\text{---}"
        return f"{v:.{d}f}"

    def eff(key):
        """Effective candidate count 2^H, formatted with LaTeX thousands sep."""
        v = vals.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return r"\text{---}"
        n = round(2 ** v)
        parts = []
        s = str(n)
        while len(s) > 3:
            parts.append(s[-3:])
            s = s[:-3]
        parts.append(s)
        return r"{,}".join(reversed(parts))

    def pct(key, d=1):
        v = vals.get(key)
        if v is None:
            return r"\text{---}"
        return f"{v * 100:.{d}f}\\%"

    def diff(key_a, key_b, d=2):
        """key_a − key_b."""
        a, b = vals.get(key_a), vals.get(key_b)
        if a is None or b is None:
            return r"\text{---}"
        return f"{a - b:.{d}f}"

    def factor_approx(key_a, key_b, d=1):
        """2^(a−b) as a decimal factor (e.g. 2.7)."""
        a, b = vals.get(key_a), vals.get(key_b)
        if a is None or b is None:
            return r"\text{---}"
        return f"{2 ** (a - b):.{d}f}"

    def nc(name, body):
        return f"\\newcommand{{\\{name}}}{{{body}}}"

    lines = [
        "% Auto-generated by scripts/compute_table2.py — do not edit by hand.",
        "% Include in the LaTeX preamble:  \\input{Figures/macros_generated}",
        "%",
        "% ── M1  uniform baseline ─────────────────────────────────────────",
        nc("RGMone",        ent("rg_m1")),
        nc("PPMone",        ent("pp_m1")),
        nc("RGeffMone",     eff("rg_m1")),
        nc("PPeffMone",     eff("pp_m1")),
        "%",
        "% ── M2  temporal ────────────────────────────────────────────────",
        nc("RGMtwo",        ent("rg_m2")),
        nc("PPMtwo",        ent("pp_m2")),
        "%",
        "% ── M3  graph evidence ──────────────────────────────────────────",
        nc("RGMthreeAR",      ent("rg_m3_ar")),
        nc("RGMthreeARGR",    ent("rg_m3_ar_gr")),
        nc("RGMthreeBest",    ent("rg_m3_ar_gr_pt_inf")),
        nc("RGeffMthreeBest", eff("rg_m3_ar_gr_pt_inf")),
        nc("PPMthreeAR",      ent("pp_m3_ar")),
        nc("PPMthreeARGR",    ent("pp_m3_ar_gr")),
        nc("RGMthreeARGRPTten", ent("rg_m3_ar_gr_pt_10")),
        nc("PPMthreeARGRPTten", ent("pp_m3_ar_gr_pt_10")),
        nc("PPMthreeBest",    ent("pp_m3_ar_gr_pt_inf")),
        nc("PPeffMthreeBest", eff("pp_m3_ar_gr_pt_inf")),
        "%",
        "% ── M4  amount decomposition ─────────────────────────────────────",
        nc("RGMfourPcov",      ent("rg_m4_pcov025")),
        nc("PPMfourPcov",      ent("pp_m4_pcov025")),
        nc("RGMfourPfeas",     ent("rg_m4_knap")),
        nc("PPMfourPfeas",     ent("pp_m4_pcov_pfeas")),
        nc("RGeffMfourPfeas",  eff("rg_m4_knap")),
        nc("PPeffMfourPfeas",  eff("pp_m4_pcov_pfeas")),
        "% M4 reductions relative to M2 (M2 − M4, and the factor 2^{M2−M4})",
        nc("RGMtwoMfourDelta",  diff("rg_m2", "rg_m4_pcov025")),
        nc("PPMtwoMfourDelta",  diff("pp_m2", "pp_m4_pcov025")),
        nc("RGMtwoMfourFactor", factor_approx("rg_m2", "rg_m4_pcov025")),
        nc("PPMtwoMfourFactor", factor_approx("pp_m2", "pp_m4_pcov025")),
        "%",
        "% ── Coverage & feasibility ──────────────────────────────────────",
        nc("RGpfeasPct",  pct("rg_p_feas")),
        nc("PPpfeasPct",  pct("pp_p_feas")),
        nc("PPARcovPct",  pct("pp_ar_cov")),
        "%",
        "% ── PP all-earlier baseline ─────────────────────────────────────",
        nc("PPallEarlier", ent("pp_all_earlier_baseline")),
        "%",
        "% ── Dataset sizes ───────────────────────────────────────────────",
        nc("RGnTotal",    "38{,}472"),
        nc("PPnTotal",    f"{vals.get('pp_n', 4206):,}".replace(",", "{,}")),
        "",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "macros_generated.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Written → {out}")
    return out


def write_table(vals: dict, output_dir: Path) -> Path:
    tex = (
        "% Generated by scripts/compute_table2.py; missing results are not zero.\n"
        "\\begin{table}[t]\n"
        "\\centering\\small\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.12}\n"
        "\\begin{tabular}{@{}ll p{5.5cm} rr@{}}\n"
        "\\toprule\n"
        "Model & Evidence & Configuration & Railgun & PP\\\\\n"
        "\\midrule\n"
        "M1 & baseline & Uniform over unique depositor addresses & @rg_m1@ & @pp_m1@\\\\\n"
        "M2 & timing & Declared temporal scenario with retained-balance mixing & @rg_m2@ & @pp_m2@\\\\\n"
        "M3 & graph & AR uniform restriction; M2 fallback & @rg_m3_ar@ & @pp_m3_ar@\\\\\n"
        " & & AR+GR uniform restriction & @rg_m3_ar_gr@ & @pp_m3_ar_gr@\\\\\n"
        " & & AR+GR; PT within-component boost ($\\alpha_p=10$) & @rg_m3_ar_gr_pt_10@ & @pp_m3_ar_gr_pt_10@\\\\\n"
        " & & AR+GR; PT uniform restriction & @rg_m3_ar_gr_pt_inf@ & @pp_m3_ar_gr_pt_inf@\\\\\n"
        "M4 & amounts & Enumerated-history coverage $p_{\\rm cov}=0.25$ & @rg_m4_pcov025@ & @pp_m4_pcov025@\\\\\n"
        " & & Theoretical lower bound ($p_{\\rm cov}=p_{\\rm feas}$) & @rg_m4_knap@ & @pp_m4_pcov_pfeas@\\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Median model entropy (bits). Railgun summaries use 38,472\n"
        "aggregate target records; M2 has 4,205 valid PP targets.\n"
        "M2 uses declared time scales of 403/246 days for the slow component,\n"
        "not an identified population holding-time distribution.\n"
        "Railgun M3 is the explicit piecewise rule of \\Cref{def:m3};\n"
        "PT boosting preserves the larger-deposit component mass.\n"
        "Dashes indicate unavailable results under the stated definitions.\n"
        "The PP archived all-earlier baseline is @pp_all_earlier_baseline@ bits,\n"
        "not the amount-threshold M1. Its one-to-one amount-support frequency\n"
        "must not be interpreted as address reuse.\n"
        "PP M4 mixes the canonical M2 distribution with a participation-weighted\n"
        "temporal distribution derived from the $k\\in\\{1,2,3\\}$ knapsack exports\n"
        "($p_{\\rm feas}=@pp_pfeas_pct@$; targets without per-deposit export data fall back to M2;\n"
        "$p_{\\rm cov}=0.25$ row uses linear distribution interpolation\n"
        "$p_4(0.25)=\\tfrac{1}{2}p_2+\\tfrac{1}{2}p_4(0.5)$, a lower bound on entropy by concavity).\n"
        "Railgun M4 is the knapsack participation entropy from the all-time\n"
        "witness enumeration (step $=0.0001$\\,ETH; $p_{\\rm feas}=@rg_pfeas_pct@$;\n"
        "skipped targets fall back to M2).}\n"
        "\\label{tab:entropy_model_numeric}\n"
        "\\end{table}\n"
    )
    for key, value in vals.items():
        if "@" + key + "@" in tex:
            tex = tex.replace("@" + key + "@", fmt(value))
    if "@" in tex.replace("@{}", ""):
        # Tabular alignment may contain @{}, but no value placeholder may remain.
        import re
        if re.search(r"@[a-z_]+@", tex):
            raise ValueError("Unresolved table value")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "sec4_entropy_model_table.tex"
    out.write_text(tex)
    return out


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global FIGS, PP_EV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-paper", type=Path, default=PAPER)
    parser.add_argument("--output-dir", type=Path, default=_FIGURES)
    parser.add_argument("--figs-dir", type=Path, default=None,
                        help="Directory with entropy_models/ outputs "
                             "[default: _FIGURES/entropy_models via _paths.py]")
    args = parser.parse_args()
    # Use entropy_models subdir from the pipeline output (not legacy Figures/data/)
    FIGS = args.figs_dir if args.figs_dir else (_FIGURES / "entropy_models")
    PP_EV = args.source_paper / "analysis" / "pp_origin_entropy" / "per_withdrawal.csv"
    vals = {}
    compute_rg(vals)
    if PP_EV.exists():
        compute_pp(vals)
        compute_coverage(vals)
    else:
        print(f"  ⚠  PP per_withdrawal.csv not found — skipping PP and coverage sections")
        # Pre-fill all PP template placeholders with None so write_table() can
        # substitute them with "---" instead of raising "Unresolved table value".
        for _k in ("pp_m1", "pp_m2", "pp_m3_ar", "pp_m3_ar_gr",
                   "pp_m3_ar_gr_pt_10", "pp_m3_ar_gr_pt_inf",
                   "pp_m4_pcov025", "pp_m4_pcov_pfeas",
                   "pp_all_earlier_baseline"):
            vals.setdefault(_k, None)
    vals["coverage_note"] = "Hit fractions are export-flag diagnostics, not verified true-history coverage."
    # Percentage strings needed in the table caption
    def _pct(v, d=1):
        return "---" if v is None else f"{v * 100:.{d}f}\\%"
    vals["pp_pfeas_pct"] = _pct(vals.get("pp_p_feas"))
    vals["rg_pfeas_pct"] = _pct(vals.get("rg_p_feas"))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Table is now static in the paper repo (uses \newcommand macros directly).
    # write_table() is kept for reference but no longer called from the pipeline.
    write_macros(vals, out_dir)
    (FIGS / "table2_values.json").write_text(json.dumps(vals, indent=2, allow_nan=False) + "\n")
    print(json.dumps(vals, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
