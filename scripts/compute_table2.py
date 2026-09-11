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

    # p_cov=0.25 for RG: p4(0.25) = 0.5*p2 + 0.5*p4(pfeas)
    # => H4_025 ≈ 0.5*H2_addr + 0.5*H_knap_addr  (lower bound by concavity)
    h4_rg_pcov025 = np.empty(len(h_all_df), dtype=float)
    for i, (is_skip, ti_val) in enumerate(zip(skip_mask, ti_all)):
        h2_i = m2_dict.get(int(ti_val), math.nan)
        if is_skip:
            h4_rg_pcov025[i] = h2_i
        else:
            h4_rg_pcov025[i] = 0.5 * h2_i + 0.5 * h4_rg[i]
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
# 4.  Generate LaTeX table
# ════════════════════════════════════════════════════════════════════════════

def fmt(v, decimals=2) -> str:
    """Format a float value or return '---' for None/nan."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{v:.{decimals}f}"


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
        "($p_{\\rm feas}=98.3\\%$; targets without per-deposit export data fall back to M2;\n"
        "$p_{\\rm cov}=0.25$ row uses linear distribution interpolation\n"
        "$p_4(0.25)=\\tfrac{1}{2}p_2+\\tfrac{1}{2}p_4(0.5)$, a lower bound on entropy by concavity).\n"
        "Railgun M4 is the knapsack participation entropy from the all-time\n"
        "witness enumeration (step $=0.0001$\\,ETH; $p_{\\rm feas}=97.5\\%$;\n"
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
    vals["coverage_note"] = "Hit fractions are export-flag diagnostics, not verified true-history coverage."
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(vals, out_dir)
    (FIGS / "table2_values.json").write_text(json.dumps(vals, indent=2, allow_nan=False) + "\n")
    print(json.dumps(vals, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
