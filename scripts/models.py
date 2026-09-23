#!/usr/bin/env python3
"""
models.py — shared entropy model computations (M1–M3).

Internal units (callers must convert before building data containers):
  time  : seconds (float64)
  amount: ETH     (float64)
  entropy: bits

Adding a new protocol requires only:
  1. Build DepositData / WithdrawalData from protocol-specific CSVs.
  2. Build an Evidence dict {withdrawal_id: {"gr": set, "pt": set}}.
  3. Call compute_m1 / compute_m2 / compute_m3 with appropriate flags.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_SECS_PER_DAY = 86_400.0

# Evidence: {withdrawal_agg_id → {"gr": set[dep_agg_id], "pt": set[dep_agg_id]}}
# AR is always computed inline from DepositData.addr_to_ids and WithdrawalData.addr.
Evidence = Dict[int, Dict[str, Set[int]]]


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class DepositData:
    """Time-sorted deposit bundle (all amounts in ETH, times in seconds)."""
    agg_id:      np.ndarray          # int (N,)
    t_s:         np.ndarray          # float64 (N,) sorted ascending
    addr:        np.ndarray          # str (N,) lowercase
    amt_eth:     np.ndarray          # float64 (N,) ETH
    cum_eth:     np.ndarray          # float64 (N,) cumulative deposited ETH
    addr_to_ids: Dict[str, Set[int]] = field(default_factory=dict)

    @classmethod
    def from_arrays(cls,
                    agg_id:  np.ndarray,
                    t_s:     np.ndarray,
                    addr:    np.ndarray,
                    amt_eth: np.ndarray) -> "DepositData":
        """Sort by time and build address → deposit-id index."""
        sort     = np.argsort(t_s)
        agg_id   = np.asarray(agg_id)[sort]
        t_s      = np.asarray(t_s,     dtype=np.float64)[sort]
        addr     = np.asarray(addr)[sort]
        amt_eth  = np.asarray(amt_eth, dtype=np.float64)[sort]
        cum_eth  = np.cumsum(amt_eth)
        a2i: Dict[str, Set[int]] = {}
        for aid, a in zip(agg_id.tolist(), addr.tolist()):
            a2i.setdefault(a, set()).add(int(aid))
        return cls(agg_id=agg_id, t_s=t_s, addr=addr,
                   amt_eth=amt_eth, cum_eth=cum_eth, addr_to_ids=a2i)


@dataclass
class WithdrawalData:
    """Withdrawal bundle — CSV order preserved; sorted copy for searchsorted."""
    agg_id:          np.ndarray  # int (M,) CSV order
    t_s:             np.ndarray  # float64 (M,) seconds, CSV order
    addr:            np.ndarray  # str (M,) lowercase
    amt_eth:         np.ndarray  # float64 (M,) ETH
    t_s_sorted:      np.ndarray  # float64 (M,) seconds, sorted ascending
    cum_eth_sorted:  np.ndarray  # float64 (M,) cumulative withdrawn ETH (sorted order)

    @classmethod
    def from_arrays(cls,
                    agg_id:  np.ndarray,
                    t_s:     np.ndarray,
                    addr:    np.ndarray,
                    amt_eth: np.ndarray) -> "WithdrawalData":
        t_s     = np.asarray(t_s,     dtype=np.float64)
        amt_eth = np.asarray(amt_eth, dtype=np.float64)
        sort    = np.argsort(t_s)
        return cls(
            agg_id         = np.asarray(agg_id),
            t_s            = t_s,
            addr           = np.asarray(addr),
            amt_eth        = amt_eth,
            t_s_sorted     = t_s[sort],
            cum_eth_sorted = np.cumsum(amt_eth[sort]),
        )


# ── Math helpers ──────────────────────────────────────────────────────────────

def mixture_density(dt_days: np.ndarray, pi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """f_θ(Δt) = Σ_k (π_k / μ_k) exp(−Δt / μ_k)  [unnormalized, per-deposit weights]"""
    out = np.zeros(len(dt_days), dtype=np.float64)
    for pi_k, mu_k in zip(pi, mu):
        out += (pi_k / mu_k) * np.exp(-dt_days / mu_k)
    return out


def entropy_bits(weights: np.ndarray) -> float:
    """Shannon entropy in bits of a weight vector (unnormalized OK)."""
    s = weights.sum()
    if s <= 0:
        return math.nan
    p = weights[weights > 0] / s
    return float(-np.dot(p, np.log2(p)))


def mean_finite(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else math.nan


def _rho_rem(deps: DepositData, wits: WithdrawalData, wit_idx: int, n_dep: int) -> float:
    """Retained-balance fraction ρ = max(0, Σin − Σout) / Σin at withdrawal time."""
    sigma_in = float(deps.cum_eth[n_dep - 1])
    if sigma_in <= 0:
        return 0.0
    n_un = int(np.searchsorted(wits.t_s_sorted, wits.t_s[wit_idx], side="left"))
    sigma_out = float(wits.cum_eth_sorted[n_un - 1]) if n_un > 0 else 0.0
    return min(1.0, max(0.0, (sigma_in - sigma_out) / sigma_in))


# ── M1 ────────────────────────────────────────────────────────────────────────

def compute_m1(deps: DepositData, wits: WithdrawalData,
               amount_filter: bool = False) -> List[float]:
    """
    M1 baseline: log2(unique depositor addresses before each withdrawal).

    amount_filter=False [RG]: count all prior deposits — any deposit can
        contribute to a withdrawal via internal UTXO combining.
    amount_filter=True  [PP]: count only deposits where amt(d) >= amt(w) —
        single-deposit protocol, one deposit must cover the full withdrawal.
    """
    result: List[float] = []
    for i in range(len(wits.t_s)):
        n = int(np.searchsorted(deps.t_s, wits.t_s[i], side="left"))
        if n == 0:
            result.append(math.nan)
            continue
        if amount_filter:
            mask   = deps.amt_eth[:n] >= wits.amt_eth[i]
            n_uniq = int(np.unique(deps.addr[:n][mask]).size)
        else:
            n_uniq = int(np.unique(deps.addr[:n]).size)
        result.append(math.log2(n_uniq) if n_uniq > 0 else math.nan)
    return result


# ── M2 ────────────────────────────────────────────────────────────────────────

def compute_m2(deps: DepositData, wits: WithdrawalData,
               pi: np.ndarray, mu: np.ndarray,
               use_rho_rem: bool = True,
               _progress_every: int = 10_000,
               ) -> Tuple[List[float], Dict[int, float]]:
    """
    M2 temporal entropy: H( p_2(d|w) ∝ f_θ(Δt_{d,w}) ).

    use_rho_rem=True  [RG]: mix D^≤(w) with weight (1−ρ) and D^>(w) with ρ,
        where ρ = retained-balance fraction (pool not yet withdrawn).
    use_rho_rem=False [PP]: use only D^≤(w), single-source model.

    Returns:
        h2_list : H2 per withdrawal in wits CSV order (math.nan if no deposits)
        m2_dict : {agg_id → H2} for fast lookup in M3 fallback
    """
    h2_list: List[float] = []
    m2_dict: Dict[int, float] = {}
    M = len(wits.t_s)
    for i in range(M):
        if _progress_every and i % _progress_every == 0:
            print(f"    M2 {i:,}/{M:,}")
        tw, ww = wits.t_s[i], wits.amt_eth[i]
        aid    = int(wits.agg_id[i])
        n      = int(np.searchsorted(deps.t_s, tw, side="left"))
        if n == 0:
            h2_list.append(math.nan); m2_dict[aid] = math.nan; continue

        dt  = (tw - deps.t_s[:n]) / _SECS_PER_DAY
        le  = deps.amt_eth[:n] <= ww
        gt  = ~le
        rho = _rho_rem(deps, wits, i, n) if use_rho_rem else 0.0

        parts: List[np.ndarray] = []
        if le.any():
            w = mixture_density(dt[le], pi, mu); z = w.sum()
            if z > 0:
                parts.append((1.0 - rho) * w / z)
        if gt.any() and rho > 0:
            w = mixture_density(dt[gt], pi, mu); z = w.sum()
            if z > 0:
                parts.append(rho * w / z)
        if not parts:
            h2_list.append(math.nan); m2_dict[aid] = math.nan; continue

        p  = np.concatenate(parts); p /= p.sum()
        h2 = entropy_bits(p)
        h2_list.append(h2); m2_dict[aid] = h2
    return h2_list, m2_dict


# ── M3 ────────────────────────────────────────────────────────────────────────

def compute_m3(deps: DepositData, wits: WithdrawalData,
               m2_dict: Dict[int, float],
               evidence: Evidence,
               pi: np.ndarray, mu: np.ndarray,
               alpha_p: float = 10.0,
               use_rho_rem: bool = True) -> pd.DataFrame:
    """
    M3 graph-conditioned entropy — all four variant configurations.

    evidence[wid] = {"gr": set[dep_agg_id], "pt": set[dep_agg_id]}
    AR is computed inline from deps.addr_to_ids and wits.addr.
    Candidate set: D^≤(w) = {d : amt(d) ≤ amt(w) AND time(d) < time(w)}.

    Priority:
      1. AR ∪ GR candidates exist → hard filter (uniform over matched set)
      2. PT candidates exist (no AR/GR) → soft boost α_p (or hard at α→∞)
      3. fallback → M2

    Returns DataFrame with unified columns (same for every protocol):
        withdrawal_id, n_ar, n_gr, n_pt,
        H2, H3_ar, H3_ar_gr, H3_ar_gr_pt10, H3_ar_gr_ptinf
    """
    def _hard(n: int) -> float:
        return 0.0 if n <= 1 else math.log2(n)

    rows = []
    for i in range(len(wits.t_s)):
        tw, ww = wits.t_s[i], wits.amt_eth[i]
        aid    = int(wits.agg_id[i])
        h2     = m2_dict.get(aid, math.nan)

        n = int(np.searchsorted(deps.t_s, tw, side="left"))
        if n == 0:
            rows.append({"withdrawal_id": aid, "n_ar": 0, "n_gr": 0, "n_pt": 0,
                         "H2": h2, "H3_ar": h2, "H3_ar_gr": h2,
                         "H3_ar_gr_pt10": h2, "H3_ar_gr_ptinf": h2})
            continue

        d_ids = deps.agg_id[:n]
        le    = deps.amt_eth[:n] <= ww
        gt    = ~le
        cands = set(d_ids[le].tolist())
        dt    = (tw - deps.t_s[:n]) / _SECS_PER_DAY
        rho   = _rho_rem(deps, wits, i, n) if use_rho_rem else 0.0

        ev    = evidence.get(aid, {})
        ar    = deps.addr_to_ids.get(wits.addr[i], set()) & cands
        gr    = ev.get("gr", set()) & cands
        pt    = ev.get("pt", set()) & cands
        ar_gr = ar | gr

        def _pt_boost(k: float) -> float:
            w_le = mixture_density(dt[le], pi, mu)
            b    = np.ones(int(le.sum()), dtype=np.float64)
            for j, a in enumerate(d_ids[le].tolist()):
                if a in pt:
                    b[j] = k
            wb = w_le * b; z = wb.sum()
            parts_: List[np.ndarray] = []
            if z > 0:
                parts_.append((1.0 - rho) * wb / z)
            if gt.any() and rho > 0:
                wg = mixture_density(dt[gt], pi, mu); zg = wg.sum()
                if zg > 0:
                    parts_.append(rho * wg / zg)
            if not parts_:
                return h2
            p = np.concatenate(parts_); p /= p.sum()
            return entropy_bits(p)

        h3_ar    = _hard(len(ar))    if ar    else h2
        h3_ar_gr = _hard(len(ar_gr)) if ar_gr else h2

        if ar_gr:
            h3_pt10 = h3_ptinf = _hard(len(ar_gr))
        elif pt:
            h3_pt10  = _pt_boost(alpha_p)
            h3_ptinf = _hard(len(pt))
        else:
            h3_pt10 = h3_ptinf = h2

        rows.append({
            "withdrawal_id":    aid,
            "n_ar":             len(ar),
            "n_gr":             len(gr),
            "n_pt":             len(pt),
            "H2":               h2,
            "H3_ar":            h3_ar,
            "H3_ar_gr":         h3_ar_gr,
            "H3_ar_gr_pt10":    h3_pt10,
            "H3_ar_gr_ptinf":   h3_ptinf,
        })
    return pd.DataFrame(rows)


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_m3(m3_df: pd.DataFrame,
                 m2_dict: Dict[int, float]) -> Dict[str, float]:
    """
    Aggregate M3 variants to mean entropy.

    Rows stored with H3 == H2 (fallback rows) are updated with fresh M2
    from m2_dict before averaging — so M3 numbers stay consistent with the
    current M2 parameters even if the canonical CSV was computed with old M2.

    Returns dict keys: m3_ar, m3_ar_gr, m3_ar_gr_pt_10, m3_ar_gr_pt_inf,
                       ar_cov, gr_cov, pt_cov
    """
    df = m3_df.copy()
    df["H2_fresh"] = df["withdrawal_id"].map(m2_dict)
    result: Dict[str, float] = {}
    for col, key in [("H3_ar",          "m3_ar"),
                     ("H3_ar_gr",        "m3_ar_gr"),
                     ("H3_ar_gr_pt10",   "m3_ar_gr_pt_10"),
                     ("H3_ar_gr_ptinf",  "m3_ar_gr_pt_inf")]:
        h3  = df[col].astype(float).values
        h2s = df["H2"].astype(float).values
        h2f = df["H2_fresh"].astype(float).values
        is_fallback = np.isclose(h3, h2s, atol=1e-6)
        corrected   = np.where(is_fallback, h2f, h3)
        result[key] = mean_finite(corrected)
    result["ar_cov"] = float((df["n_ar"] > 0).mean())
    result["gr_cov"] = float((df["n_gr"] > 0).mean())
    result["pt_cov"] = float(((df["n_pt"] > 0) & (df["n_ar"] == 0)).mean())
    return result
