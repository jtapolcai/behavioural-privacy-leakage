#!/usr/bin/env python3
"""
compute_fee_compensated_hamming.py
===================================
Fee-compensated Hamming-distance fingerprint analysis for Railgun.

Three comparisons are made for each (shield, unshield) candidate pair:
  RAW   : hamming(frac18(deposit_amount_wei), frac18(unshield_amount_wei))
            — what Ali's thesis does (incorrect: ignores fees)
  PROTO : hamming(frac18(deposit_wei),        frac18(note_wei))
            — protocol-fee corrected (deposit_wei = post-0.25% note;
              note_wei = amount_wei + fee_wei, recovering the shielded note)
  BROAD : hamming(frac18(deposit_wei),        frac18(note_wei + broadcaster_est))
            — additionally compensates for the broadcaster gas fee
              broadcaster_est ≈ 1.10 × transaction_fee_eth × 1e18  (wei)

Key insight (from thesis_kanan §H3):
  - Shield protocol fee 0.25% is already absorbed: deposit_wei is post-fee
  - Unshield protocol fee 0.25%: note_wei = amount_wei + fee_wei (reconstructed)
  - Broadcaster gas fee: non-deterministic, approximated as rolling mean × 1.10
    (see MonthlyFeeSchedule in thesis_kanan); here we use a per-transaction
    estimate of 1.10 × transaction_fee_eth as a single-value correction.

Candidate pairs: all (shield, unshield) pairs where:
  - unshield_time > shield_time
  - time gap ≤ WINDOW_DAYS (default 95, matching Ali's analysis)
  - both are WETH, non-protocol-address, non-swap

Monte Carlo null model: same count of random (shield, unshield) pairs
  drawn without replacement from the cross-product, ignoring time order.

Outputs (in Figures/data/ relative to repo root):
  hamming_fee_compensated_rg.csv   columns: dist, raw_count, proto_count, broad_count, mc_count
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ── path config ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, FIGURES as _FIGURES

TX_DIR = _RG_DATA.parent / "data" / "transactions"
OUT    = _FIGURES / "data" / "hamming_fee_compensated_rg.csv"

WINDOW_DAYS  = 95          # temporal window (same as Ali's thesis)
MC_SEED      = 42
MC_SAMPLE    = 500_000     # random pairs for null model
MAX_PAIRS    = 5_000_000   # cap to avoid OOM on the full cross-product

PROTOCOL_ADDRS = frozenset({
    "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9",
    "0xac9f360ae85469b27aeddeafc579ef2d052ad405",
    "0x4025ee6512dbbda97049bcf5aa5d38c54af6be8a",
    "0xe8a8b458bcd1ececc6b6b58f80929b29ccecff40",
    "0x22af4edbea3de885dda8f0a0653e6209e44e5b84",
    "0xc3f2c8f9d5f0705de706b1302b7a039e1e11ac88",
    "0x0000000000000000000000000000000000000000",
})

BROADCASTER_MARKUP = 1.10   # empirical: broadcaster charges ~10% above gas cost


# ── helpers ──────────────────────────────────────────────────────────────────

def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def wei_to_frac18_strings(wei_list) -> np.ndarray:
    """Convert a list of Python ints (wei) to (N, 18) uint8 digit array.
    Uses string conversion for arbitrary-precision correctness."""
    MOD = 10 ** 18
    strs = [f"{int(w) % MOD:018d}" for w in wei_list]
    # Stack into character array, convert digit chars to int
    arr = np.frombuffer("".join(strs).encode(), dtype=np.uint8).reshape(len(strs), 18)
    return arr - ord('0')   # ASCII '0'=48 → digit 0


def pairwise_hamming(d_digits: np.ndarray, w_digits: np.ndarray) -> np.ndarray:
    """Hamming distance between each row of d_digits and w_digits (same length)."""
    return (d_digits != w_digits).sum(axis=1).astype(np.int32)


def frac18_array(wei_arr_obj: np.ndarray) -> np.ndarray:
    """Wrapper: accepts object-dtype numpy array of Python ints."""
    return wei_to_frac18_strings(wei_arr_obj.tolist())


# ── load data ────────────────────────────────────────────────────────────────

print("Loading shields …")
sh = pd.read_csv(TX_DIR / "eth_shield.csv")
sh["_t"] = pd.to_datetime(sh["time"], utc=True, errors="coerce")
sh["_fr"] = norm(sh["from_address"])
sh = sh[
    (sh["token_symbol"].str.upper() == "WETH")
    & sh["_t"].notna()
    & ~sh["_fr"].isin(PROTOCOL_ADDRS)
].copy().reset_index(drop=True)
print(f"  {len(sh):,} shields after filtering")

print("Loading unshields …")
un = pd.read_csv(TX_DIR / "eth_unshields.csv")
un["_t"] = pd.to_datetime(un["time"], utc=True, errors="coerce")
un["_to"] = norm(un["to_address"])
un = un[
    (un["token_symbol"].str.upper() == "WETH")
    & un["_t"].notna()
    & ~un["_to"].isin(PROTOCOL_ADDRS)
].copy().reset_index(drop=True)
print(f"  {len(un):,} unshields after filtering")

# ── fee-aware wei columns ─────────────────────────────────────────────────────
# Use Python-object dtype to handle values > int64 max (e.g. 25 ETH = 2.5e19 wei)
# Shield side: deposit_wei is already the post-0.25%-fee note value
sh_note_wei = sh["deposit_wei"].astype(object).tolist()

# Unshield side:
#   RAW    → use raw amount_wei (what Ali does)
#   PROTO  → note_wei = amount_wei + fee_wei  (protocol fee recovered)
#   BROAD  → note_wei + broadcaster_estimate
un_raw_wei   = un["amount_wei"].astype(object).tolist()
un_proto_wei = un["note_wei"].astype(object).tolist()   # already = amount_wei + fee_wei
# Broadcaster estimate: 1.10 × gas_cost in wei
gas_cost_eth = un["transaction_fee_eth"].to_numpy(dtype=np.float64)
un_broad_wei = [int(p) + int(BROADCASTER_MARKUP * g * 1e18)
                for p, g in zip(un_proto_wei, gas_cost_eth)]

sh_ts = sh["_t"].to_numpy(dtype="datetime64[ns]")
un_ts = un["_t"].to_numpy(dtype="datetime64[ns]")
sh_note_arr = np.array(sh_note_wei, dtype=object)
un_raw_arr   = np.array(un_raw_wei,   dtype=object)
un_proto_arr = np.array(un_proto_wei, dtype=object)
un_broad_arr = np.array(un_broad_wei, dtype=object)

# ── build candidate pairs within time window ──────────────────────────────────
print(f"\nBuilding candidate pairs (window ≤ {WINDOW_DAYS} days) …")
WINDOW_NS = int(WINDOW_DAYS * 86_400 * 1e9)

# Sort shields by time for efficient searchsorted
sh_order  = np.argsort(sh_ts)
sh_ts_s   = sh_ts[sh_order]
sh_note_s = sh_note_arr[sh_order]

pair_d_idx = []
pair_w_idx = []

for wi in range(len(un)):
    wt = un_ts[wi]
    # shields that occurred before this unshield
    lo = int(np.searchsorted(sh_ts_s, wt - np.timedelta64(WINDOW_NS, 'ns'), side='left'))
    hi = int(np.searchsorted(sh_ts_s, wt, side='left'))
    if lo >= hi:
        continue
    for di_sorted in range(lo, hi):
        pair_d_idx.append(sh_order[di_sorted])
        pair_w_idx.append(wi)
    if len(pair_d_idx) > MAX_PAIRS:
        print(f"  Reached MAX_PAIRS cap ({MAX_PAIRS:,}) — truncating")
        break

pair_d_idx = np.array(pair_d_idx, dtype=np.int64)
pair_w_idx = np.array(pair_w_idx, dtype=np.int64)
print(f"  {len(pair_d_idx):,} candidate pairs")

# ── compute Hamming distances ─────────────────────────────────────────────────
print("Computing Hamming distances …")

d_dig  = frac18_array(sh_note_arr[pair_d_idx])

w_raw_dig   = frac18_array(un_raw_arr[pair_w_idx])
w_proto_dig = frac18_array(un_proto_arr[pair_w_idx])
w_broad_dig = frac18_array(un_broad_arr[pair_w_idx])

ham_raw   = pairwise_hamming(d_dig, w_raw_dig)
ham_proto = pairwise_hamming(d_dig, w_proto_dig)
ham_broad = pairwise_hamming(d_dig, w_broad_dig)

# ── Monte Carlo null model ───────────────────────────────────────────────────
print(f"Monte Carlo null model ({MC_SAMPLE:,} random pairs, seed={MC_SEED}) …")
rng = np.random.default_rng(MC_SEED)
mc_d = rng.integers(0, len(sh), size=MC_SAMPLE)
mc_w = rng.integers(0, len(un), size=MC_SAMPLE)
mc_d_dig = frac18_array(sh_note_arr[mc_d])
mc_w_dig = frac18_array(un_proto_arr[mc_w])    # use proto for null too
ham_mc = pairwise_hamming(mc_d_dig, mc_w_dig)

# ── aggregate ────────────────────────────────────────────────────────────────
print("Aggregating …")
DMAX = 18
rows = []
for d in range(DMAX + 1):
    rows.append({
        "dist":        d,
        "raw_count":   int((ham_raw   == d).sum()),
        "proto_count": int((ham_proto == d).sum()),
        "broad_count": int((ham_broad == d).sum()),
        "mc_count":    int((ham_mc    == d).sum()),
    })
df = pd.DataFrame(rows)

# Normalise MC to same total as proto_count (for visual comparison)
total_obs = df["proto_count"].sum()
total_mc  = df["mc_count"].sum()
df["mc_scaled"] = (df["mc_count"] * total_obs / total_mc).round().astype(int)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\nWritten → {OUT}")

# ── summary stats ────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
for label, col in [("RAW  ", "raw_count"), ("PROTO", "proto_count"), ("BROAD", "broad_count")]:
    n0 = int(df.loc[df["dist"] == 0, col].iloc[0])
    n_le2 = int(df.loc[df["dist"] <= 2, col].sum())
    total = int(df[col].sum())
    mc0 = int(df.loc[df["dist"] == 0, "mc_scaled"].iloc[0])
    ratio = n0 / mc0 if mc0 > 0 else float("inf")
    print(f"  {label}  d=0: {n0:>8,}  (MC null scaled: {mc0:>8,}  ratio: {ratio:.1f}×)"
          f"   d≤2: {n_le2:>8,}  total: {total:>10,}")
