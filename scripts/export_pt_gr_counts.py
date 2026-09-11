#!/usr/bin/env python3
"""
export_pt_gr_counts.py
======================
Exports per-withdrawal-agg_id counts of PT and GR evidence.

PT (public transfer): a public on-chain ETH or ERC20 transaction exists
  between the withdrawal recipient address and some deposit address.
  Both directions count: deposit→withdraw and withdraw→deposit.

GR (gas-payer reuse): the gas payer (tx sender) of a withdrawal transaction
  matches a deposit address.  Self-broadcast withdrawals are included
  (gas_payer == withdrawal_recipient is a special case of address reuse,
  already captured by AR; we flag GR separately as it uses gas_payer field).

Output columns:
  withdrawal_agg_id, n_pt, n_gr,
  deposit_agg_ids_pt  (pipe-separated list, for audit),
  deposit_agg_ids_gr  (pipe-separated list, for audit)

Data sources (railgun_deanonymization-48A0):
  data/aggregated/eth_shield_aggregated.csv
  data/aggregated/eth_unshields_aggregated.csv
  data/transactions/eth_external2.csv   (ETH public transfers)
  data/transactions/eth_erc20.csv       (ERC20 public transfers)
  data/transactions/eth_unshields.csv   (raw unshields with gas_payer)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

# ── Paths ────────────────────────────────────────────────────────────────────
RAW   = _RG_DATA.parent  # via _paths
PAPER = Path(__file__).resolve().parents[1]
OUT   = PAPER / "Figures/entropy_models/rg_pt_gr_counts.csv"


# ── Relayer exclusion threshold ─────────────────────────────────────────────
# Gas payers appearing in >= RELAYER_THRESHOLD withdrawal transactions are
# treated as relayers and excluded from GR evidence.
# Rationale: a broadcaster submitting withdrawals for many users is unlikely
# to be the depositor; threshold=10 is conservative (12 addresses appear
# >1000x, 193 appear >10x).
RELAYER_THRESHOLD = 10


def norm(s: pd.Series) -> pd.Series:
    """Lowercase address strings."""
    return s.str.lower().str.strip()


# ── Load aggregated tables ───────────────────────────────────────────────────
print("Loading aggregated tables …")
sh_agg = pd.read_csv(RAW / "aggregated/eth_shield_aggregated.csv",
                     usecols=["agg_id", "address"])
sh_agg["address"] = norm(sh_agg["address"])
sh_agg = sh_agg.rename(columns={"agg_id": "deposit_agg_id",
                                  "address": "deposit_addr"})

un_agg = pd.read_csv(RAW / "aggregated/eth_unshields_aggregated.csv",
                     usecols=["agg_id", "address", "member_tx_hashes"])
un_agg["address"] = norm(un_agg["address"])
un_agg = un_agg.rename(columns={"agg_id": "withdrawal_agg_id",
                                  "address": "withdrawal_addr"})

# address → set of deposit agg_ids  (one address can appear in multiple agg groups)
dep_addr_to_agg: dict[str, set[int]] = {}
for _, row in sh_agg.iterrows():
    dep_addr_to_agg.setdefault(row["deposit_addr"], set()).add(row["deposit_agg_id"])

# address → set of withdrawal agg_ids
with_addr_to_agg: dict[str, set[int]] = {}
for _, row in un_agg.iterrows():
    with_addr_to_agg.setdefault(row["withdrawal_addr"], set()).add(row["withdrawal_agg_id"])

dep_addresses  = set(dep_addr_to_agg.keys())
with_addresses = set(with_addr_to_agg.keys())

print(f"  Shield agg: {len(sh_agg):,} rows, {len(dep_addresses):,} unique addresses")
print(f"  Unshield agg: {len(un_agg):,} rows, {len(with_addresses):,} unique addresses")


# ── PT: public transfer links ────────────────────────────────────────────────
print("\nComputing PT links …")

def load_tx(path: Path, addr_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=addr_cols + ["direction"])
    for c in addr_cols:
        df[c] = norm(df[c])
    return df

ext = load_tx(RAW / "transactions/eth_external2.csv",
              ["from_address", "to_address"])
erc = load_tx(RAW / "transactions/eth_erc20.csv",
              ["from_address", "to_address"])
txs = pd.concat([ext, erc], ignore_index=True)
print(f"  Public tx rows: {len(txs):,}  "
      f"(ETH {len(ext):,} + ERC20 {len(erc):,})")

# For each tx row, one address is a withdrawal, the other is a deposit.
# direction "withdraw→deposit": from=withdrawal, to=deposit
# direction "deposit→withdraw": from=deposit,    to=withdrawal
# We collect (withdrawal_addr, deposit_addr) pairs.

pt_pairs: list[tuple[str, str]] = []
for _, row in txs.iterrows():
    fa, ta, d = row["from_address"], row["to_address"], row["direction"]
    if d == "withdraw→deposit":
        # from=withdrawal recipient, to=deposit address
        if fa in with_addresses and ta in dep_addresses:
            pt_pairs.append((fa, ta))
    elif d == "deposit→withdraw":
        # from=deposit address, to=withdrawal recipient
        if ta in with_addresses and fa in dep_addresses:
            pt_pairs.append((ta, fa))

print(f"  Valid (withdrawal_addr, deposit_addr) PT pairs: {len(pt_pairs):,}")

# Expand to (withdrawal_agg_id, deposit_agg_id)
pt_links: dict[int, set[int]] = {}   # withdrawal_agg_id → set of deposit_agg_ids
for w_addr, d_addr in pt_pairs:
    for w_id in with_addr_to_agg.get(w_addr, set()):
        for d_id in dep_addr_to_agg.get(d_addr, set()):
            pt_links.setdefault(w_id, set()).add(d_id)

print(f"  Withdrawal agg_ids with ≥1 PT link: {len(pt_links):,}")


# ── GR: gas-payer reuse ──────────────────────────────────────────────────────
print("\nComputing GR links …")

un_raw = pd.read_csv(RAW / "transactions/eth_unshields.csv",
                     usecols=["transaction_hash", "gas_payer", "from_address"])
un_raw["gas_payer"]    = norm(un_raw["gas_payer"])
un_raw["from_address"] = norm(un_raw["from_address"])
un_raw["transaction_hash"] = un_raw["transaction_hash"].str.lower().str.strip()
print(f"  Raw unshield tx rows: {len(un_raw):,}")

# Identify and exclude relayers: gas_payers with >= RELAYER_THRESHOLD tx
gp_freq = un_raw["gas_payer"].value_counts()
relayer_addresses = set(gp_freq[gp_freq >= RELAYER_THRESHOLD].index)
print(f"  Relayer addresses excluded (>= {RELAYER_THRESHOLD} tx): "
      f"{len(relayer_addresses):,}")
un_raw_no_relayer = un_raw[~un_raw["gas_payer"].isin(relayer_addresses)].copy()
print(f"  Unshield rows after relayer exclusion: {len(un_raw_no_relayer):,}")

# Build tx_hash → gas_payer lookup (relayer-excluded)
hash_to_gaspayer: dict[str, str] = dict(
    zip(un_raw_no_relayer["transaction_hash"], un_raw_no_relayer["gas_payer"])
)

# Expand member_tx_hashes (semicolon-separated) for each unshield agg_id
# → collect gas_payers per agg_id
gr_links: dict[int, set[int]] = {}   # withdrawal_agg_id → set of deposit_agg_ids

for _, row in un_agg.iterrows():
    w_id   = int(row["withdrawal_agg_id"])
    hashes_raw = str(row["member_tx_hashes"]) if pd.notna(row["member_tx_hashes"]) else ""
    hashes = [h.strip().lower() for h in hashes_raw.split(";") if h.strip()]

    for tx_hash in hashes:
        gp = hash_to_gaspayer.get(tx_hash)
        if gp is None:
            continue
        # GR: gas_payer is a known deposit address
        for d_id in dep_addr_to_agg.get(gp, set()):
            gr_links.setdefault(w_id, set()).add(d_id)

print(f"  Withdrawal agg_ids with ≥1 GR link: {len(gr_links):,}")


# ── Assemble output ──────────────────────────────────────────────────────────
print("\nAssembling output …")

all_w_ids = un_agg["withdrawal_agg_id"].tolist()
records = []
for w_id in all_w_ids:
    pt_deps = sorted(pt_links.get(w_id, set()))
    gr_deps = sorted(gr_links.get(w_id, set()))
    records.append({
        "withdrawal_agg_id":  w_id,
        "n_pt":               len(pt_deps),
        "n_gr":               len(gr_deps),
        "deposit_agg_ids_pt": "|".join(map(str, pt_deps)),
        "deposit_agg_ids_gr": "|".join(map(str, gr_deps)),
    })

out_df = pd.DataFrame(records)

print(f"\n{'='*50}")
print(f"Total withdrawal agg_ids: {len(out_df):,}")
print(f"PT coverage: {(out_df['n_pt']>0).sum():,}  "
      f"({100*(out_df['n_pt']>0).mean():.2f}%)")
print(f"GR coverage: {(out_df['n_gr']>0).sum():,}  "
      f"({100*(out_df['n_gr']>0).mean():.2f}%)")
print(f"PT median n (when >0): "
      f"{out_df.loc[out_df['n_pt']>0,'n_pt'].median():.1f}")
print(f"GR median n (when >0): "
      f"{out_df.loc[out_df['n_gr']>0,'n_gr'].median():.1f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)
print(f"\nWritten → {OUT}")
