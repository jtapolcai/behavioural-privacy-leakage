"""
compute_pp_m3.py
================
Computes the canonical M3 origin entropy for Privacy Pools and writes
the unified per-withdrawal CSV used by compute_table2.py.

Evidence signals:
  AR  – deposit address == withdrawal recipient (inline)
  GR  – deposit address == withdrawal relayer (inline from withdrawal data)
  PT  – direct transaction link (from transaction_link_leaks.csv)

Output: Figures/entropy_models/pp_m3_canonical.csv
Columns (unified format, identical to rg_m3_canonical.csv):
  withdrawal_id, n_ar, n_gr, n_pt,
  H2, H3_ar, H3_ar_gr, H3_ar_gr_pt10, H3_ar_gr_ptinf
"""

import argparse, csv, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scripts",    type=Path, required=True)
ap.add_argument("--source",     type=Path, required=True,
                help="root of privacypools-deanonymization repo")
ap.add_argument("--output-dir", type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.scripts))

from estimate_pp_origin_entropy import day, SCALE
from estimate_pp_entropy import decimal_amount
from _scenario_params import PP_PI, PP_MU
from models import (DepositData, WithdrawalData, Evidence,
                    compute_m2, compute_m3, mean_finite)

OUT = a.output_dir / "entropy_models"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA_P      = 10.0
_SECS_PER_DAY = 86_400.0
_SCALE_F      = float(SCALE)   # conversion: SCALE units → ETH


# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def unique_rows(rows):
    return list({tuple(r.items()): r for r in rows}.values())


# ── load PP raw data ──────────────────────────────────────────────────────────

base = a.source / "data" / "processed"
dr_raw = unique_rows(read_csv(base / "processed_privacypools_eth_pool_deposits.csv"))
wr_raw = unique_rows(read_csv(base / "processed_privacypools_data_withdraws.csv"))

# Deposits
deposits_raw = []
for r in dr_raw:
    amt = decimal_amount(r["amount_eth"])
    if amt <= 0:
        continue
    deposits_raw.append((
        day(r["time"]),
        float(amt),
        r["depositor"].lower(),
    ))
deposits_raw.sort()

# Withdrawals — aggregate by tx_hash (multiple rows per note)
agg: dict = {}
for r in wr_raw:
    t   = day(r["evt_block_time"])
    a_w = decimal_amount(r["eth_amount"])
    if a_w <= 0:
        continue
    h   = r["tx_hash"]
    rec = (r.get("recipient") or "").lower()
    rel = (r.get("relayer")   or "").lower()
    if h in agg:
        assert agg[h][0] == t
        agg[h][1] += a_w
    else:
        agg[h] = [t, a_w, rec, rel]

# shared end cutoff
end = min(deposits_raw[-1][0], max(v[0] for v in agg.values()))
deposits_raw = [(t, a, addr) for t, a, addr in deposits_raw if t <= end]
withdrawals_raw = sorted(
    (t, float(a_w), rec, rel)
    for t, a_w, rec, rel in agg.values() if t <= end
)

# Convert times from days → seconds (internal unit)
dep_t_s   = np.array([t * _SECS_PER_DAY  for t, _, _     in deposits_raw])
dep_eth   = np.array([a                   for _, a, _     in deposits_raw])
dep_addr  = np.array([addr                for _, _, addr  in deposits_raw])
dep_agg   = np.arange(len(deposits_raw))  # synthetic integer id

wit_t_s   = np.array([t * _SECS_PER_DAY  for t, _, _, _  in withdrawals_raw])
wit_eth   = np.array([a                   for _, a, _, _  in withdrawals_raw])
wit_addr  = np.array([rec                 for _, _, rec, _ in withdrawals_raw])
wit_rel   = np.array([rel                 for _, _, _, rel in withdrawals_raw])
wit_agg   = np.arange(len(withdrawals_raw))  # withdrawal_id = row index

deps = DepositData.from_arrays(dep_agg, dep_t_s, dep_addr, dep_eth)
wits = WithdrawalData.from_arrays(wit_agg, wit_t_s, wit_addr, wit_eth)

print(f"  {len(dep_t_s):,} deposits, {len(wit_t_s):,} withdrawals")


# ── GR evidence (relayer → deposit address match, inline) ────────────────────

# Build relayer address → set of deposit agg_ids (only non-empty relayers)
rel_to_dep_ids: dict[str, set[int]] = {}
for aid, addr in zip(deps.agg_id.tolist(), deps.addr.tolist()):
    rel_to_dep_ids.setdefault(addr, set()).add(int(aid))

gr_evidence: dict[int, set[int]] = {}
for i, rel in enumerate(wit_rel.tolist()):
    if rel:
        gr_evidence[i] = rel_to_dep_ids.get(rel, set())

# ── PT evidence (direct tx link) ─────────────────────────────────────────────

tx_link_path = (a.source / "results" / "data" / "Heuristics" /
                "Direct Tx Link" / "transaction_link_leaks.csv")
pt_links: set[tuple[str, str]] = set()
if tx_link_path.exists():
    for row in read_csv(tx_link_path):
        dep_addr_s = row["deposit_address"].lower()
        for wa in row["withdraw_addresses"].split(","):
            pt_links.add((dep_addr_s, wa.strip().lower()))
    print(f"  PT links: {len(pt_links):,}")
else:
    print(f"  Warning: {tx_link_path.name} not found — PT evidence empty")

# Build evidence dict: {withdrawal_id → {gr: set, pt: set}}
# PT: all deposits whose (deposit_addr, withdrawal_recipient) is a known link
evidence: Evidence = {}
for i, rec in enumerate(wit_addr.tolist()):
    gr_ids = gr_evidence.get(i, set())
    pt_ids = {int(aid) for aid, addr in zip(deps.agg_id.tolist(), deps.addr.tolist())
              if (addr, rec) in pt_links}
    if gr_ids or pt_ids:
        evidence[i] = {"gr": gr_ids, "pt": pt_ids}


# ── compute M2 then M3 ────────────────────────────────────────────────────────

print("Computing M2 …")
_, m2_dict = compute_m2(deps, wits, PP_PI, PP_MU,
                        use_rho_rem=True, _progress_every=1_000)
print(f"  M2 mean: {mean_finite(list(m2_dict.values())):.4f} bits")

print("Computing M3 …")
m3_df = compute_m3(deps, wits, m2_dict, evidence,
                   PP_PI, PP_MU, alpha_p=ALPHA_P, use_rho_rem=True)

for col in ["H3_ar", "H3_ar_gr", "H3_ar_gr_pt10", "H3_ar_gr_ptinf"]:
    print(f"  {col:<22}: {mean_finite(m3_df[col]):.4f}  "
          f"(AR cov {(m3_df['n_ar']>0).mean()*100:.1f}%)")

out_path = OUT / "pp_m3_canonical.csv"
m3_df.to_csv(out_path, index=False)
print(f"\nWritten {len(m3_df):,} rows → {out_path}")
