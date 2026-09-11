#!/usr/bin/env python3
"""
generate_dataset_table.py
=========================
Computes event-count breakdown for the dataset-overview table
(Railgun vs Privacy Pools × asset × before/after 2023 × deposit/withdrawal).

Outputs
-------
Figures/table_dataset_overview.csv   — one row per (system, asset, period)
                                       plus printed LaTeX-ready numbers.

Input files (absolute paths; update DATA_DIR if your layout differs)
-----------------------------------------------------------------------
Railgun WETH weekly counts:
  Figures/figure_ch4_02_weekly_boundary_counts.csv   [week_start, shields, unshields]
Railgun total inventory:
  Figures/figure_ch4_01_dataset_inventory.csv         [label, color, rows, start, end]
Privacy Pools deposits:
  <PP_DATA>/processed_privacypools_eth_pool_deposits.csv   [amount_eth, depositor, time]
Privacy Pools withdrawals:
  <PP_DATA>/processed_privacypools_data_withdraws.csv      [eth_amount, evt_block_time, ...]

Usage
-----
  python3 generate_dataset_table.py [--pp-data PATH]
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
FIG    = _FIGURES  # via _paths
PP_DIR_DEFAULT = (
    _PP_DATA
)

CUTOFF = datetime(2023, 1, 1)
LABEL_BEFORE = "Before 2023"
LABEL_AFTER  = "After 2023"

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f UTC",
                "%Y-%m-%d %H:%M:%S,000 UTC"):
        try:
            return datetime.strptime(s.split(".")[0].split(",")[0].strip(), fmt[:len(s.split(".")[0])])
        except ValueError:
            pass
    # fallback: just grab the year-month-day prefix
    return datetime.strptime(s[:10], "%Y-%m-%d")

def period(dt: datetime) -> str:
    return LABEL_BEFORE if dt < CUTOFF else LABEL_AFTER

# ── Railgun WETH (weekly boundary counts) ─────────────────────────────────────
rg_weth = {LABEL_BEFORE: {"D": 0, "W": 0}, LABEL_AFTER: {"D": 0, "W": 0}}
with open(FIG / "figure_ch4_02_weekly_boundary_counts.csv") as f:
    for row in csv.DictReader(f):
        p = period(parse_date(row["week_start"]))
        rg_weth[p]["D"] += int(row["shields"])
        rg_weth[p]["W"] += int(row["unshields"])

# ── Railgun totals (all assets, from inventory) ────────────────────────────────
rg_all = {"D": 0, "W": 0}
with open(FIG / "figure_ch4_01_dataset_inventory.csv") as f:
    for row in csv.DictReader(f):
        if "Shield" in row["label"] and "Unshield" not in row["label"]:
            rg_all["D"] += int(row["rows"])
        elif "Unshield" in row["label"]:
            rg_all["W"] += int(row["rows"])

# Railgun ERC-20 = total - WETH
rg_weth_total = {k: rg_weth[LABEL_BEFORE][k] + rg_weth[LABEL_AFTER][k]
                 for k in ("D", "W")}
rg_erc20_total = {k: rg_all[k] - rg_weth_total[k] for k in ("D", "W")}

# ── Privacy Pools ETH deposits ────────────────────────────────────────────────
pp_dep = {LABEL_BEFORE: 0, LABEL_AFTER: 0}
with open(PP_DIR_DEFAULT / "processed_privacypools_eth_pool_deposits.csv") as f:
    for row in csv.DictReader(f):
        t_str = row.get("time", "")
        if t_str:
            pp_dep[period(parse_date(t_str))] += 1

# ── Privacy Pools ETH withdrawals ─────────────────────────────────────────────
pp_wit = {LABEL_BEFORE: 0, LABEL_AFTER: 0}
with open(PP_DIR_DEFAULT / "processed_privacypools_data_withdraws.csv") as f:
    for row in csv.DictReader(f):
        t_str = row.get("evt_block_time", "")
        if t_str:
            pp_wit[period(parse_date(t_str))] += 1

# ── assemble table ─────────────────────────────────────────────────────────────
rows = [
    # (system, asset, before_D, before_W, after_D, after_W, total_D, total_W, note)
    ("Railgun", "WETH",
     rg_weth[LABEL_BEFORE]["D"], rg_weth[LABEL_BEFORE]["W"],
     rg_weth[LABEL_AFTER]["D"],  rg_weth[LABEL_AFTER]["W"],
     rg_weth_total["D"],         rg_weth_total["W"],
     "WETH-filtered population used in H4/H5 analysis"),
    ("Railgun", "Other ERC-20",
     None, None,          # before 2023 split not tracked separately
     None, None,
     rg_erc20_total["D"], rg_erc20_total["W"],
     "Derived: inventory total minus WETH; pre-2023 split not available"),
    ("Privacy Pools", "ETH",
     pp_dep[LABEL_BEFORE], pp_wit[LABEL_BEFORE],
     pp_dep[LABEL_AFTER],  pp_wit[LABEL_AFTER],
     pp_dep[LABEL_BEFORE] + pp_dep[LABEL_AFTER],
     pp_wit[LABEL_BEFORE] + pp_wit[LABEL_AFTER],
     "ETH pool only; launched 2023"),
    ("Privacy Pools", "ERC-20",
     None, None, None, None, None, None,
     "Not deployed"),
]

# ── write CSV ─────────────────────────────────────────────────────────────────
out_csv = FIG / "table_dataset_overview.csv"
fieldnames = ["system", "asset",
              "before2023_D", "before2023_W",
              "after2023_D",  "after2023_W",
              "total_D",      "total_W",
              "note"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(dict(zip(fieldnames, r)))
print(f"Written: {out_csv}")

# ── print LaTeX-ready numbers ──────────────────────────────────────────────────
def fmt(n):
    return r"\num{" + f"{n:,}" + "}" if n is not None else r"---"

print("\n% ── LaTeX numbers for table_dataset_overview ──────────────────────────")
for r in rows:
    sys_, asset, bd, bw, ad, aw, td, tw, note = r
    print(f"% {sys_} / {asset}:")
    print(f"%   before 2023: D={bd}, W={bw}")
    print(f"%   after  2023: D={ad}, W={aw}")
    print(f"%   total:       D={td}, W={tw}  ({note})")
    print()
