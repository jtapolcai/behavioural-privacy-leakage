#!/usr/bin/env python3
"""Aggregate the PP export's relayer field; do not infer gas-payer identity."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_PP_DATA / "processed_privacypools_data_withdraws.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "Figures")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.source.open(newline="")))
    groups = defaultdict(lambda: {"recipients": set(), "volume": Decimal(0), "rows": 0})
    times = []
    for row in rows:
        if row["flow_type"] != "Withdrawal":
            raise ValueError("Unexpected flow type")
        address, recipient = row["relayer"].strip().lower(), row["recipient"].strip().lower()
        if not address or not recipient:
            raise ValueError("Missing address")
        amount = Decimal(row["eth_amount"].replace(",", "."))
        if not amount.is_finite() or amount < 0:
            raise ValueError("Invalid gross amount")
        group = groups[address]
        group["recipients"].add(recipient)
        group["volume"] += amount
        group["rows"] += 1
        times.append(datetime.strptime(row["evt_block_time"].replace(",", "."), "%Y-%m-%d %H:%M:%S.%f UTC"))
    if not rows or any(g["volume"] <= 0 for g in groups.values()):
        raise ValueError("Log plot requires positive group volumes")
    args.output_dir.mkdir(exist_ok=True, parents=True)
    # Plot IDs do not disclose flagged address pairs. Preserve source rows:
    # a transaction hash is not a unique withdrawal-event identifier.
    with (args.output_dir / "pp_broadcaster_points.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "n_unique_recipients", "total_eth_volume", "withdrawal_rows"])
        for i, (address, group) in enumerate(sorted(groups.items()), 1):
            writer.writerow([i, len(group["recipients"]), str(group["volume"]), group["rows"]])
    manifest = {
        "source": str(args.source.resolve()),
        "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "grouping": "export relayer field; not verified transaction sender",
        "volume": "sum eth_amount (gross ETH), decimal arithmetic",
        "rows": len(rows),
        "distinct_transaction_hashes": len({r["tx_hash"] for r in rows}),
        "groups": len(groups),
        "start_utc": min(times).isoformat(),
        "end_utc": max(times).isoformat(),
        "total_eth": str(sum((g["volume"] for g in groups.values()), Decimal(0))),
        "deduplication": "none: no event log index; repeated hashes retained",
    }
    (args.output_dir / "pp_broadcaster_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
