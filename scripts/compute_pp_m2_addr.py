"""Compute address-aggregated PP M2.

Same methodology as recompute_pp_m2.py but each unique depositor
address is one candidate entity (as in the RG address-level M2),
rather than each individual deposit note.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scripts', type=Path, required=True)
ap.add_argument('--source',  type=Path, required=True)
ap.add_argument('--output',  type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.scripts))

from estimate_pp_origin_entropy import day, SCALE
from estimate_pp_entropy import decimal_amount

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def unique_rows(rows):
    return list({tuple(r.items()): r for r in rows}.values())

base = a.source / 'data/processed'
dr_raw = unique_rows(read_csv(base / 'processed_privacypools_eth_pool_deposits.csv'))
wr_raw = unique_rows(read_csv(base / 'processed_privacypools_data_withdraws.csv'))

# Deposits WITH depositor address
deposits_full = []
for r in dr_raw:
    amt = decimal_amount(r['amount_eth'])
    if amt <= 0:
        continue
    deposits_full.append((
        day(r['time']),
        int((amt * SCALE).to_integral_value()),
        r['depositor']
    ))
deposits_full.sort()

# Withdrawals aggregated by tx_hash (same as load_events)
agg = {}
for r in wr_raw:
    t = day(r['evt_block_time'])
    a_w = decimal_amount(r['eth_amount'])
    if a_w <= 0:
        raise ValueError('Nonpositive withdrawal')
    h = r['tx_hash']
    if h in agg:
        assert agg[h][0] == t
        agg[h][1] += a_w
    else:
        agg[h] = [t, a_w]

# Shared-end cutoff (identical to load_events)
end = min(deposits_full[-1][0], max(v[0] for v in agg.values()))
withdrawals = sorted((t, int((a_w * SCALE).to_integral_value()))
                     for t, a_w in agg.values() if t <= end)
deposits_full = [(t, v, addr) for t, v, addr in deposits_full if t <= end]

dt    = np.array([t    for t, v, addr in deposits_full])
da    = np.array([v    for t, v, addr in deposits_full], dtype=float)
addrs = [addr for t, v, addr in deposits_full]

wt = np.array([t for t, v in withdrawals])
wa = np.array([v for t, v in withdrawals], dtype=float)

ci = np.cumsum(da)
co = np.cumsum(wa)

from _scenario_params import PP_PI as pi, PP_MU as mu  # declared scenario weights/scales

unique_addrs = sorted(set(addrs))
print(f"deposit records: {len(deposits_full)}, unique depositor addrs: {len(unique_addrs)}")
print(f"withdrawal transactions (cohort): {len(withdrawals)}")

results = []
for i, (t, amount) in enumerate(withdrawals):
    nd = int(np.searchsorted(dt, t))
    nw = int(np.searchsorted(wt, t))
    if nd == 0:
        continue

    rho = float(np.clip(
        (ci[nd-1] - (co[nw-1] if nw else 0)) / ci[nd-1], 0, 1))
    ages = t - dt[:nd]
    f = sum(p / m * np.exp(-ages / m) for p, m in zip(pi, mu))

    small = da[:nd] <= amount

    # Per-note weights (same retained-balance split as note-level M2)
    note_w = np.zeros(nd)
    for mask, mass in ((small, 1 - rho), (~small, rho)):
        if mask.any() and mass > 0:
            note_w[np.where(mask)[0]] = mass * f[mask] / f[mask].sum()

    if note_w.sum() == 0:
        continue
    note_w /= note_w.sum()

    # Aggregate note weights by depositor address
    addr_w: dict[str, float] = {}
    for j in range(nd):
        addr_w[addrs[j]] = addr_w.get(addrs[j], 0.0) + note_w[j]

    p = np.array(list(addr_w.values()))
    p /= p.sum()
    h = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    results.append({
        'withdrawal_sorted_index': i,
        'H2_addr': h,
        'n_addr_candidates': len(addr_w),
    })

median_addr = float(np.median([r['H2_addr'] for r in results]))
print(f"\nn_valid: {len(results)}")
print(f"Median H2 addr-level : {median_addr:.4f} bits")
print(f"Median H2 note-level : 9.2001 bits  (archived)")

out = {
    'cohort': {
        'deposit_records': len(deposits_full),
        'unique_depositor_addresses': len(unique_addrs),
        'withdrawal_transactions': len(withdrawals),
    },
    'pi': pi.tolist(), 'mu_days': mu.tolist(),
    'n_valid': len(results),
    'median': median_addr,
    'model': 'M2; retained-balance split; address-aggregated depositor prior',
    'per_withdrawal': results,
}
a.output.write_text(json.dumps(out, indent=2) + '\n')
print(f"Written to {a.output}")
