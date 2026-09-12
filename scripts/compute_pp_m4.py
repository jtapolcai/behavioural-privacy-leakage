"""Compute PP M4 canonical CSV.

For each withdrawal computes M4 entropy variants using knapsack (amount-matching)
participation evidence from k=1, k=2, k=3 sum exports.

Deposits are identified by (address, timestamp) since the processed deposit CSV
has no transaction hash column; the knapsack score files carry dep_time which
is matched at 6-decimal-day precision.

M4 mixes the M2 temporal distribution with the M2-weighted participation
distribution for targets that have at least one knapsack match:

  H4(p_cov) = entropy of  alpha * P_knap + (1-alpha) * P_m2
  alpha      = p_cov / p_feas    (for feasible targets)
  H_knap     = entropy of M2 restricted to participating deposits

Infeasible targets fall back to H2.

Outputs: entropy_models/pp_m4_canonical.csv
"""
import argparse, csv, io, sys, zipfile
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scripts',    type=Path, required=True)
ap.add_argument('--source',     type=Path, required=True)
ap.add_argument('--output-dir', type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.scripts))

from estimate_pp_origin_entropy import day, SCALE
from estimate_pp_entropy import decimal_amount

OUT = a.output_dir / 'entropy_models'
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv_file(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def read_csv_from_bytes(data):
    return list(csv.DictReader(io.TextIOWrapper(io.BytesIO(data), encoding='utf-8-sig')))

def unique_rows(rows):
    return list({tuple(r.items()): r for r in rows}.values())

def day_key(t_str):
    return round(day(t_str), 6)

# ── load PP processed data ────────────────────────────────────────────────────

base = a.source / 'data' / 'processed'
dr_raw = unique_rows(read_csv_file(base / 'processed_privacypools_eth_pool_deposits.csv'))
wr_raw = unique_rows(read_csv_file(base / 'processed_privacypools_data_withdraws.csv'))

# deposits_full: sorted list of (time_float, amount_int, addr)
deposits_full = []
for r in dr_raw:
    amt = decimal_amount(r['amount_eth'])
    if amt <= 0:
        continue
    t = day(r['time'])
    v = int((amt * SCALE).to_integral_value())
    deposits_full.append((t, v, r['depositor'].lower()))
deposits_full.sort()

# withdrawals: aggregated by tx_hash, sorted by time
agg: dict[str, list] = {}
for r in wr_raw:
    t  = day(r['evt_block_time'])
    a_w = decimal_amount(r['eth_amount'])
    if a_w <= 0:
        raise ValueError('Nonpositive withdrawal')
    h = r['tx_hash'].lower()
    rec = (r.get('recipient') or '').lower()
    if h in agg:
        assert agg[h][0] == t
        agg[h][1] += a_w
    else:
        agg[h] = [t, a_w, rec, h]

end = min(deposits_full[-1][0], max(v[0] for v in agg.values()))
withdrawals_sorted = sorted(
    (t, int((a_w * SCALE).to_integral_value()), rec, tx_h)
    for t, a_w, rec, tx_h in agg.values() if t <= end
)
deposits_full = [(t, v, addr) for t, v, addr in deposits_full if t <= end]

dt    = np.array([t    for t, v, addr in deposits_full])
da    = np.array([v    for t, v, addr in deposits_full], dtype=float)
addrs = [addr for t, v, addr in deposits_full]
wt    = np.array([t for t, v, rec, tx_h in withdrawals_sorted])
wa    = np.array([v for t, v, rec, tx_h in withdrawals_sorted], dtype=float)
ci    = np.cumsum(da)
co    = np.cumsum(wa)

pi = np.array([0.065, 0.349, 0.586])
mu = np.array([0.09,  8.79,  246.0])

# ── index deposit by (addr, day_key) → list of indices in deposits_full ───────

dep_key_to_idxs: dict[tuple, list[int]] = {}
for j, (t, v, addr) in enumerate(deposits_full):
    k = (addr, round(t, 6))
    dep_key_to_idxs.setdefault(k, []).append(j)

# withdrawal tx_hash → sorted index
wh_to_idx: dict[str, int] = {
    tx_h.lower(): i for i, (_, _, _, tx_h) in enumerate(withdrawals_sorted)
}

# ── build knapsack participation map ──────────────────────────────────────────
# dep_hash → (dep_addr, dep_time_key)  from score files

knap_dir = a.source / 'results' / 'data' / 'Heuristics' / 'Knapsack'

def build_dep_hash_map(score_rows):
    """dep_hash → (addr_lower, day_key_float)"""
    return {
        r['dep_hash'].lower(): (r['dep_address'].lower(), day_key(r['dep_time']))
        for r in score_rows
    }

def register_participation(dep_hash: str, w_hashes: list[str],
                            dep_hash_map: dict, participation: dict):
    info = dep_hash_map.get(dep_hash.lower())
    if info is None:
        return
    dep_idxs = dep_key_to_idxs.get(info, [])
    for w_h in w_hashes:
        w_i = wh_to_idx.get(w_h.lower())
        if w_i is not None:
            for d_i in dep_idxs:
                participation.setdefault(w_i, set()).add(d_i)

participation: dict[int, set[int]] = {}

# k=1
score1 = read_csv_file(knap_dir / '1-sum' / 'knapsack_1_withdraw_s_score.csv')
dep_map1 = build_dep_hash_map(score1)
for row in read_csv_file(knap_dir / '1-sum' / 'knapsack_1_withdraw_matches.csv'):
    if float(row.get('difference', 1)) != 0:
        continue
    register_participation(row['dep_hash'], [row['w1_hash']], dep_map1, participation)
print(f"  k=1: {len(participation)} withdrawals with participation")

# k=2
k2_score_path = knap_dir / '2-sum' / 'knapsack_2_withdraws_s_score.csv'
k2_zip_path   = knap_dir / '2-sum' / 'knapsack_2_withdraws_matches.zip'
if k2_score_path.exists() and k2_zip_path.exists():
    score2   = read_csv_file(k2_score_path)
    dep_map2 = build_dep_hash_map(score2)
    with zipfile.ZipFile(k2_zip_path) as z:
        rows2 = read_csv_from_bytes(z.read(z.namelist()[0]))
    for row in rows2:
        if float(row.get('difference', 1)) != 0:
            continue
        register_participation(row['dep_hash'], [row['w1_hash'], row['w2_hash']],
                               dep_map2, participation)
    print(f"  k=2: {len(participation)} withdrawals with participation")

# k=3
k3_score_path = knap_dir / '3-sum' / 'knapsack_3_withdraws_s_score.csv'
k3_zip_path   = knap_dir / '3-sum' / 'knapsack_3_withdraws_matches.zip'
if k3_score_path.exists() and k3_zip_path.exists():
    score3   = read_csv_file(k3_score_path)
    dep_map3 = build_dep_hash_map(score3)
    with zipfile.ZipFile(k3_zip_path) as z:
        rows3 = read_csv_from_bytes(z.read(z.namelist()[0]))
    for row in rows3:
        if float(row.get('difference', 1)) != 0:
            continue
        register_participation(row['dep_hash'],
                               [row['w1_hash'], row['w2_hash'], row['w3_hash']],
                               dep_map3, participation)
    print(f"  k=3: {len(participation)} withdrawals with participation")

p_feas = len(participation) / len(withdrawals_sorted)
print(f"\n  p_feas = {p_feas:.4f}  ({len(participation)}/{len(withdrawals_sorted)} targets)")

# ── M2 weight helper ──────────────────────────────────────────────────────────

def m2_weights(nd, nw, t, amount):
    rho = float(np.clip(
        (ci[nd-1] - (co[nw-1] if nw else 0)) / ci[nd-1], 0, 1))
    ages = t - dt[:nd]
    f = sum(p / m * np.exp(-ages / m) for p, m in zip(pi, mu))
    small = da[:nd] <= amount
    note_w = np.zeros(nd)
    for mask, mass in ((small, 1 - rho), (~small, rho)):
        if mask.any() and mass > 0:
            note_w[np.where(mask)[0]] = mass * f[mask] / f[mask].sum()
    s = note_w.sum()
    return note_w / s if s > 0 else None

def h_of(p):
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def mix_entropy(note_w, knap_idxs, alpha):
    knap_mask = np.zeros(len(note_w), dtype=bool)
    knap_mask[list(knap_idxs)] = True
    if not knap_mask.any():
        return None
    knap_w = note_w.copy()
    knap_w[~knap_mask] = 0.0
    s = knap_w.sum()
    if s == 0:
        return None
    knap_w /= s
    p_mix = alpha * knap_w + (1.0 - alpha) * note_w
    p_mix /= p_mix.sum()
    return h_of(p_mix)

# ── main loop ─────────────────────────────────────────────────────────────────

rows = []
for i, (t, amount, rec, tx_h) in enumerate(withdrawals_sorted):
    nd = int(np.searchsorted(dt, t))
    nw = int(np.searchsorted(wt, t))
    if nd == 0:
        continue

    note_w = m2_weights(nd, nw, t, amount)
    if note_w is None:
        continue

    h2 = h_of(note_w)
    feasible = False
    n_knap = 0
    h4_p05 = h4_pfeas = h4_p025 = alpha05 = alpha_pfeas = None

    if i in participation and p_feas > 0:
        knap_in_range = {j for j in participation[i] if j < nd}
        if knap_in_range:
            feasible     = True
            n_knap       = len(knap_in_range)
            alpha05      = min(0.5  / p_feas, 1.0)
            alpha_pfeas  = min(1.0  / p_feas, 1.0)   # 1 for all feasible targets
            alpha025     = min(0.25 / p_feas, 1.0)

            h4_p05   = mix_entropy(note_w, knap_in_range, alpha05)
            h4_pfeas = mix_entropy(note_w, knap_in_range, alpha_pfeas)
            # p_cov=0.25: concavity lower bound
            if h4_p05 is not None:
                h4_p025 = 0.5 * h2 + 0.5 * h4_p05

    rows.append({
        'withdrawal_index':    i,
        'time_day':            round(t, 8),
        'recipient':           rec,
        'F':                   int(feasible),
        'H2':                  h2,
        'n_knap_deps':         n_knap,
        'alpha_w_pcov05':      '' if alpha05     is None else round(alpha05,     6),
        'alpha_w_pcov_pfeas':  '' if alpha_pfeas  is None else round(alpha_pfeas,  6),
        'H4_pcov05':           '' if h4_p05    is None else h4_p05,
        'H4_pcov_pfeas':       '' if h4_pfeas   is None else h4_pfeas,
        'H4_pcov025':          '' if h4_p025   is None else h4_p025,
    })

# ── summary ───────────────────────────────────────────────────────────────────

def med_col(col):
    vals = [float(r[col]) for r in rows if str(r[col]) not in ('', 'None')]
    return float(np.median(vals)) if vals else None

n_feasible = sum(r['F'] for r in rows)
print(f"\n  n_valid     : {len(rows)}")
print(f"  n_feasible  : {n_feasible}  ({100*n_feasible/len(rows):.1f}%)")
print(f"  Median H2           : {med_col('H2'):.4f}")
for label, col in [('H4(p=0.25)', 'H4_pcov025'),
                    ('H4(p=0.50)', 'H4_pcov05'),
                    ('H4(p=pfeas)', 'H4_pcov_pfeas')]:
    v = med_col(col)
    if v is not None:
        print(f"  Median {label}  : {v:.4f}")

# ── write ─────────────────────────────────────────────────────────────────────

out_path = OUT / 'pp_m4_canonical.csv'
fields = ['withdrawal_index','time_day','recipient','F','H2','n_knap_deps',
          'alpha_w_pcov05','alpha_w_pcov_pfeas',
          'H4_pcov05','H4_pcov_pfeas','H4_pcov025']
with out_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"\n  Written {len(rows):,} rows → {out_path}")
