"""Compute PP M3 canonical CSV.

For each withdrawal computes M3 entropy variants:
  H3_ar        — restrict candidate set to AR deposits (recipient == depositor);
                 falls back to H2 if no AR match.
  H3_ar_gr     — restrict to AR ∪ GR deposits (GR: relayer == depositor);
                 falls back to H2 if no match.
  H3_ar_gr_pt10   — within-component PT boost (α_p=10) on top of AR+GR.
  H3_ar_gr_ptinf  — PT boost α_p→∞ (hard restriction to PT deposits if any exist
                    among AR+GR, else identical to H3_ar_gr).

PT (public-tag) evidence comes from the Direct Tx Link leaks CSV:
  deposit_address → directly linked to withdrawal address/tx.

Outputs: entropy_models/pp_m3_canonical.csv
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scripts', type=Path, required=True)
ap.add_argument('--source',  type=Path, required=True,
                help='root of privacypools-deanonymization repo')
ap.add_argument('--output-dir', type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.scripts))

from estimate_pp_origin_entropy import day, SCALE
from estimate_pp_entropy import decimal_amount

OUT = a.output_dir / 'entropy_models'
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def unique_rows(rows):
    return list({tuple(r.items()): r for r in rows}.values())

# ── load & deduplicate PP data ────────────────────────────────────────────────

base = a.source / 'data' / 'processed'
dr_raw = unique_rows(read_csv(base / 'processed_privacypools_eth_pool_deposits.csv'))
wr_raw = unique_rows(read_csv(base / 'processed_privacypools_data_withdraws.csv'))

# Deposits with depositor address
deposits_full = []
for r in dr_raw:
    amt = decimal_amount(r['amount_eth'])
    if amt <= 0:
        continue
    deposits_full.append((
        day(r['time']),
        int((amt * SCALE).to_integral_value()),
        r['depositor'].lower()
    ))
deposits_full.sort()

# Withdrawals aggregated by tx_hash, keeping recipient + relayer
agg = {}  # tx_hash → [time, amount_int, recipient, relayer]
for r in wr_raw:
    t  = day(r['evt_block_time'])
    a_w = decimal_amount(r['eth_amount'])
    if a_w <= 0:
        raise ValueError('Nonpositive withdrawal')
    h = r['tx_hash']
    rec = (r.get('recipient') or '').lower()
    rel = (r.get('relayer')   or '').lower()
    if h in agg:
        assert agg[h][0] == t
        agg[h][1] += a_w
    else:
        agg[h] = [t, a_w, rec, rel]

# shared-end cutoff
end = min(deposits_full[-1][0], max(v[0] for v in agg.values()))
withdrawals = sorted(
    (t, int((a_w * SCALE).to_integral_value()), rec, rel)
    for t, a_w, rec, rel in agg.values() if t <= end
)
deposits_full = [(t, v, addr) for t, v, addr in deposits_full if t <= end]

dt    = np.array([t    for t, v, addr in deposits_full])
da    = np.array([v    for t, v, addr in deposits_full], dtype=float)
addrs = [addr for t, v, addr in deposits_full]

wt = np.array([t for t, v, rec, rel in withdrawals])
wa = np.array([v for t, v, rec, rel in withdrawals], dtype=float)

ci = np.cumsum(da)
co = np.cumsum(wa)

from _scenario_params import PP_PI as pi, PP_MU as mu  # declared scenario weights/scales

# ── PT (direct tx link) evidence ─────────────────────────────────────────────
# Build set of (deposit_addr, withdraw_addr) pairs with a direct link

tx_link_path = (a.source / 'results' / 'data' / 'Heuristics' /
                'Direct Tx Link' / 'transaction_link_leaks.csv')
pt_links: set[tuple[str, str]] = set()
if tx_link_path.exists():
    for row in read_csv(tx_link_path):
        dep_addr = row['deposit_address'].lower()
        for wa_addr in row['withdraw_addresses'].split(','):
            pt_links.add((dep_addr, wa_addr.strip().lower()))
    print(f"  PT links loaded: {len(pt_links):,}")
else:
    print(f"  Warning: {tx_link_path} not found — PT evidence will be empty")

# ── per-withdrawal M2 weight helper ──────────────────────────────────────────

def m2_weights(nd, nw, t, amount):
    """Unnormalised per-note M2 weights for the nd deposits preceding time t."""
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

def restricted_entropy(note_w, mask):
    """Entropy of note_w restricted to mask; returns None if no mass."""
    sub = note_w[mask]
    if sub.sum() == 0:
        return None
    sub = sub / sub.sum()
    return float(-np.sum(sub[sub > 0] * np.log2(sub[sub > 0])))

def pt_boost_entropy(note_w, base_mask, pt_mask, alpha_p):
    """Within-component PT boost on top of base_mask restriction.

    Deposits in pt_mask get weight multiplied by alpha_p relative to
    other deposits in base_mask.  Falls back to restricted entropy if
    no PT deposit in base_mask.
    """
    sub_w = note_w.copy()
    in_base = base_mask & (note_w > 0)
    in_pt   = in_base & pt_mask
    if not in_pt.any():
        # No PT within the AR+GR set — identical to H3_ar_gr
        return restricted_entropy(note_w, base_mask)
    # Boost PT notes by alpha_p
    boosted = sub_w.copy()
    boosted[in_pt] *= alpha_p
    sub = boosted[in_base]
    sub = sub / sub.sum()
    return float(-np.sum(sub[sub > 0] * np.log2(sub[sub > 0])))

# ── main loop ─────────────────────────────────────────────────────────────────

rows = []
for i, (t, amount, rec, rel) in enumerate(withdrawals):
    nd = int(np.searchsorted(dt, t))
    nw = int(np.searchsorted(wt, t))
    if nd == 0:
        continue

    note_w = m2_weights(nd, nw, t, amount)
    if note_w is None:
        continue

    h2 = float(-np.sum(note_w[note_w > 0] * np.log2(note_w[note_w > 0])))

    # AR mask: depositor == withdrawal recipient
    ar_mask = np.array([addr == rec for addr in addrs[:nd]])
    n_ar = int(ar_mask.sum())

    # GR mask: depositor == relayer
    gr_mask = (np.array([addr == rel for addr in addrs[:nd]])
               if rel else np.zeros(nd, dtype=bool))
    n_gr = int(gr_mask.sum())

    # AR+GR union
    ar_gr_mask = ar_mask | gr_mask

    # PT mask: deposit has a direct tx link to this withdrawal's recipient
    pt_mask = np.array([(addrs[j], rec) in pt_links for j in range(nd)])
    n_pt = int(pt_mask.sum())

    # H3_ar: restrict to AR; fallback H2
    h3_ar = restricted_entropy(note_w, ar_mask) if n_ar > 0 else h2

    # H3_ar_gr: restrict to AR+GR; fallback H2
    h3_ar_gr = restricted_entropy(note_w, ar_gr_mask) if ar_gr_mask.any() else h2

    # H3_ar_gr_pt10: PT boost α=10 within AR+GR; fallback H3_ar_gr
    if ar_gr_mask.any():
        h3_pt10 = pt_boost_entropy(note_w, ar_gr_mask, pt_mask, alpha_p=10)
    else:
        h3_pt10 = h2

    # H3_ar_gr_ptinf: hard restrict to PT within AR+GR if any, else H3_ar_gr
    if ar_gr_mask.any():
        h3_ptinf = pt_boost_entropy(note_w, ar_gr_mask, pt_mask, alpha_p=1e9)
    else:
        h3_ptinf = h2

    rows.append({
        'withdrawal_index': i,
        'time_day':         round(t, 8),
        'recipient':        rec,
        'n_le':             int((da[:nd] <= amount).sum()),
        'n_gt':             int((da[:nd] >  amount).sum()),
        'n_ar':             n_ar,
        'n_gr':             n_gr,
        'n_pt':             n_pt,
        'H2':               h2,
        'H3_ar':            h3_ar,
        'H3_ar_gr':         h3_ar_gr,
        'H3_ar_gr_pt10':    h3_pt10,
        'H3_ar_gr_ptinf':   h3_ptinf,
    })

# ── summary ───────────────────────────────────────────────────────────────────

def med(col):
    vals = [r[col] for r in rows if r[col] is not None]
    return float(np.median(vals)) if vals else None

print(f"\n  n_valid : {len(rows)}")
print(f"  H2      : {med('H2'):.4f}")
print(f"  H3_ar   : {med('H3_ar'):.4f}  (AR cov {100*sum(r['n_ar']>0 for r in rows)/len(rows):.1f}%)")
print(f"  H3_ar+gr: {med('H3_ar_gr'):.4f}")
print(f"  H3_pt10 : {med('H3_ar_gr_pt10'):.4f}")
print(f"  H3_ptinf: {med('H3_ar_gr_ptinf'):.4f}")

# ── write output ──────────────────────────────────────────────────────────────

out_path = OUT / 'pp_m3_canonical.csv'
fields = ['withdrawal_index','time_day','recipient','n_le','n_gt',
          'n_ar','n_gr','n_pt','H2','H3_ar','H3_ar_gr','H3_ar_gr_pt10','H3_ar_gr_ptinf']
with out_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\n  Written {len(rows):,} rows → {out_path}")
