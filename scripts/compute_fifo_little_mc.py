#!/usr/bin/env python3
"""
compute_fifo_little_mc.py
=========================
Monte Carlo validation of the declared scenario parameters against two
aggregate pool-level diagnostics:

  FIFO lag   — horizontal shift between cumulative deposit and withdrawal
               curves; answered: "how old is the deposit being matched now?"
  Little's   — retained_balance / outflow_rate = L/lambda
               answered: "how long does an average ETH-unit stay in the pool?"

For each system (PP, RG) we:
  1. Use the ACTUAL deposit arrival times and amounts from the data.
  2. For N_SIM Monte Carlo runs, sample holding times from the declared
     scenario mixture (π^est, μ^est) for every deposit.
  3. Build the simulated withdrawal stream, then compute FIFO lag and
     Little's law time series from that stream.
  4. Compare the simulated distribution (median ± CI) against the
     observed values from the pipeline CSVs.

The key question: are the observed FIFO / Little's values plausible *given*
the declared scenario, or do they falsify it?

Usage
-----
  python scripts/compute_fifo_little_mc.py \
      --pp-source  data/privacypools-deanonymization \
      --pp-obs-csv Figures/data/figure_ch4_03_pp_retained_fifo.csv \
      --rg-obs-csv Figures/data/figure_ch4_03_little_law_timeseries.csv \
      [--n-sim 1000] [--seed 42] [--output analysis/mc_fifo_little]
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# ── scenario parameters (single source of truth) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from _scenario_params import PP_PI, PP_MU, RG_PI, RG_MU

SECS_PER_DAY = 86_400.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_amount(s: str) -> float:
    """Handle both '0.5' and '0,5' decimal formats."""
    s = s.strip().strip('"').replace(',', '.')
    try:
        return float(Decimal(s))
    except InvalidOperation:
        return float('nan')


def parse_time(s: str) -> float:
    """Return Unix timestamp (seconds) from ISO-like strings."""
    s = s.strip().rstrip(' UTC').strip()
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S,000', '%Y-%m-%d'):
        try:
            dt = datetime.strptime(s.replace(',000', '.000'), fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse time: {s!r}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# Core computations
# ─────────────────────────────────────────────────────────────────────────────

def sample_holding_days(n: int, pi: np.ndarray, mu: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw n holding times (days) from the declared mixture."""
    components = rng.choice(len(pi), size=n, p=pi / pi.sum())
    return rng.exponential(mu[components])


def build_timeseries(dep_times_days: np.ndarray,
                     dep_amounts: np.ndarray,
                     holding_days: np.ndarray,
                     eval_grid: np.ndarray,
                     window_days: float = 90.0
                     ) -> dict[str, np.ndarray]:
    """
    Given deposit arrivals and sampled holding times, compute:
      cum_dep(t), cum_with(t), retained(t), fifo_lag(t), little_law(t)
    evaluated at eval_grid (days since pool start).
    """
    with_times = dep_times_days + holding_days   # simulated withdrawal days

    # Only include withdrawals that happen before or at max eval grid time
    t_max = eval_grid[-1]
    valid = with_times <= t_max

    dep_s = dep_times_days
    dep_a = dep_amounts
    with_s = with_times[valid]
    with_a = dep_amounts[valid]

    # Sort
    dep_order = np.argsort(dep_s)
    dep_s, dep_a = dep_s[dep_order], dep_a[dep_order]
    with_order = np.argsort(with_s)
    with_s, with_a = with_s[with_order], with_a[with_order]

    cum_dep_at  = np.array([dep_a[dep_s <= t].sum()  for t in eval_grid])
    cum_with_at = np.array([with_a[with_s <= t].sum() for t in eval_grid])
    retained    = cum_dep_at - cum_with_at

    # FIFO lag: for each t, find t' such that cum_dep(t') = cum_with(t)
    fifo_lag = np.full(len(eval_grid), np.nan)
    for i, t in enumerate(eval_grid):
        target = cum_with_at[i]
        if target <= 0 or cum_dep_at[i] <= 0:
            continue
        # Find earliest t' where cum_dep ≥ target
        idx = np.searchsorted(cum_dep_at, target, side='left')
        if idx < len(eval_grid):
            fifo_lag[i] = t - eval_grid[idx]

    # Little's law: retained / (outflow rate over window_days)
    little_law = np.full(len(eval_grid), np.nan)
    for i, t in enumerate(eval_grid):
        if retained[i] <= 0:
            continue
        t_lo = t - window_days
        if t_lo < 0:
            continue
        out_hi = cum_with_at[i]
        out_lo = with_a[with_s <= t_lo].sum() if t_lo > 0 else 0.0
        rate = (out_hi - out_lo) / window_days  # units/day
        if rate > 0:
            little_law[i] = retained[i] / rate

    return {
        'cum_dep':   cum_dep_at,
        'cum_with':  cum_with_at,
        'retained':  retained,
        'fifo_lag':  fifo_lag,
        'little':    little_law,
    }


def summarise(ts: dict[str, np.ndarray]) -> dict[str, float]:
    """Median over the time series, ignoring NaN.

    Returns keys: fifo_median, fifo_last, little_median, little_last
    """
    result = {}
    for ts_key, out_key in [('fifo_lag', 'fifo'), ('little', 'little')]:
        v = ts[ts_key]
        v = v[np.isfinite(v)]
        result[out_key + '_median'] = float(np.median(v)) if len(v) else float('nan')
        result[out_key + '_last']   = float(v[-1])         if len(v) else float('nan')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Load observed diagnostics from pipeline CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_observed_pp(csv_path: Path) -> dict[str, float]:
    rows = read_csv(csv_path)
    fifo_vals = [float(r['fifo_horizontal_lag_days'])
                 for r in rows
                 if r.get('fifo_horizontal_lag_days') not in ('', 'nan', 'inf', None)]
    return {
        'fifo_median': float(np.median(fifo_vals)) if fifo_vals else float('nan'),
        'fifo_last':   fifo_vals[-1]               if fifo_vals else float('nan'),
        'little_median': float('nan'),   # not in PP CSV; computed here from flow
        'little_last':   float('nan'),
    }


def load_observed_rg(csv_path: Path) -> dict[str, float]:
    rows = read_csv(csv_path)
    fifo_vals  = [float(r['fifo_horizontal_lag_days'])
                  for r in rows
                  if r.get('fifo_horizontal_lag_days') not in ('', 'nan', 'inf', None)]
    little_vals = [float(r['little_days_global'])
                   for r in rows
                   if r.get('little_days_global') not in ('', 'nan', 'inf', None)]
    return {
        'fifo_median':   float(np.median(fifo_vals))   if fifo_vals   else float('nan'),
        'fifo_last':     fifo_vals[-1]                  if fifo_vals   else float('nan'),
        'little_median': float(np.median(little_vals)) if little_vals else float('nan'),
        'little_last':   little_vals[-1]                if little_vals else float('nan'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load deposit data
# ─────────────────────────────────────────────────────────────────────────────

def load_pp_deposits(source: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (dep_times_days_from_start, dep_amounts_eth)."""
    raw = read_csv(source / 'data/processed/processed_privacypools_eth_pool_deposits.csv')
    events = []
    for r in raw:
        try:
            t = parse_time(r['time'])
            a = parse_amount(r['amount_eth'])
            if a > 0:
                events.append((t, a))
        except Exception:
            pass
    events.sort()
    t0 = events[0][0]
    times = np.array([(t - t0) / SECS_PER_DAY for t, _ in events])
    amts  = np.array([a for _, a in events])
    return times, amts


def load_rg_deposits_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load RG cumulative flow CSV and recover approximate daily deposit increments.
    The CSV has cum_shield and date; we diff to get daily amounts.
    """
    rows = read_csv(csv_path)
    rows = [r for r in rows
            if r.get('cum_shield') not in ('', 'nan', None)]
    rows.sort(key=lambda r: r['date'])
    cum = [float(r['cum_shield']) for r in rows]
    dates = [r['date'] for r in rows]

    # daily increments (approximate deposit amounts per day)
    amounts, times = [], []
    for i in range(1, len(cum)):
        delta = cum[i] - cum[i-1]
        if delta > 0:
            # treat as a single deposit at mid-day
            t0_str, t_str = dates[0], dates[i]
            t0 = datetime.strptime(t0_str, '%Y-%m-%d')
            t  = datetime.strptime(t_str,  '%Y-%m-%d')
            day_offset = (t - t0).days
            times.append(float(day_offset))
            amounts.append(delta)

    return np.array(times), np.array(amounts)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mc(dep_times: np.ndarray, dep_amts: np.ndarray,
           pi: np.ndarray, mu: np.ndarray,
           n_sim: int, rng: np.random.Generator,
           window_days: float = 90.0) -> dict[str, np.ndarray]:
    """
    Run n_sim simulations. Return arrays of shape (n_sim,) for each metric.
    """
    n_dep = len(dep_times)
    t_span = dep_times[-1] - dep_times[0]
    eval_grid = np.linspace(0, t_span, min(500, int(t_span) + 1))

    results = {k: np.full(n_sim, np.nan)
               for k in ('fifo_median', 'fifo_last', 'little_median', 'little_last')}

    for i in range(n_sim):
        holding = sample_holding_days(n_dep, pi, mu, rng)
        ts = build_timeseries(dep_times, dep_amts, holding, eval_grid, window_days)
        s  = summarise(ts)
        for k in results:
            results[k][i] = s[k]

        if (i + 1) % max(1, n_sim // 10) == 0:
            print(f"  sim {i+1}/{n_sim} — "
                  f"FIFO med={np.nanmedian(results['fifo_median'][:i+1]):.1f}d  "
                  f"Little med={np.nanmedian(results['little_median'][:i+1]):.1f}d",
                  flush=True)

    return results


def report(label: str, results: dict[str, np.ndarray],
           observed: dict[str, float], est_ET: float) -> dict:
    """Print and return a summary report."""
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  Declared scenario E[T]^est = {est_ET:.1f} d")
    print(f"  N_sim = {len(next(iter(results.values())))}")
    out = {'label': label, 'E_T_est': est_ET, 'observed': observed}

    for metric, obs_key_med, obs_key_last in [
        ('fifo_median',   'fifo_median',   'fifo_last'),
        ('little_median', 'little_median', 'little_last'),
    ]:
        vals = results[metric]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        name = 'FIFO lag' if 'fifo' in metric else "Little's law"
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        obs_med = observed.get(obs_key_med, float('nan'))
        obs_las = observed.get(obs_key_last, float('nan'))

        # one-sided p-value: P(sim ≤ observed_median)
        p_val = float(np.mean(vals <= obs_med)) if np.isfinite(obs_med) else float('nan')

        print(f"\n  {name}:")
        print(f"    Simulated — p5={p5:.1f}d  p25={p25:.1f}d  "
              f"med={p50:.1f}d  p75={p75:.1f}d  p95={p95:.1f}d")
        print(f"    Observed  — median={obs_med:.1f}d  last={obs_las:.1f}d")
        print(f"    P(sim ≤ obs_median) = {p_val:.3f}  "
              f"({'consistent ✓' if 0.05 <= p_val <= 0.95 else 'INCONSISTENT ✗'})")

        out[name] = {
            'sim_p5': p5, 'sim_p25': p25, 'sim_p50': p50,
            'sim_p75': p75, 'sim_p95': p95,
            'obs_median': obs_med, 'obs_last': obs_las,
            'p_value': p_val,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pp-source',  type=Path,
                    default=Path(__file__).parents[1] /
                            'data/privacypools-deanonymization',
                    help='PP repo root (contains data/processed/)')
    # The observed diagnostic CSVs live in the paper repo (sister directory)
    _paper = Path(__file__).parents[1].parent / '6aa410a17e14d2dde604af68' / 'Figures' / 'data'
    ap.add_argument('--pp-obs-csv', type=Path,
                    default=_paper / 'figure_ch4_03_pp_retained_fifo.csv')
    ap.add_argument('--rg-obs-csv', type=Path,
                    default=_paper / 'figure_ch4_03_little_law_timeseries_plot.csv')
    ap.add_argument('--rg-flow-csv', type=Path,
                    default=_paper / 'figure_ch4_03_little_law_timeseries_plot.csv')
    ap.add_argument('--n-sim',  type=int, default=500)
    ap.add_argument('--seed',   type=int, default=42)
    ap.add_argument('--output', type=Path,
                    default=Path(__file__).parents[1] / 'analysis/mc_fifo_little')
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    E_T_pp = float(PP_PI @ PP_MU)
    E_T_rg = float(RG_PI @ RG_MU)

    all_results = []

    # ── PP ────────────────────────────────────────────────────────────────────
    print("\n=== Privacy Pools ===")
    print(f"  E[T]^est = {E_T_pp:.1f} d  (π·μ = {PP_PI}·{PP_MU})")
    try:
        dep_times_pp, dep_amts_pp = load_pp_deposits(args.pp_source)
        print(f"  Loaded {len(dep_times_pp):,} deposits, "
              f"span {dep_times_pp[-1] - dep_times_pp[0]:.0f} d, "
              f"total {dep_amts_pp.sum():.1f} ETH")

        obs_pp = load_observed_pp(args.pp_obs_csv)

        print(f"\n  Running {args.n_sim} MC simulations …")
        mc_pp = run_mc(dep_times_pp, dep_amts_pp, PP_PI, PP_MU,
                       args.n_sim, rng, window_days=90.0)

        rep_pp = report('Privacy Pools', mc_pp, obs_pp, E_T_pp)
        all_results.append(rep_pp)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # ── RG ────────────────────────────────────────────────────────────────────
    print("\n\n=== Railgun ===")
    print(f"  E[T]^est = {E_T_rg:.1f} d  (π·μ = {RG_PI}·{RG_MU})")
    try:
        # Use the cumulative flow CSV to reconstruct approximate daily deposits
        dep_times_rg, dep_amts_rg = load_rg_deposits_from_csv(args.rg_flow_csv)
        print(f"  Reconstructed {len(dep_times_rg):,} daily deposit events, "
              f"span {dep_times_rg[-1]:.0f} d, "
              f"total {dep_amts_rg.sum():.1f} WETH")

        obs_rg = load_observed_rg(args.rg_obs_csv)

        print(f"\n  Running {args.n_sim} MC simulations …")
        mc_rg = run_mc(dep_times_rg, dep_amts_rg, RG_PI, RG_MU,
                       args.n_sim, rng, window_days=180.0)

        rep_rg = report('Railgun', mc_rg, obs_rg, E_T_rg)
        all_results.append(rep_rg)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # ── Cross-validation: PP with RG's π (same μ, but RG mixing weights) ─────
    # Question: if PP users had the same type distribution as RG users, would
    # the PP FIFO / Little's values look different?  If yes → the different π
    # is justified; if no → one shared user model is sufficient.
    print("\n\n=== Cross-validation: PP deposits + RG's π_est (shared μ) ===")
    E_T_pp_rg_pi = float(RG_PI @ PP_MU)   # RG weights, but PP's μ_h=246d
    print(f"  E[T]^cross = {E_T_pp_rg_pi:.1f} d  "
          f"(RG_PI={RG_PI}, PP_MU={PP_MU})")
    try:
        # Reuse dep_times_pp / dep_amts_pp from the PP block above
        if 'dep_times_pp' not in dir():
            dep_times_pp, dep_amts_pp = load_pp_deposits(args.pp_source)
        obs_pp_xv = load_observed_pp(args.pp_obs_csv)

        print(f"\n  Running {args.n_sim} MC simulations (RG π, PP μ) …")
        mc_pp_xv = run_mc(dep_times_pp, dep_amts_pp,
                          RG_PI, PP_MU,          # ← RG mixing weights
                          args.n_sim, rng, window_days=90.0)
        rep_pp_xv = report('PP + RG-π (cross-validation)', mc_pp_xv, obs_pp_xv, E_T_pp_rg_pi)
        all_results.append(rep_pp_xv)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = args.output / 'mc_summary.json'
    with out_json.open('w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults written → {out_json}")

    # ── Print final comparison table ──────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  SUMMARY: Observed vs. Simulated (scenario) — median ± 90% CI")
    print(f"{'━'*70}")
    for rep in all_results:
        print(f"\n  {rep['label']}   (E[T]^est = {rep['E_T_est']:.0f} d)")
        for mname in ("FIFO lag", "Little's law"):
            if mname not in rep:
                continue
            d = rep[mname]
            ci = f"[{d['sim_p5']:.0f}, {d['sim_p95']:.0f}]"
            obs = d['obs_median']
            flag = '✓' if 0.05 <= d['p_value'] <= 0.95 else '✗'
            print(f"    {mname:<14}  obs={obs:.0f}d  "
                  f"sim_med={d['sim_p50']:.0f}d  90%CI={ci}  p={d['p_value']:.3f} {flag}")


if __name__ == '__main__':
    main()
