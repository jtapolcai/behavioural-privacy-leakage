
# ── path config (override via env vars BPLEAK_RG_DATA / BPLEAK_PP_DATA / BPLEAK_FIGURES) ──
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from _paths import RG_DATA as _RG_DATA, PP_DATA as _PP_DATA, FIGURES as _FIGURES
#!/usr/bin/env python3
"""
Joint 3-component exponential mixture fitted to Railgun and Privacy Pools
address-reuse delay data simultaneously.

Shared parameters : mu_1, mu_2, mu_3  (common user-type time scales)
System-specific   : pi^R, pi^PP       (mixing weights per system)

This assumes the same three user archetypes exist in both pools:
  "sprinters"  -- exit within hours
  "regulars"   -- typical weekly users
  "hodlers"    -- long-term holders (months)

Run from BehaviouralPrivacyLeakageFC/:
  python3 scripts/fit_joint_mixture.py
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import brentq

# ── 1. Load raw delay data ────────────────────────────────────────────────────
rg = pd.read_csv(_FIGURES / "cdf_deltas_data_full.csv")
x_R = rg["days"].values
x_R = x_R[x_R > 1e-6]

pp_csv = _FIGURES / "pp_h1_pairs.csv"
if pp_csv.exists():
    pp = pd.read_csv(pp_csv)
    x_P = pp["dt_days"].values
else:
    import warnings
    import warnings as _w
    _w.warn("pp_h1_pairs.csv not found — running RG-only fit")
    x_P = None

if x_P is None:
    # ── RG-only 3-component EM (no PP data) ──────────────────────────────────
    from scipy.optimize import minimize

    def rg_only_em(x, pi0, mu0, n_iter=3000, tol=1e-12):
        pi, mu = np.array(pi0, float), np.array(mu0, float)
        K, n, ll_prev = len(mu), len(x), -np.inf
        for _ in range(n_iter):
            lr = np.column_stack([np.log(pi[k]) - np.log(mu[k]) - x / mu[k] for k in range(K)])
            r = np.exp(lr - logsumexp(lr, axis=1, keepdims=True))
            N = r.sum(0)
            pi = N / n
            mu = (r * x[:, None]).sum(0) / N
            ll = logsumexp(lr, axis=1).sum()
            if ll - ll_prev < tol:
                break
            ll_prev = ll
        return pi, mu

    pi_R, mu = rg_only_em(x_R, [0.10, 0.50, 0.40], [0.10, 8.0, 45.0])
    order = np.argsort(mu); pi_R, mu = pi_R[order], mu[order]
    labels = ["sprinters", "regulars", "hodlers"]
    print("RG-only 3-component fit:")
    print(f"{'Component':<12} {'μ (days)':>10}  {'π_Railgun':>10}")
    for k in range(3):
        print(f"  {labels[k]:<10}  {mu[k]:10.3f} d  ({mu[k]*24:6.1f} h)  {pi_R[k]:10.4f}")

    t_lo_R = np.log10(max(x_R.min() * 0.5, 1e-3))
    t_hi_R = np.log10(x_R.max() * 1.05)

    def make_tex_rg(pi, mu, t_lo, t_hi, style, legend, outfile):
        terms = " + ".join(
            f"{pi[k]:.10f}*(1-exp(-10^\\u/{mu[k]:.10f}))"
            for k in range(3)
        )
        lines = [
            f"% RG-only 3-component fit — {legend}",
            f"% pi = [{pi[0]:.6f}, {pi[1]:.6f}, {pi[2]:.6f}]",
            f"% mu = [{mu[0]:.6f}, {mu[1]:.6f}, {mu[2]:.6f}] days",
            f"\\addplot [{style}, no marks, domain={t_lo:.4f}:{t_hi:.4f},",
            r"  samples=300, variable=\u]",
            f"  ({{10^\\u}},{{{terms}}});",
            f"\\addlegendentry{{{legend}}}",
        ]
        with open(outfile, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Wrote {outfile}")

    make_tex_rg(pi_R, mu, t_lo_R, t_hi_R,
                style="thick, colorRailgun, densely dotted",
                legend="Railgun 3-component fit",
                outfile=str(_FIGURES / "timing_joint_railgun_fit.tex"))
    print("\nDone (RG-only; re-run without --skip-pp for joint PP fit).")
    import sys; sys.exit(0)

x_P = x_P[x_P > 1e-6]
n_R, n_P = len(x_R), len(x_P)
print(f"Railgun : {n_R} observations  (min={x_R.min():.3f} d, median={np.median(x_R):.2f} d)")
print(f"PP      : {n_P} observations  (min={x_P.min():.3f} d, median={np.median(x_P):.2f} d)")

# ── 2. Joint EM ───────────────────────────────────────────────────────────────
def joint_em(x_R, x_P, pi_R, pi_P, mu, n_iter=3000, tol=1e-12):
    pi_R = np.array(pi_R, dtype=float)
    pi_P = np.array(pi_P, dtype=float)
    mu   = np.array(mu,   dtype=float)
    K    = len(mu)
    ll_prev = -np.inf

    for it in range(n_iter):
        # E-step — log responsibilities
        def log_r(x, pi):
            return np.column_stack([
                np.log(pi[k]) - np.log(mu[k]) - x / mu[k]
                for k in range(K)
            ])

        lr_R = log_r(x_R, pi_R)
        lr_P = log_r(x_P, pi_P)

        norm_R = logsumexp(lr_R, axis=1, keepdims=True)
        norm_P = logsumexp(lr_P, axis=1, keepdims=True)

        r_R = np.exp(lr_R - norm_R)   # (n_R, K)
        r_P = np.exp(lr_P - norm_P)   # (n_P, K)

        # M-step
        N_R = r_R.sum(axis=0)   # (K,)
        N_P = r_P.sum(axis=0)

        pi_R = N_R / n_R
        pi_P = N_P / n_P

        # shared mu: weighted mean across both datasets
        num = (r_R * x_R[:, None]).sum(axis=0) + (r_P * x_P[:, None]).sum(axis=0)
        den = N_R + N_P
        mu  = num / den

        ll = norm_R.sum() + norm_P.sum()
        if ll - ll_prev < tol:
            print(f"  Converged at iter {it+1},  joint log-lik = {ll:.4f}")
            break
        ll_prev = ll

    return pi_R, pi_P, mu

# Initialise: three modes visible in both CDFs
pi_R0 = [0.10, 0.50, 0.40]
pi_P0 = [0.05, 0.45, 0.50]   # PP visibly has more hodlers
mu0   = [0.10, 8.0, 45.0]

print("\nRunning joint EM …")
pi_R, pi_P, mu = joint_em(x_R, x_P, pi_R0, pi_P0, mu0)

# Sort by mean (ascending)
order = np.argsort(mu)
pi_R, pi_P, mu = pi_R[order], pi_P[order], mu[order]

labels = ["sprinters", "regulars", "hodlers"]
print("\nJoint 3-component fit  (shared μ, separate π):")
print(f"{'Component':<12} {'μ (days)':>10}  {'μ':>8}  {'π_Railgun':>10}  {'π_PP':>8}")
for k in range(3):
    print(f"  {labels[k]:<10}  {mu[k]:10.3f} d"
          f"  ({mu[k]*24:6.1f} h)  {pi_R[k]:10.4f}  {pi_P[k]:8.4f}")

# ── 3. Sanity: fitted medians ─────────────────────────────────────────────────
def cdf(t, pi, mu):
    return sum(pi[k] * (1 - np.exp(-t / mu[k])) for k in range(3))

for sys, x, pi in [("Railgun", x_R, pi_R), ("PP", x_P, pi_P)]:
    med_d = np.median(x)
    med_f = brentq(lambda t: cdf(t, pi, mu) - 0.5, 1e-4, max(x))
    print(f"\n{sys}: median_data={med_d:.2f} d  median_fit={med_f:.2f} d")

# ── 4. Write PGFPlots snippets ────────────────────────────────────────────────
t_lo_R = np.log10(max(x_R.min() * 0.5, 1e-3))
t_hi_R = np.log10(x_R.max() * 1.05)
t_lo_P = np.log10(max(x_P.min() * 0.5, 1e-3))
t_hi_P = np.log10(x_P.max() * 1.05)

def make_tex(pi, mu, t_lo, t_hi, style, legend, outfile):
    terms = " + ".join(
        f"{pi[k]:.10f}*(1-exp(-10^\\u/{mu[k]:.10f}))"
        for k in range(3)
    )
    lines = [
        f"% Joint 3-component fit — {legend}",
        f"% pi = [{pi[0]:.6f}, {pi[1]:.6f}, {pi[2]:.6f}]",
        f"% mu = [{mu[0]:.6f}, {mu[1]:.6f}, {mu[2]:.6f}] days  (shared)",
        f"\\addplot [{style}, no marks, domain={t_lo:.4f}:{t_hi:.4f},",
        r"  samples=300, variable=\u]",
        f"  ({{10^\\u}},{{{terms}}});",
        f"\\addlegendentry{{{legend}}}",
    ]
    with open(outfile, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {outfile}")

make_tex(pi_R, mu,
         t_lo_R, t_hi_R,
         style="thick, colorRailgun, densely dotted",
         legend="Railgun joint 3-component fit",
         outfile=str(_FIGURES / "timing_joint_railgun_fit.tex"))

make_tex(pi_P, mu,
         t_lo_P, t_hi_P,
         style="thick, colorPrivacyPools, densely dotted",
         legend="PP joint 3-component fit",
         outfile=str(_FIGURES / "timing_joint_pp_fit.tex"))

print("\nDone.")
