#!/usr/bin/env python3
"""generate_figures.py
=============================================================
Downloads (or updates) the two raw-data repositories, then
runs every figure-generation script in order.  All outputs
land in ./Figures/ and are ready to copy into the LaTeX repo.

Usage
-----
    python3 generate_figures.py [options]

Options
-------
  --skip-clone   skip git clone / pull (use data/ as-is)
  --skip-rg      skip Railgun-specific scripts
  --skip-pp      skip Privacy Pools-specific scripts
  --rg-data DIR  override path to railgun data/ directory
  --pp-data DIR  override path to PP processed/ directory
  --figures DIR  override output directory (default: ./Figures)
  -v / --verbose print each script's stdout in real time
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ─── repo layout ────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data"
SCRIPTS = ROOT / "scripts"

RG_REPO = "https://github.com/c0rt3x1337x/railgun_deanonymization"
PP_REPO = "git@github.com:alexistrihine-pixel/privacypools-deanonymization.git"

RG_DIR  = DATA / "railgun_deanonymization"
PP_DIR  = DATA / "privacypools-deanonymization"

# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--skip-clone", action="store_true",
                   help="do not git clone / pull")
    p.add_argument("--skip-rg",    action="store_true",
                   help="skip Railgun-specific scripts")
    p.add_argument("--skip-pp",    action="store_true",
                   help="skip Privacy Pools-specific scripts")
    p.add_argument("--rg-data",    type=Path, default=RG_DIR / "data",
                   metavar="DIR",  help="railgun data directory")
    p.add_argument("--pp-data",    type=Path,
                   default=PP_DIR / "data" / "processed",
                   metavar="DIR",  help="PP processed/ directory")
    p.add_argument("--figures",    type=Path, default=ROOT / "Figures",
                   metavar="DIR",  help="output directory")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="stream each script's output to the terminal")
    return p.parse_args()

# ─── helpers ─────────────────────────────────────────────────────────────────
SEP = "━" * 70

def banner(label: str) -> None:
    print(f"\n{SEP}\n  {label}\n{SEP}")

def clone_or_update(url: str, dest: Path) -> None:
    name = dest.name
    if (dest / ".git").is_dir():
        print(f"  [{name}] already cloned — pulling …")
        r = subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  Warning: pull failed ({r.stderr.strip()}) — using existing checkout.")
        else:
            print(f"  [{name}] up to date.")
    else:
        print(f"  [{name}] cloning from {url} …")
        subprocess.run(["git", "clone", url, str(dest)], check=True)

def run(label: str, script: Path, args: list[str], verbose: bool,
        env: dict[str, str]) -> bool:
    """Run a script; return True on success, False on error (never raises)."""
    banner(label)
    cmd = [sys.executable, str(script)] + args
    print("  " + " ".join(cmd))
    result = subprocess.run(
        cmd, env=env,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stdout or ""
        print(f"  ⚠  exited with code {result.returncode}" +
              (f"\n{msg.rstrip()}" if msg else ""))
        return False
    return True

# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # Propagate paths to all child scripts via env vars (_paths.py reads these)
    env = os.environ.copy()
    env["BPLEAK_RG_DATA"] = str(args.rg_data)
    env["BPLEAK_PP_DATA"] = str(args.pp_data)
    env["BPLEAK_FIGURES"] = str(args.figures)

    # Convenience aliases
    rg  = str(args.rg_data)
    pp  = str(args.pp_data)
    fig = str(args.figures)
    v   = args.verbose

    # ── 1. Clone / update source repos ──────────────────────────────────────
    if not args.skip_clone:
        banner("Clone / update source repos")
        clone_or_update(RG_REPO, RG_DIR)
        clone_or_update(PP_REPO, PP_DIR)
    else:
        print("--skip-clone: skipping git operations")

    # ── 2. Timing model fit ──────────────────────────────────────────────────
    # Outputs: figure_ch4_h1_temporal_{cdf,pdf}*.csv
    if not args.skip_rg:
        run("Fit 3-component timing mixture (RG + PP joint)",
            SCRIPTS / "fit_joint_mixture.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)

        run("Refit timing mixture (sanity check)",
            SCRIPTS / "refit_timing_mixture.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)

    # ── 3. PP H1 pairs + Monte Carlo null ────────────────────────────────────
    # Outputs: pp_h1_pairs.csv, pp_cdf_deltas_pdf_data.csv,
    #          mc_pp_timing_pdf.csv, pp_h1_coverage.csv, pp_h1_summary_stats.csv
    if not args.skip_pp:
        run("PP H1 pairs + Monte Carlo null",
            SCRIPTS / "generate_pp_and_montecarlo.py",
            ["--pp-data", pp, "--output-dir", fig], v, env)

        run("Figure 4 — timing CDF series",
            SCRIPTS / "generate_fig4_timing_cdf.py",
            ["--figures-dir", fig], v, env)

    # ── 4. Dataset inventory table ───────────────────────────────────────────
    # Outputs: figure_ch4_01_dataset_inventory.csv, table_dataset_overview.csv
    run("Dataset inventory table",
        SCRIPTS / "generate_dataset_table.py",
        ["--pp-data", pp, "--figures-dir", fig], v, env)

    # ── 5. PP cumulative flow + FIFO retention ───────────────────────────────
    # Outputs: figure_ch4_03_pp_*.csv
    if not args.skip_pp:
        run("PP cumulative flow + FIFO retention",
            SCRIPTS / "generate_fig5_privacypools.py",
            ["--pp-data", pp, "--figures-dir", fig], v, env)

        run("Figure 2 — PP amount distribution",
            SCRIPTS / "generate_fig2_privacypools.py",
            ["--pp-data", pp,
             "--output", str(args.figures / "fig2_pp_amount_distribution.csv")],
            v, env)

        run("PP broadcaster cluster",
            SCRIPTS / "generate_broadcaster_privacypools.py",
            ["--source",
             str(args.pp_data / "processed_privacypools_data_withdraws.csv"),
             "--output-dir", fig], v, env)

    # ── 6. H1 address-reuse subset + FIFO correction (RG) ───────────────────
    if not args.skip_rg:
        run("H1 address-reuse subset + FIFO correction",
            SCRIPTS / "compute_h1_subset_fifo.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)

    # ── 7. M2 canonical (temporal weighting) ────────────────────────────────
    if not args.skip_rg:
        run("M2 canonical entropy (RG)",
            SCRIPTS / "compute_m2_canonical.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)

    # ── 8. M3 variants (public-relation leakage) ────────────────────────────
    if not args.skip_rg:
        run("M3 canonical (address/gas-payer reuse)",
            SCRIPTS / "compute_m3_canonical.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)
        run("M3 hard filter",
            SCRIPTS / "compute_m3_hard_filter.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)
        run("M3 stochastic",
            SCRIPTS / "compute_m3_stochastic.py",
            ["--rg-data", rg, "--output-dir", fig], v, env)

    # ── 9. PP origin entropy (knapsack support) ──────────────────────────────
    if not args.skip_pp:
        run("PP origin entropy (knapsack support)",
            SCRIPTS / "estimate_pp_origin_entropy.py",
            ["--pp-data", pp, "--output-dir", fig], v, env)
        run("PP k-window entropy",
            SCRIPTS / "estimate_pp_k_window.py",
            ["--pp-data", pp, "--output-dir", fig], v, env)

    # ── 10. Full entropy model table (M1–M4, RG + PP) ───────────────────────
    # Outputs: sec4_entropy_model_table.tex, figure_ch5_* CSVs
    run("RG entropy models (M1–M4, all windows)",
        SCRIPTS / "compute_rg_entropy_models.py",
        ["--rg-data", rg, "--output-dir", fig], v, env)

    run("Combined entropy model table + tikz (M1–M4 RG+PP)",
        SCRIPTS / "compute_entropy_models.py",
        ["--rg-data", rg, "--pp-data", pp, "--output-dir", fig], v, env)

    run("Table 2 — calibration parameters (obs→est)",
        SCRIPTS / "compute_table2.py",
        ["--rg-data", rg, "--output-dir", fig], v, env)

    # ── 11. H5 amount fingerprint + MC null ─────────────────────────────────
    run("H5 amount fingerprint + Monte Carlo (RG + PP)",
        SCRIPTS / "refresh_pp_measurements.py",
        ["--pp-data", pp, "--rg-data", rg, "--output-dir", fig], v, env)

    # ── done ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Done.  Outputs are in:  {args.figures}/")
    print()
    print("  Copy to LaTeX Figures/:")
    print("    *.csv        → tikz figures that use \\addplot table{…}")
    print("    *_tikz.tex   → self-contained tikz figures")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
