#!/usr/bin/env bash
# =============================================================================
# get_measurement_data.sh
# =============================================================================
# Downloads (or updates) the two source repos, then runs every figure-
# generation script.  Outputs land in ./Figures/.
#
# Usage:
#   ./get_measurement_data.sh [--skip-clone] [--skip-rg] [--skip-pp]
#
# Options:
#   --skip-clone  do not git clone / pull (use whatever is already in data/)
#   --skip-rg     skip Railgun-specific figure scripts
#   --skip-pp     skip Privacy Pools-specific figure scripts
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$REPO_ROOT/data"
SCRIPTS="$REPO_ROOT/scripts"
FIGURES="$REPO_ROOT/Figures"

RG_REPO_URL="https://github.com/c0rt3x1337x/railgun_deanonymization"
PP_REPO_URL="git@github.com:alexistrihine-pixel/privacypools-deanonymization.git"

RG_DIR="$DATA_DIR/railgun_deanonymization"
PP_DIR="$DATA_DIR/privacypools-deanonymization"

SKIP_CLONE=0
SKIP_RG=0
SKIP_PP=0

for arg in "$@"; do
  case "$arg" in
    --skip-clone) SKIP_CLONE=1 ;;
    --skip-rg)    SKIP_RG=1    ;;
    --skip-pp)    SKIP_PP=1    ;;
  esac
done

mkdir -p "$DATA_DIR" "$FIGURES"

# ─────────────────────────────────────────────────────────────────────────────
# Export paths so _paths.py (imported by all scripts) picks them up
# ─────────────────────────────────────────────────────────────────────────────
export BPLEAK_RG_DATA="$RG_DIR/data"
export BPLEAK_PP_DATA="$PP_DIR/data/processed"
export BPLEAK_FIGURES="$FIGURES"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Clone or update the two source repos
# ─────────────────────────────────────────────────────────────────────────────
clone_or_update() {
  local url="$1"
  local dest="$2"
  local name
  name="$(basename "$dest")"

  if [ -d "$dest/.git" ]; then
    echo ">>> [$name] already cloned — pulling latest …"
    git -C "$dest" pull --ff-only || {
      echo "    Warning: pull failed (network issue or diverged). Continuing with existing checkout."
    }
  else
    echo ">>> [$name] cloning from $url …"
    git clone "$url" "$dest"
  fi
}

if [ "$SKIP_CLONE" -eq 0 ]; then
  clone_or_update "$RG_REPO_URL" "$RG_DIR"
  clone_or_update "$PP_REPO_URL" "$PP_DIR"
else
  echo ">>> --skip-clone: skipping git operations"
fi

# Convenience paths used by the scripts below
RG_DATA="$RG_DIR/data"                                  # aggregated CSVs live here
PP_DATA="$PP_DIR/data/processed"                        # processed_privacypools_*.csv

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
run_py() {
  # run_py LABEL script.py [args …]
  local label="$1"; shift
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $label"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 "$@" || echo "  ⚠  $label exited with error $? — continuing"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Timing model fit  (needs RG H1 pairs; outputs CSV + tex into Figures/)
#    → figure_ch4_h1_temporal_{cdf,pdf}*.csv, mc_railgun_timing_pdf.csv
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_RG" -eq 0 ]; then
  run_py "Fit 3-component timing mixture (RG + PP joint)" \
    "$SCRIPTS/fit_joint_mixture.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"

  run_py "Refit timing mixture (sanity check)" \
    "$SCRIPTS/refit_timing_mixture.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Privacy Pools H1 pairs + Monte Carlo null  (CSV only — tikz loads these)
#    → pp_h1_pairs.csv, pp_cdf_deltas_pdf_data.csv, mc_pp_timing_pdf.csv,
#      pp_h1_coverage.csv, pp_dataset_inventory.csv, pp_h1_summary_stats.csv
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_PP" -eq 0 ]; then
  run_py "PP H1 pairs + Monte Carlo null (generate_pp_and_montecarlo)" \
    "$SCRIPTS/generate_pp_and_montecarlo.py" \
    --pp-data "$PP_DATA" \
    --output-dir "$FIGURES"

  # Figure 4: empirical CDF series (RG observed + PP observed)
  run_py "Figure 4 — timing CDF series (CSV)" \
    "$SCRIPTS/generate_fig4_timing_cdf.py" \
    --figures-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset inventory table  (CSV — tikz loads it)
# ─────────────────────────────────────────────────────────────────────────────
run_py "Dataset table (figure_ch4_01 + table_dataset_overview)" \
  "$SCRIPTS/generate_dataset_table.py" \
  --pp-data "$PP_DATA" \
  --figures-dir "$FIGURES"

# ─────────────────────────────────────────────────────────────────────────────
# 5. PP cumulative flow + FIFO retention  (CSV — tikz loads these)
#    → figure_ch4_03_pp_*.csv
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_PP" -eq 0 ]; then
  run_py "PP cumulative flow + FIFO (Figure 5 equivalent)" \
    "$SCRIPTS/generate_fig5_privacypools.py" \
    --pp-data "$PP_DATA" \
    --figures-dir "$FIGURES"

  # Figure 2: PP amount distribution
  run_py "Figure 2 — PP amount distribution (CSV)" \
    "$SCRIPTS/generate_fig2_privacypools.py" \
    --pp-data "$PP_DATA" \
    --output "$FIGURES/fig2_pp_amount_distribution.csv"

  # Broadcaster cluster
  run_py "PP broadcaster cluster (CSV + tikz)" \
    "$SCRIPTS/generate_broadcaster_privacypools.py" \
    --source "$PP_DATA/processed_privacypools_data_withdraws.csv" \
    --output-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. H1 address-reuse subset + FIFO correction  (RG)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_RG" -eq 0 ]; then
  run_py "H1 address-reuse subset + FIFO correction (RG)" \
    "$SCRIPTS/compute_h1_subset_fifo.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. M2 canonical (temporal weighting) — CSV only
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_RG" -eq 0 ]; then
  run_py "M2 canonical entropy (RG)" \
    "$SCRIPTS/compute_m2_canonical.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. M3 variants (public-relation leakage) — CSV only
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_RG" -eq 0 ]; then
  run_py "M3 canonical (address/gas-payer reuse) (RG)" \
    "$SCRIPTS/compute_m3_canonical.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"

  run_py "M3 hard filter (RG)" \
    "$SCRIPTS/compute_m3_hard_filter.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"

  run_py "M3 stochastic (RG)" \
    "$SCRIPTS/compute_m3_stochastic.py" \
    --rg-data "$RG_DATA" \
    --output-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 9. PP origin entropy (knapsack support) — CSV only
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_PP" -eq 0 ]; then
  run_py "PP origin entropy (knapsack support)" \
    "$SCRIPTS/estimate_pp_origin_entropy.py" \
    --pp-data "$PP_DATA" \
    --output-dir "$FIGURES"

  run_py "PP k-window entropy" \
    "$SCRIPTS/estimate_pp_k_window.py" \
    --pp-data "$PP_DATA" \
    --output-dir "$FIGURES"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 10. Full entropy model table (M1–M4, RG + PP) — CSV + tex
#     → sec4_entropy_model_table.tex, figure_ch5_* CSVs
# ─────────────────────────────────────────────────────────────────────────────
run_py "RG entropy models (M1-M4, all windows)" \
  "$SCRIPTS/compute_rg_entropy_models.py" \
  --rg-data "$RG_DATA" \
  --output-dir "$FIGURES"

run_py "Combined entropy model table + tikz (M1-M4 RG+PP)" \
  "$SCRIPTS/compute_entropy_models.py" \
  --rg-data "$RG_DATA" \
  --pp-data "$PP_DATA" \
  --output-dir "$FIGURES"

run_py "Table 2 calibration (obs→est parameters)" \
  "$SCRIPTS/compute_table2.py" \
  --rg-data "$RG_DATA" \
  --output-dir "$FIGURES"

# ─────────────────────────────────────────────────────────────────────────────
# 11. H5 amount fingerprint (MC null, KS test) — CSV only
# ─────────────────────────────────────────────────────────────────────────────
run_py "H5 amount fingerprint + Monte Carlo (RG+PP)" \
  "$SCRIPTS/refresh_pp_measurements.py" \
  --pp-data "$PP_DATA" \
  --rg-data "$RG_DATA" \
  --output-dir "$FIGURES"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Done.  Generated files are in:"
echo "    $FIGURES/"
echo ""
echo "  CSVs for tikz figures: copy matching *.csv to your LaTeX Figures/ dir."
echo "  Self-contained tikz  : copy the *_tikz.tex files directly."
echo "════════════════════════════════════════════════════════════════════════"
