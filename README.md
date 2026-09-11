# behavioural-privacy-leakage — Figure Generation Pipeline

Reproducible figure and data pipeline for the paper  
**"A Tattered Cloak of Invisibility: Measuring Behavioural Privacy Leakage in Railgun and Privacy Pools on Ethereum"**

---

## Quick start

```bash
python3 generate_figures.py          # clone repos + run everything
python3 generate_figures.py -v       # same, with live script output
```

All outputs land in `Figures/`.  Copy the files you need into the LaTeX
repo's `Figures/` directory:

| File pattern | Used by LaTeX as |
|---|---|
| `*.csv` | `\addplot table{…}` inside a tikz figure |
| `*_tikz.tex` | `\input{Figures/…}` self-contained tikz figure |

---

## Data repositories

The script clones two repos into `data/` on first run and pulls updates
on subsequent runs:

| Repo | Contents |
|---|---|
| `https://github.com/c0rt3x1337x/railgun_deanonymization` | Railgun aggregated exports, H1 pairs, knapsack witness outputs |
| `git@github.com:alexistrihine-pixel/privacypools-deanonymization.git` | Privacy Pools processed deposit/withdrawal CSVs |

If you already have the repos locally, point to them directly:

```bash
python3 generate_figures.py \
  --skip-clone \
  --rg-data /path/to/railgun_deanonymization/data \
  --pp-data /path/to/privacypools-deanonymization/data/processed
```

---

## Options

```
--skip-clone     do not git clone / pull (use data/ as-is)
--skip-rg        skip Railgun-specific scripts
--skip-pp        skip Privacy Pools-specific scripts
--rg-data DIR    override railgun data directory
--pp-data DIR    override PP processed/ directory
--figures DIR    override output directory  [default: ./Figures]
-v / --verbose   stream each script's stdout to the terminal
```

---

## What is generated

### Timing model  (RG + PP)
| Output file(s) | Figure / table |
|---|---|
| `figure_ch4_h1_temporal_cdf*.csv` | Fig. — same-address delay CDF |
| `figure_ch4_h1_temporal_pdf*.csv` | Fig. — delay PDF |
| `mc_railgun_timing_pdf.csv`, `mc_pp_timing_pdf.csv` | Fig. 4 Monte Carlo null curves |
| `fig4_cdf_railgun_observed.csv`, `fig4_cdf_privacypools_observed.csv` | Fig. 4 empirical CDFs |

### Dataset overview
| Output file(s) | Figure / table |
|---|---|
| `figure_ch4_01_dataset_inventory.csv` | Tab. — dataset inventory |
| `table_dataset_overview.csv` | Inline numbers (Sec. 3) |
| `figure_ch4_02_weekly_boundary_counts*.csv` | Fig. — weekly shield/unshield counts |

### Privacy Pools
| Output file(s) | Figure / table |
|---|---|
| `pp_h1_pairs.csv`, `pp_h1_coverage.csv` | Fig. — PP H1 coverage |
| `pp_h1_summary_stats.csv` | Inline stats (Sec. 5) |
| `pp_cdf_deltas_pdf_data.csv` | Fig. 4 PP timing |
| `figure_ch4_03_pp_cumulative_pool_flow*.csv` | Fig. — PP cumulative flow |
| `figure_ch4_03_pp_retained_fifo*.csv` | Fig. — PP FIFO retention |
| `pp_broadcaster_cluster_points.csv`, `pp_broadcaster_tikz.tex` | Fig. — PP broadcaster cluster |
| `fig2_pp_amount_distribution.csv` | Fig. 2 PP amount CDF |

### M1–M4 entropy models
| Output file(s) | Figure / table |
|---|---|
| `figure_ch5_wA_entropy_heatmap_*.csv` | Fig. — entropy heatmap |
| `figure_ch5_wB_deltaH_hist_*.csv` | Fig. — ΔH histogram |
| `figure_ch5_wC_window_scan_*.csv` | Fig. — M4 sensitivity curve H₄(p_cov) |
| `sec4_entropy_model_table.tex` | Tab. — M1–M4 summary (RG + PP) |
| `figure_ch4_23_interpretation_synthesis*.csv` | Fig. — entropy reduction summary |

### Calibration (Table 2)
| Output file(s) | Figure / table |
|---|---|
| `calibration_table.tex` | Tab. 2 — obs→est timing parameters |

### H5 amount fingerprints
| Output file(s) | Figure / table |
|---|---|
| `frac3_fingerprint_histogram_data*.csv` | Fig. — fingerprint histogram (RG + PP) |
| `frac3_ks_cdf_*.csv`, `frac3_ks_stat_line.csv` | Fig. — KS test CDF |
| `h5_mc_null_distribution_pp.csv` | Fig. — MC null distribution (PP) |
| `hamming_distance_rg_data.csv`, `hamming_distance_pp_data.csv` | Fig. — Hamming distance |

---

## Repository layout

```
behavioural-privacy-leakage/
├── generate_figures.py          ← entry point
├── README.md
├── scripts/
│   ├── _paths.py                ← path config (reads BPLEAK_* env vars)
│   ├── fit_joint_mixture.py
│   ├── refit_timing_mixture.py
│   ├── generate_pp_and_montecarlo.py
│   ├── generate_fig2_privacypools.py
│   ├── generate_fig4_timing_cdf.py
│   ├── generate_fig5_privacypools.py
│   ├── generate_broadcaster_privacypools.py
│   ├── generate_dataset_table.py
│   ├── compute_h1_subset_fifo.py
│   ├── compute_m2_canonical.py
│   ├── compute_m3_{canonical,hard_filter,stochastic}.py
│   ├── compute_rg_entropy_models.py
│   ├── compute_entropy_models.py
│   ├── compute_table2.py
│   ├── estimate_pp_origin_entropy.py
│   ├── estimate_pp_k_window.py
│   └── refresh_pp_measurements.py
├── data/                        ← cloned by generate_figures.py (git-ignored)
│   ├── railgun_deanonymization/
│   └── privacypools-deanonymization/
└── Figures/                     ← all outputs land here (git-ignored)
```

---

## Path override via environment variables

All scripts read three env vars set automatically by `generate_figures.py`.
You can also set them manually to run individual scripts:

```bash
export BPLEAK_RG_DATA=/path/to/railgun_deanonymization/data
export BPLEAK_PP_DATA=/path/to/privacypools-deanonymization/data/processed
export BPLEAK_FIGURES=/path/to/output/Figures
python3 scripts/compute_entropy_models.py
```

---

## Requirements

```bash
pip install numpy pandas scipy matplotlib
```

The knapsack scripts (`estimate_pp_origin_entropy.py`) additionally
require the compiled C witness kernel from the Railgun repo
(`src/h4_knapsack/witness_kernel`).  Run `bash src/h4_knapsack/build.sh`
inside the cloned railgun repo to build it.
