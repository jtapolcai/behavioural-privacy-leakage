"""_paths.py — canonical data/output paths for all measurement scripts.

Every script imports this module and uses its constants.  Override via env vars:

  BPLEAK_RG_DATA    path to railgun_deanonymization/data/
  BPLEAK_PP_DATA    path to privacypools-deanonymization/data/processed/
  BPLEAK_FIGURES    output directory (default: <repo_root>/Figures/)
"""
import os
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent   # …/behavioural-privacy-leakage/scripts/
_ROOT    = _SCRIPTS.parent                   # …/behavioural-privacy-leakage/

# ── Railgun raw data ─────────────────────────────────────────────────────────
RG_DATA = Path(os.environ.get(
    "BPLEAK_RG_DATA",
    str(_ROOT / "data" / "railgun_deanonymization" / "data"),
))

# ── Privacy Pools processed exports ──────────────────────────────────────────
PP_DATA = Path(os.environ.get(
    "BPLEAK_PP_DATA",
    str(_ROOT / "data" / "privacypools-deanonymization" / "data" / "processed"),
))

# ── Figure output directory ───────────────────────────────────────────────────
FIGURES = Path(os.environ.get(
    "BPLEAK_FIGURES",
    str(_ROOT / "Figures"),
))
