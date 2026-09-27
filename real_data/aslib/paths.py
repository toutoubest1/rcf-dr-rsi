"""Paths of the ASlib application (all relative to the repository root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "aslib" / "aslib_data"          # git clone of coseal/aslib_data (see data/README.md)
RES = ROOT / "results" / "aslib"
RAW, TAB = RES / "raw", RES / "tables"
FIG = ROOT / "figures"
