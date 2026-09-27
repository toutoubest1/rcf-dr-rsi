"""Paths of the KuaiRec application (all relative to the repository root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "kuairec"                  # put KuaiRec.zip here (see data/README.md)
DATA_RED = DATA_RAW / "reduced"                        # written by reduce_kuairec.py
RES = ROOT / "results" / "kuairec"
RAW, TAB = RES / "raw", RES / "tables"
FIG = ROOT / "figures"
MAIN_TAB = ROOT / "tables"
SUPP_TAB = ROOT / "supp_tables"
