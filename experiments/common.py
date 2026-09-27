"""Shared paths and plot style for the simulation study."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.simulation import RAW  # noqa: E402  results/simulation/raw

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "simulation" / "tables"      # summary CSV / markdown tables
FIG = ROOT / "figures"                                  # paper figures
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def trim_png(path, border=12):
    """Crop white margins and add a 12-pixel white border (the form in which the figures enter the paper)."""
    from PIL import Image, ImageChops, ImageOps
    im = Image.open(path).convert("RGB")
    bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    ImageOps.expand(im.crop(bb), border=border, fill="white").save(path)
