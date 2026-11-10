"""One visual style for every figure in the report and the slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from elliptic import config  # noqa: E402

ILLICIT = "#B03A2E"
LICIT = "#1F4E8C"
UNKNOWN = "#BDBDBD"
INK = "#222222"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"

# Colour-blind safe palette (Okabe-Ito) for comparing several models.
MODEL_COLORS = {
    "LogReg": "#999999",
    "RandomForest": "#0072B2",
    "XGBoost": "#D55E00",
    "MLP": "#009E73",
    "GCN": "#CC79A7",
    "SkipGCN": "#E69F00",
    "RF+NE": "#56B4E9",
    "IsolationForest": "#000000",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def mark_timeline(ax, train_end: float = 34.5, shutdown: bool = True, label: bool = True) -> None:
    """Shade the test period and mark the dark market shutdown on a time-step axis."""
    lo, hi = ax.get_ylim()
    ax.axvspan(train_end, config.N_STEPS + 0.5, color="#F2F2F2", zorder=0)
    if label:
        ax.text(train_end + 0.4, hi, "test period", va="top", ha="left", color=MUTED, fontsize=8)
    if shutdown:
        ax.axvline(config.DARK_MARKET_STEP - 0.5, color=ILLICIT, ls="--", lw=1)
        if label:
            ax.text(
                config.DARK_MARKET_STEP - 0.3,
                hi * 0.9 + lo * 0.1,
                "dark market\nshutdown",
                color=ILLICIT,
                fontsize=8,
                va="top",
            )
    ax.set_ylim(lo, hi)


def save(fig, name: str, folder: Path = config.FIGURES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)
    return path
