"""
Shared plotting utilities used across all phases.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from src.utils.config import OUTPUTS_PLOTS

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "accent":    "#0891B2",
    "warm":      "#EA580C",
    "success":   "#16A34A",
    "muted":     "#6B7280",
    "light":     "#F3F4F6",
}

def set_style():
    """Apply a clean, consistent matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.color":        "#E5E7EB",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "figure.dpi":        120,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
    })


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure to outputs/plots/ as PNG."""
    out = OUTPUTS_PLOTS / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")
    return out