"""Shared figure style: one look across all notebooks.

Rules: every figure has title, units, data-source + as-of date in a caption;
export to figures/ at 150 dpi; colorblind-safe palette.
"""
import matplotlib.pyplot as plt

PALETTE = ["#2b4c7e", "#d9a441", "#7a9e7e", "#b55a5a", "#6b6b6b", "#8e6fae", "#4aa3a3"]


def use_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (12, 6), "figure.dpi": 110, "savefig.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 11,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
