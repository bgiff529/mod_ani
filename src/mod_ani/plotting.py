"""Plot helpers for notebook reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_history(histories: dict[str, list[dict[str, Any]]], output: Path | None = None):
    """Plot validation RMSE curves for one or more model histories."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, history in histories.items():
        epochs = [row["epoch"] for row in history]
        rmse = [row["rmse_kcal_mol"] for row in history]
        ax.plot(epochs, rmse, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (kcal/mol)")
    ax.set_title("Energy Model Validation Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig, ax
