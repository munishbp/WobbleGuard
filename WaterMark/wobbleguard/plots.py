"""
Publication-quality figure generation for the DNA watermarking paper.

All figures saved to D:/Research Paper/figures/.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm

from .theory import (
    detection_power,
    min_sequence_length,
    p_green_under_watermark,
    entropy_loss_per_position,
    expected_z_score,
    power_after_mutations,
)
from .simulate import SimulationResult

FIGURES_DIR = Path("D:/Research Paper/figures")

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_power_curves(
    output_path: Optional[Path] = None,
    deltas: Optional[list[float]] = None,
    gammas: Optional[list[float]] = None,
    max_L: int = 800,
    alpha: float = 0.01,
) -> Figure:
    """
    Detection power vs. sequence length for various (delta, gamma).
    Vertical dashed line at L=636 (640bp minus k=4 context).
    """
    if deltas is None:
        deltas = [0.5, 1.0, 1.5, 2.0]
    if gammas is None:
        gammas = [0.5]
    if output_path is None:
        output_path = FIGURES_DIR / "power_curves.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    Ls = np.arange(10, max_L + 1, 2)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(deltas)))
    linestyles = {0.25: "--", 0.5: "-"}

    for gamma in gammas:
        for i, delta in enumerate(deltas):
            powers = [detection_power(int(L), delta, gamma, alpha) for L in Ls]
            ls = linestyles.get(gamma, "-")
            label = f"$\\delta$={delta}, $\\gamma$={gamma}"
            ax.plot(Ls, powers, color=colors[i], linestyle=ls, linewidth=2, label=label)

    ax.axvline(x=636, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="L=636 (Evo2)")
    ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Scored sequence length (L)")
    ax.set_ylabel("Detection power (TPR)")
    ax.set_title("Watermark Detection Power vs. Sequence Length")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, max_L)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_min_length_heatmap(
    output_path: Optional[Path] = None,
    alpha: float = 0.01,
    power: float = 0.95,
) -> Figure:
    """Heatmap of L_min over (delta, gamma) grid."""
    if output_path is None:
        output_path = FIGURES_DIR / "lmin_heatmap.png"

    deltas = np.arange(0.25, 2.51, 0.25)
    gammas = np.arange(0.1, 0.91, 0.05)

    grid = np.zeros((len(gammas), len(deltas)))
    for i, gamma in enumerate(gammas):
        for j, delta in enumerate(deltas):
            lmin = min_sequence_length(float(delta), float(gamma), alpha, power)
            grid[i, j] = min(lmin, 1000)  # cap for visualization

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid, origin="lower", aspect="auto",
        extent=[deltas[0] - 0.125, deltas[-1] + 0.125, gammas[0] - 0.025, gammas[-1] + 0.025],
        cmap="viridis_r",
    )
    cbar = fig.colorbar(im, ax=ax, label="$L_{min}$ for 95% power")
    ax.set_xlabel("$\\delta$ (watermark strength)")
    ax.set_ylabel("$\\gamma$ (green-list fraction)")
    ax.set_title(f"Minimum Sequence Length for Detection (FPR={alpha})")

    # Mark Evo2 generation length
    ax.contour(
        deltas, gammas, grid, levels=[636], colors=["red"], linewidths=[2],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_z_score_distributions(
    sim_result: SimulationResult,
    output_path: Optional[Path] = None,
) -> Figure:
    """Histogram of z-scores for watermarked vs. unwatermarked sequences."""
    if output_path is None:
        output_path = FIGURES_DIR / "z_score_histograms.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(-4, max(25, sim_result.z_scores_watermarked.max() + 2), 80)

    ax.hist(
        sim_result.z_scores_unwatermarked, bins=bins, alpha=0.6,
        color="#4C72B0", label="Unwatermarked (H$_0$)", density=True,
    )
    ax.hist(
        sim_result.z_scores_watermarked, bins=bins, alpha=0.6,
        color="#DD8452", label="Watermarked (H$_1$)", density=True,
    )

    threshold = float(norm.ppf(0.99))
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"z threshold = {threshold:.2f}")

    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    ax.set_title(f"Detection z-scores ({sim_result.config_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_roc_curves(
    sim_results: dict[str, SimulationResult],
    output_path: Optional[Path] = None,
) -> Figure:
    """ROC curves for multiple parameter settings overlaid."""
    if output_path is None:
        output_path = FIGURES_DIR / "roc_curves.png"

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = plt.cm.tab20(np.linspace(0, 1, len(sim_results)))

    for idx, (label, sim) in enumerate(sim_results.items()):
        all_z = np.concatenate([sim.z_scores_watermarked, sim.z_scores_unwatermarked])
        labels_arr = np.concatenate([
            np.ones(len(sim.z_scores_watermarked)),
            np.zeros(len(sim.z_scores_unwatermarked)),
        ])

        thresholds = np.sort(np.unique(all_z))
        tprs = []
        fprs = []
        for t in thresholds[::max(1, len(thresholds) // 200)]:
            tprs.append(np.mean(sim.z_scores_watermarked > t))
            fprs.append(np.mean(sim.z_scores_unwatermarked > t))

        ax.plot(fprs, tprs, color=colors[idx], linewidth=1.5,
                label=f"{label} (AUC={sim.auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: DNA Watermark Detection")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_robustness_curves(
    detection_rates: dict[float, float],
    config_label: str = "",
    output_path: Optional[Path] = None,
) -> Figure:
    """Detection rate vs. mutation rate."""
    if output_path is None:
        output_path = FIGURES_DIR / "robustness_curves.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    rates = sorted(detection_rates.keys())
    detections = [detection_rates[r] for r in rates]

    ax.plot(
        [r * 100 for r in rates], detections,
        "o-", color="#4C72B0", linewidth=2, markersize=6,
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=0.95, color="green", linestyle="--", linewidth=1, alpha=0.5, label="95% target")

    ax.set_xlabel("Point mutation rate (%)")
    ax.set_ylabel("Detection rate (TPR)")
    ax.set_title(f"Watermark Robustness to Random Mutations ({config_label})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.5, max(rates) * 100 + 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


def plot_quality_tradeoff(
    output_path: Optional[Path] = None,
) -> Figure:
    """KL divergence (quality cost) and expected z-score vs. delta."""
    if output_path is None:
        output_path = FIGURES_DIR / "quality_tradeoff.png"

    deltas = np.arange(0.1, 3.01, 0.1)
    gamma = 0.5
    L = 636

    kls = [entropy_loss_per_position(d, gamma) for d in deltas]
    zs = [expected_z_score(L, d, gamma) for d in deltas]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    line1, = ax1.plot(deltas, kls, "b-", linewidth=2, label="KL divergence (bits/pos)")
    line2, = ax2.plot(deltas, zs, "r--", linewidth=2, label="Expected z-score at L=636")

    ax1.set_xlabel("$\\delta$ (watermark strength)")
    ax1.set_ylabel("KL(watermarked || uniform) [bits/position]", color="b")
    ax2.set_ylabel("Expected z-score", color="r")
    ax1.set_title("Quality-Detectability Tradeoff ($\\gamma$=0.5, L=636)")

    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig
