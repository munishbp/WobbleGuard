#!/usr/bin/env python3
# phase 4b: theory analysis + monte carlo validation of the dna watermark.
# runs fully local — no gpu, no evo2, no ncbi.
# outputs: figures/, simulation results json, and a theory table csv.
# usage: python run_watermark_simulation.py

# make the wobbleguard package importable when run from this directory.
# import sys as _sys
# from pathlib import Path as _Path
# _WG_ROOT = _Path(__file__).resolve().parent
# for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
#     if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
#         _sys.path.insert(0, str(_cand))

import json
import csv
import time
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.theory import generate_theory_table, expected_z_score
from wobbleguard.simulate import run_monte_carlo, sweep_parameters, EVO2_BASE_LOGITS
from wobbleguard.plots import (
    plot_power_curves,
    plot_min_length_heatmap,
    plot_z_score_distributions,
    plot_roc_curves,
    plot_quality_tradeoff,
)

OUTPUT_DIR = Path("D:/Research Paper")
FIGURES_DIR = OUTPUT_DIR / "figures"
SECRET_KEY = b"neurips2026-anonymous-evo2-watermark"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. theory table
    print("=" * 60)
    print("PHASE 4b: DNA Watermark — Theoretical Analysis + Simulation")
    print("=" * 60)

    print("\n--- Theoretical Predictions ---")
    theory_rows = generate_theory_table()
    with open(OUTPUT_DIR / "watermark_theory_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=theory_rows[0].keys())
        writer.writeheader()
        writer.writerows(theory_rows)

    for row in theory_rows:
        print(f"  delta={row['delta']}, gamma={row['gamma']}: "
              f"z={row['expected_z_at_L']}, L_min={row['L_min_95']}, "
              f"KL={row['kl_bits_per_pos']} bits/pos")

    # 2. theoretical figures, no simulation needed
    print("\n--- Generating theoretical figures ---")
    plot_power_curves()
    print("  [+] figures/power_curves.png")

    plot_min_length_heatmap()
    print("  [+] figures/lmin_heatmap.png")

    plot_quality_tradeoff()
    print("  [+] figures/quality_tradeoff.png")

    # 3. monte carlo simulation with the default config
    print("\n--- Monte Carlo: default config (delta=1.5, gamma=0.5, k=4) ---")
    default_config = WatermarkConfig(
        key=SECRET_KEY, context_width=4, gamma=0.5, delta=1.5,
    )

    # uniform base distribution
    print("  Uniform base distribution (10,000 per class)...")
    sim_uniform = run_monte_carlo(
        default_config, n_watermarked=10_000, n_unwatermarked=10_000,
        base_logits=None, seed=42,
    )
    print(f"    TPR={sim_uniform.true_positive_rate:.4f}, "
          f"FPR={sim_uniform.false_positive_rate:.4f}, "
          f"AUC={sim_uniform.auc:.6f}")
    print(f"    Mean z (watermarked)={sim_uniform.mean_z_watermarked:.2f}, "
          f"Mean z (null)={sim_uniform.mean_z_unwatermarked:.2f}")
    print(f"    Theory predicted z={expected_z_score(636, 1.5, 0.5):.2f}")

    # at-biased base distribution, evo2-like
    print("  AT-biased base distribution (10,000 per class)...")
    sim_biased = run_monte_carlo(
        default_config, n_watermarked=10_000, n_unwatermarked=10_000,
        base_logits=EVO2_BASE_LOGITS, seed=43,
    )
    print(f"    TPR={sim_biased.true_positive_rate:.4f}, "
          f"FPR={sim_biased.false_positive_rate:.4f}, "
          f"AUC={sim_biased.auc:.6f}")

    # z-score histogram for the default config, uniform
    plot_z_score_distributions(sim_uniform)
    print("  [+] figures/z_score_histograms.png")

    # 4. parameter sweep
    print("\n--- Parameter sweep (2,000 per class per setting) ---")
    sweep_results = sweep_parameters(
        key=SECRET_KEY,
        deltas=[0.5, 1.0, 1.5, 2.0],
        gammas=[0.25, 0.5],
        context_widths=[1, 4],
        n_samples=2_000,
        seq_length=640,
        seed=44,
    )

    # roc curves
    plot_roc_curves(sweep_results)
    print("  [+] figures/roc_curves.png")

    # 5. save sweep results
    sweep_summary = []
    for label, result in sweep_results.items():
        sweep_summary.append({
            "config": label,
            "tpr": round(result.true_positive_rate, 4),
            "fpr": round(result.false_positive_rate, 4),
            "auc": round(result.auc, 6),
            "mean_z_wm": round(result.mean_z_watermarked, 2),
            "mean_z_null": round(result.mean_z_unwatermarked, 2),
        })
        print(f"  {label}: TPR={result.true_positive_rate:.4f}, "
              f"FPR={result.false_positive_rate:.4f}, AUC={result.auc:.6f}")

    with open(OUTPUT_DIR / "watermark_simulation_results.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
