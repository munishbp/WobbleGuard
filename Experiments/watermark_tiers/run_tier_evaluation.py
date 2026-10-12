#!/usr/bin/env python3
# phase 4d: full 5-tier evaluation of dna watermark detection.
#
# tiers 1-3 run locally. tier 4 needs watermarked evo2 generation results.
# tier 5 runs on whatever watermarked sequences are lying around.
#
# usage: python run_watermark_evaluation.py [--skip-ncbi] [--skip-tier4]
#
# the bootstrap snippet below was never live code (it just sat inside the old
# docstring), so it stays commented out:
# import sys as _sys
# from pathlib import Path as _Path
# _WG_ROOT = _Path(__file__).resolve().parent
# for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
#     if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
#         _sys.path.insert(0, str(_cand))

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.simulate import run_monte_carlo, generate_synthetic_sequence, EVO2_BASE_LOGITS
from wobbleguard.evaluate import (
    evaluate_ncbi_negatives,
    evaluate_evo2_unwatermarked,
    evaluate_robustness_mutations,
    evaluate_robustness_truncation,
    apply_synonymous_substitutions,
)
from wobbleguard.plots import (
    plot_z_score_distributions,
    plot_robustness_curves,
)

OUTPUT_DIR = Path("D:/Research Paper")
FIGURES_DIR = OUTPUT_DIR / "figures"
SECRET_KEY = b"neurips2026-anonymous-evo2-watermark"


def main():
    parser = argparse.ArgumentParser(description="Phase 4d: Watermark evaluation")
    parser.add_argument("--skip-ncbi", action="store_true",
                        help="Skip Tier 3 (NCBI negatives)")
    parser.add_argument("--skip-tier4", action="store_true",
                        help="Skip Tier 4 (watermarked Evo2, requires generation results)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    config = WatermarkConfig(key=SECRET_KEY, context_width=4, gamma=0.5, delta=1.5)
    all_results = {}

    print("=" * 60)
    print("PHASE 4d: DNA Watermark — Full 5-Tier Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # tier 2: negative control on existing evo2 output
    # ------------------------------------------------------------------
    print("\n--- Tier 2: Negative control on unwatermarked Evo2 output ---")

    for model, json_file in [("7B", "7B_all_results.json"), ("1B", "1B_all_results.json")]:
        json_path = OUTPUT_DIR / json_file
        if json_path.exists():
            result = evaluate_evo2_unwatermarked(json_path, config)
            all_results[f"tier2_{model}"] = result
            print(f"  {model}: {result['total_sequences']} sequences, "
                  f"{result['false_positives']} FP (FPR={result['fpr']:.4f}), "
                  f"mean z={result['mean_z']:.3f}, max z={result['max_z']:.3f}")
        else:
            print(f"  {model}: {json_file} not found, skipping")

    # ------------------------------------------------------------------
    # tier 3: negative control on ncbi natural pathogens
    # ------------------------------------------------------------------
    if not args.skip_ncbi:
        ncbi_path = OUTPUT_DIR / "ncbi_negative_sequences.json"
        if ncbi_path.exists():
            print("\n--- Tier 3: Negative control on NCBI natural pathogens ---")
            ncbi_result = evaluate_ncbi_negatives(ncbi_path, config)
            # the raw z-scores are huge, drop them from the summary
            all_results["tier3_ncbi"] = {
                k: v for k, v in ncbi_result.items() if k != "all_z_scores"
            }
            print(f"  Total fragments: {ncbi_result['total_fragments']}")
            print(f"  False positives: {ncbi_result['total_false_positives']}")
            print(f"  Overall FPR: {ncbi_result['overall_fpr']:.4f} "
                  f"(95% CI: [{ncbi_result['fpr_95ci_low']:.4f}, {ncbi_result['fpr_95ci_high']:.4f}])")
            print(f"  Mean z: {ncbi_result['mean_z']:.3f}, Max z: {ncbi_result['max_z']:.3f}")

            print("\n  Per-virus FPR:")
            for virus, stats in sorted(ncbi_result["per_virus"].items()):
                # flag anything with a suspiciously high per-virus rate
                flag = " ***" if stats["fpr"] > 0.03 else ""
                print(f"    {virus}: {stats['false_positives']}/{stats['n_fragments']} "
                      f"(FPR={stats['fpr']:.4f}, max_z={stats['max_z']:.3f}){flag}")
        else:
            print(f"\n--- Tier 3: SKIPPED (run fetch_ncbi_negatives.py first) ---")
    else:
        print("\n--- Tier 3: SKIPPED (--skip-ncbi) ---")

    # ------------------------------------------------------------------
    # tier 5: stress tests (on synthetic watermarked sequences)
    # ------------------------------------------------------------------
    print("\n--- Tier 5: Stress tests ---")

    # 50 synthetic watermarked sequences for the stress tests
    print("  Generating 50 watermarked sequences for stress tests...")
    rng = np.random.default_rng(999)
    watermarked_seqs = []
    for i in range(50):
        seq = generate_synthetic_sequence(
            config, length=640, base_logits=EVO2_BASE_LOGITS,
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        watermarked_seqs.append(seq)

    # 5a: point mutation robustness
    print("  5a: Point mutation robustness...")
    mutation_results = evaluate_robustness_mutations(
        watermarked_seqs, config,
        mutation_rates=[0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
        n_trials=20, seed=42,
    )
    detection_rates = {}
    for r in mutation_results:
        detection_rates[r.perturbation_param] = r.detection_rate
        print(f"    Mutation rate {r.perturbation_param:.0%}: "
              f"detection={r.detection_rate:.3f}, mean_z={r.mean_z_score:.2f}")

    plot_robustness_curves(
        detection_rates, "delta=1.5, gamma=0.5, k=4",
        FIGURES_DIR / "robustness_mutations.png",
    )
    print("    [+] figures/robustness_mutations.png")

    all_results["tier5_mutations"] = [
        {"rate": r.perturbation_param, "detection": r.detection_rate, "mean_z": r.mean_z_score}
        for r in mutation_results
    ]

    # 5b: truncation robustness
    print("  5b: Truncation robustness...")
    trunc_results = evaluate_robustness_truncation(watermarked_seqs, config)
    trunc_rates = {}
    for r in trunc_results:
        trunc_rates[r.perturbation_param] = r.detection_rate
        kept_bp = int(640 * r.perturbation_param)
        print(f"    Keep {r.perturbation_param:.0%} ({kept_bp}bp): "
              f"detection={r.detection_rate:.3f}, mean_z={r.mean_z_score:.2f}")

    all_results["tier5_truncation"] = [
        {"keep_frac": r.perturbation_param, "detection": r.detection_rate, "mean_z": r.mean_z_score}
        for r in trunc_results
    ]

    # 5e: synonymous codon substitution
    print("  5e: Synonymous codon substitution...")
    detector = WatermarkDetector(config, alpha=0.01)
    syn_rates_to_test = [0.0, 0.25, 0.50, 0.75, 1.0]
    syn_rng = np.random.default_rng(77)

    for syn_rate in syn_rates_to_test:
        detected = 0
        for seq in watermarked_seqs:
            mutated = apply_synonymous_substitutions(seq, syn_rate, syn_rng)
            if detector.detect(mutated).is_watermarked:
                detected += 1
        det_rate = detected / len(watermarked_seqs)
        print(f"    Synonymous sub rate {syn_rate:.0%}: detection={det_rate:.3f}")

    # ------------------------------------------------------------------
    # save all the results
    # ------------------------------------------------------------------
    with open(OUTPUT_DIR / "watermark_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results saved to watermark_evaluation_results.json")


if __name__ == "__main__":
    main()
