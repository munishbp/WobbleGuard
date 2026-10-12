# WobbleGuard

Code for **"Brute-Force Jailbreaks and Codon-Aware Watermarking for DNA
Foundation Models"** (NeurIPS 2026). The paper PDF is at the repo root
(`WobbleGuard_NeurIPS2026.pdf`).

Two contributions:

1. **Attack (Pillars 1-2):** Best-of-N sampling jailbreaks Evo2 into
   generating pathogen-adjacent DNA, matching the engineered GeneBreaker
   pipeline (Section 4).
2. **Defense (Pillar 4):** WobbleGuard — a zero-training, hash-based
   green-list watermark for DNA, detected at the codon synonym layer so it
   survives synonymous codon substitution that erases nucleotide-level
   statistics (Sections 5-6).

## Layout

    Jailbreak/       Best-of-N attack pipeline (Pillar 1) and the
                     same-prompt watermarked/unwatermarked control (Tier 4).
    WaterMark/       The wobbleguard package (embedder, vanilla detector,
                     wobble-aware detector, genetic code, theory, plots),
                     its test suite, and the theory/Monte Carlo simulation.
    Experiments/    Scripts reproducing the paper's analyses:
      phase3_memorization/   Codon-mismatch, single-strain recall, and
                            verbatim-detection analyses (Pillar 3).
      watermark_tiers/       Tier 1-5 watermark evaluation, calibration,
                            robustness sweeps, and supplementary
                            diagnostics A-E (Pillar 4).
      stats_and_tables/      Wilson CIs, permutation tests, category
                            summaries, FP filtering, and the Best-of-N vs.
                            GeneBreaker comparison (Pillars 1-2).
    scripts/         Run infrastructure: Modal H100 panel driver, local
                     5090 trial controller and monitor.

## Quick start

```bash
# Run the wobbleguard test suite (29 tests)
cd WaterMark && python -m pytest test_watermark.py test_wobble_detector.py

# Theory + Monte Carlo simulation (no GPU needed)
cd WaterMark && python run_theory_simulation.py

# Full Tier 1-5 watermark evaluation (local parts; Tier 4 needs Evo2
# results from the generation script)
cd Experiments/watermark_tiers && python run_tier_evaluation.py
```

GPU generation (Evo2 1B/7B/40B, SLURM on an HPC cluster; see the paper's
reproducibility appendix for pinned versions):

```bash
cd Jailbreak
sbatch run_jailbreak_7b.slurm            # Best-of-N attack trial
sbatch run_same_prompt_tier4_1b.slurm     # Tier 4 same-prompt control
```

## Notes

- Scripts bootstrap `wobbleguard` onto `sys.path` themselves, so each
  directory runs standalone; the package lives in `WaterMark/wobbleguard/`.
- The secret-key literals in the scripts are the paper's published demo
  keys; deployment keys should be freshly generated and calibrated
  (paper Appendix F/H).
- Detection numbers computed from this repo must use the same key that
  generated the sequences being scored.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
