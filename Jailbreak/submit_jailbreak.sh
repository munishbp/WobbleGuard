#!/bin/bash
# ===========================================================================
# submit_jailbreak.sh — Submit Evo2 jailbreak pipeline to SLURM
#
# Automatically selects GPU count and time limit based on model size.
#
# Usage:
#   ./submit_jailbreak.sh                                    # 1B, all viruses
#   ./submit_jailbreak.sh evo2_7b                            # 7B, all viruses
#   ./submit_jailbreak.sh evo2_40b                           # 40B, all viruses
#   ./submit_jailbreak.sh evo2_7b hiv,sars_cov2,dengue      # 7B, specific viruses
#   ./submit_jailbreak.sh evo2_1b_base "" 5                  # 1B, all, 5 samples
#   N_SAMPLES=3 SEED=123 ./submit_jailbreak.sh evo2_7b      # env overrides
# ===========================================================================

set -euo pipefail

MODEL="${1:-evo2_1b_base}"
VIRUSES="${2:-}"
N_SAMPLES="${N_SAMPLES:-${3:-1}}"
SEED="${SEED:-42}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-evo2test}"

# Resource allocation by model size
case "${MODEL}" in
    evo2_1b_base)
        NUM_GPUS=1
        MEM_PER_CPU="6G"
        TIME_LIMIT="2-00:00:00"
        ;;
    evo2_7b)
        NUM_GPUS=1
        MEM_PER_CPU="8G"
        TIME_LIMIT="3-00:00:00"
        ;;
    evo2_40b)
        NUM_GPUS=4
        MEM_PER_CPU="8G"
        TIME_LIMIT="5-00:00:00"
        ;;
    *)
        echo "ERROR: Unknown model '${MODEL}'. Choose: evo2_1b_base, evo2_7b, evo2_40b"
        exit 1
        ;;
esac

# Create log directory
mkdir -p slurm/logs

echo "Submitting jailbreak pipeline:"
echo "  Model:     ${MODEL} (${NUM_GPUS} GPU(s))"
echo "  Viruses:   ${VIRUSES:-ALL}"
echo "  Samples:   ${N_SAMPLES}"
echo "  Seed:      ${SEED}"
echo "  Time:      ${TIME_LIMIT}"
echo ""

sbatch \
    --job-name="jb_${MODEL}" \
    --output="slurm/logs/%j_jailbreak_${MODEL}.txt" \
    --partition=normal \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task="${NUM_GPUS}" \
    --cpus-per-task=12 \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --constraint="h100" \
    --time="${TIME_LIMIT}" \
    --export=ALL,MODEL="${MODEL}",VIRUSES="${VIRUSES}",N_SAMPLES="${N_SAMPLES}",SEED="${SEED}",PROJECT_DIR="${PROJECT_DIR}",CONDA_ENV="${CONDA_ENV}" \
    run_jailbreak.slurm
