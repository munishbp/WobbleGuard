#!/bin/bash
#SBATCH --job-name=evo2_7b
#SBATCH --output=slurm/logs/%j_trial%a_jailbreak_7b.txt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=3-00:00:00
#SBATCH --account=abedi
#SBATCH --qos=abedi
#SBATCH --exclude=evc44,evc31,evc39,evc40,evc26
# (array removed; use 4 individual sbatch submissions)

module load gcc cuda/cuda-12.1.0
eval "$(conda shell.bash hook)"
conda activate evo2test

echo "=== GPU diagnostics ==="
nvidia-smi || echo "nvidia-smi FAILED"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo "========================"

MODEL="evo2_7b"
VIRUSES=""
N_SAMPLES=1
SEED=$((42 + TRIAL))
OUTPUT_DIR="results/${MODEL}_trial${TRIAL}_${SLURM_JOB_ID}"

cd /home/mu901052/neurips

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING 2>/dev/null || true
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1

CLI_ARGS="--model_name ${MODEL} --n_samples ${N_SAMPLES} --seed ${SEED} --output_dir ${OUTPUT_DIR}"

if [[ -n "${VIRUSES}" ]]; then
    CLI_ARGS="${CLI_ARGS} --virus_keys ${VIRUSES}"
fi

echo "============================================================"
echo "Jailbreak Pipeline — Job ${SLURM_JOB_ID}, Array Task ${TRIAL}"
echo "============================================================"
echo "  Model:      ${MODEL}"
echo "  Viruses:    ${VIRUSES:-ALL}"
echo "  N_samples:  ${N_SAMPLES}"
echo "  Seed:       ${SEED}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Node:       $(hostname)"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Started:    $(date -Iseconds)"
echo "============================================================"

python jailbreak_pipeline.py ${CLI_ARGS}

echo "Job ${SLURM_JOB_ID} task ${TRIAL} finished at $(date -Iseconds)"