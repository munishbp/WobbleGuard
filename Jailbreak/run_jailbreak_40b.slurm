#!/bin/bash
#SBATCH --job-name=jb_40b_fill
#SBATCH --output=slurm/logs/%j_jailbreak_40b_fill.txt
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3-00:00:00
#SBATCH --account=abedi
#SBATCH --qos=abedi

module load gcc cuda/cuda-12.1.0
eval "$(conda shell.bash hook)"
conda activate evo2test

echo "=== GPU diagnostics ==="
nvidia-smi || echo "nvidia-smi FAILED"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo "========================"

MODEL="evo2_40b"
VIRUSES="hiv,sars_cov2,hpv,hsv1,hku1,229e,dengue,reovirus,influenza"
N_SAMPLES=1
SEED=42

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING 2>/dev/null || true
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1

CLI_ARGS="--model_name ${MODEL} --n_samples ${N_SAMPLES} --seed ${SEED} --output_dir results/evo2_40b_20260326_103204 --virus_keys ${VIRUSES}"

echo "============================================================"
echo "Jailbreak Pipeline — Job ${SLURM_JOB_ID}"
echo "============================================================"
echo "  Model:      ${MODEL}"
echo "  Viruses:    ${VIRUSES}"
echo "  N_samples:  ${N_SAMPLES}"
echo "  Node:       $(hostname)"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  Started:    $(date -Iseconds)"
echo "============================================================"

python jailbreak_pipeline.py ${CLI_ARGS}

echo "Job ${SLURM_JOB_ID} finished at $(date -Iseconds)"
