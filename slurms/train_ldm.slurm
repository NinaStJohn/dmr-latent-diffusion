#!/bin/bash
#SBATCH --job-name=ldm10_yamlupdate2
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1          # 1 GPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o sout/%x-%j.out
#SBATCH -e sout/%x-%j.err

set -euo pipefail
mkdir -p sout

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldm310
cd /u/$USER/dmr-latent-diffusion

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Helpful envs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# CONFIG=/u/nstjohn/dmr-latent-diffusion/configs/latent-diffusion/dmr-ACCESS.yaml
# LOGDIR_BASE=/u/nstjohn/dmr-latent-diffusion/logs

# # Optional auto-resume from last.ckpt in a given run directory (set RUN_ID if you want)
# RUN_ID=""   # e.g. 2025-08-27T20-09-27_autoencoder_ACCESS
# RESUME=""
# if [[ -n "${RUN_ID}" ]]; then
#   LAST="${LOGDIR_BASE}/${RUN_ID}/checkpoints/last.ckpt"
#   if [[ -f "$LAST" ]]; then
#     echo "[resume] Found $LAST"
#     RESUME="--resume $LAST"
#   fi
# fi

# YAML should contain:
# lightning.trainer:
#   gpus: "0"
#   accelerator: gpu
#   precision: 16-true
# (Don't pass --gpus here; your main.py reads from YAML.)
# srun --ntasks=1 python -u main.py \
#   --base "$CONFIG" \
#   -t $RESUME

srun python -u main2.py \
  --base /u/$USER/dmr-latent-diffusion/configs/latent-diffusion/dmr-ACCESS.yaml \
  -t \
  --logdir /work/hdd/bdxf/$USER/ldmlog
  # -r /u/$USER/dmr-latent-diffusion/logs/2025-09-15T23-14-39_dmr-ACCESS/checkpoints/last.ckpt


# output /work/hdd/bdxf/$USER