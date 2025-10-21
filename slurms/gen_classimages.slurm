#!/bin/bash
#SBATCH --job-name=ldm-gen_cond_final
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-59%3                     # 6 classes * 10 repeats = 60 tasks
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

set -euo pipefail

# ---- user knobs ----
NUM_CLASSES=6                               # classes 0..5
REPEATS=2                                  # repeats per class
N_PER=1
H=384
W=576
STEPS=100
ETA=0.0
SCALE=2.0

CONFIG=/u/$USER/dmr-latent-diffusion/configs/latent-diffusion/dmr-ACCESS.yaml
# 256 squares
# CKPT=/u/$USER/dmr-latent-diffusion/logs/2025-09-25T23-39-04_dmr-ACCESS/checkpoints/last.ckpt
CKPT=/work/hdd/bdxf/$USER/ldmlog/2025-10-13T22-16-51_dmr-ACCESS/checkpoints/last.ckpt
OUTROOT=/u/$USER/dmr-latent-diffusion/sample_cond_384x576


mkdir -p soutgen "$OUTROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldm310
cd /u/$USER/dmr-latent-diffusion

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Map array index -> (repeat, class)
IDX=${SLURM_ARRAY_TASK_ID}
REPEAT_IDX=$(( IDX / NUM_CLASSES ))
CLASS_ID=$(( IDX % NUM_CLASSES ))

# Stable seed per task
SEED=$(( 23 + CLASS_ID*100 + REPEAT_IDX*10000 + IDX ))

# Per-class / per-repeat outdir + informative prefix
OUTDIR="${OUTROOT}/class_${CLASS_ID}"
PREFIX="c${CLASS_ID}_r${REPEAT_IDX}_s${SEED}"
mkdir -p "$OUTDIR"

echo "class_id=$CLASS_ID repeat=$REPEAT_IDX seed=$SEED outdir=$OUTDIR"

python /u/$USER/dmr-latent-diffusion/scripts/class2imgDMR.py \
  --config "$CONFIG" \
  --ckpt   "$CKPT" \
  --class_id "$CLASS_ID" \
  --n "$N_PER" \
  --H "$H" --W "$W" \
  --steps "$STEPS"  \
  --seed "$SEED" \
  --autocast \
  --scale "$SCALE" \
  --outdir "$OUTDIR"
