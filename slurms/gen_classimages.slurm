#!/bin/bash
#SBATCH --job-name=flip_2extra_2.0
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-119%6                 # 6 classes * 20 repeats = 120 tasks; keep this in sync with NUM_CLASSES*REPEATS
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

set -euo pipefail

# ---- user knobs ----
NUM_CLASSES=6                            # classes 0..5
REPEATS=20                               # repeats per class (array “rows”)y
N_PER=1                                  # images per invocation (per task)
H=384
W=576
STEPS=250
ETA=0.0                                  
SCALE=2.0

CONFIG=/work/hdd/bdxf/nstjohn/ldmlog/flips/2025-12-16T16-48-21_dmr-ACCESS/configs/2025-12-16T16-48-21-project.yaml
CKPT="/work/hdd/bdxf/nstjohn/ldmlog/flips/2025-12-16T16-48-21_dmr-ACCESS/checkpoints/epoch=000701.ckpt"
OUTROOT="/work/hdd/bdxf/$USER/c&flip2.5-cp701"

mkdir -p "soutgen" "$OUTROOT"

# ---- env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldmgen
cd "/u/$USER/dmr-latent-diffusion"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- map array idx -> (repeat, class) ----
IDX=${SLURM_ARRAY_TASK_ID}
REPEAT_IDX=$(( IDX / NUM_CLASSES ))      # 0..(REPEATS-1)
CLASS_ID=$(( IDX % NUM_CLASSES ))        # 0..(NUM_CLASSES-1)

# Stable, distinct seed per (class, repeat)
BASE_SEED=23
SEED=$(( BASE_SEED + IDX ))

# Per-class / per-repeat outdir + informative prefix
OUTDIR="${OUTROOT}/class_${CLASS_ID}"
PREFIX="c${CLASS_ID}_r${REPEAT_IDX}_s${SEED}"
mkdir -p "$OUTDIR"

echo "idx=$IDX class_id=$CLASS_ID repeat=$REPEAT_IDX seed=$SEED outdir=$OUTDIR"

python scripts/class2imgDMR.py \
  --config "$CONFIG" \
  --ckpt   "$CKPT" \
  --class_id "$CLASS_ID" \
  --n "$N_PER" \
  --H "$H" --W "$W" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --autocast \
  --scale "$SCALE" \
  --outdir "$OUTDIR" \
  --prefix "$PREFIX"
