#!/bin/bash
#SBATCH --job-name=flip_phatgen
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-140%7
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

set -euo pipefail

# ---- user knobs ----
NUM_CLASSES=6
REPEATS=20
N_PER=1
H=384
W=576
STEPS=250
ETA=0.0
SCALE=2.0

CONFIG=/work/hdd/bdxf/$USER/ldmlog/flips_instu/2026-04-04T18-47-56_insitu-ACCESS/configs/2026-04-04T18-47-56-project.yaml
CKPT=/work/hdd/bdxf/$USER/ldmlog/flips_instu/2026-04-04T18-47-56_insitu-ACCESS/checkpoints/epoch=002150.ckpt
OUTROOT="/work/hdd/bdxf/$USER/flip_s2.0_instu_flip"

mkdir -p "soutgen" "$OUTROOT"

# ---- map array idx -> (repeat, class) ----
IDX=${SLURM_ARRAY_TASK_ID}
REPEAT_IDX=$(( IDX / NUM_CLASSES ))
CLASS_ID=$(( IDX % NUM_CLASSES ))

# Stable, distinct seed per (class, repeat)
BASE_SEED=23
SEED=$(( BASE_SEED + IDX ))


# ---- env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldmgen
cd "/u/$USER/dmr-latent-diffusion"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

OUTDIR="${OUTROOT}/class_${CLASS_ID}"
PREFIX="c${CLASS_ID}_r${REPEAT_IDX}_s${SEED}_scale${SCALE}"
mkdir -p "$OUTDIR"

echo "idx=$IDX class_id=$CLASS_ID repeat=$REPEAT_IDX seed=$SEED scale=$SCALE outdir=$OUTDIR"

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
