#!/bin/bash
#SBATCH --job-name=strain_ldm
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-119%6
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

set -euo pipefail

# ---- sweep knobs ----
STRAIN_START=0.00
STRAIN_STEP=0.04
N_STRAIN=6        # 0.00, 0.04, 0.08, 0.12, 0.16, 0.20
REPEATS=20        # 20 repeats per strain
N_PER=1

H=384
W=576
STEPS=250
ETA=0.0
SCALE=2.0
STRAIN_KEY="strain_frac"

# ---- paths ----
CONFIG="/work/hdd/bdxf/$USER/ldmlog/strain/2026-02-27T15-55-27_dmr-ACCESS/configs/2026-02-27T15-55-27-project.yaml"
CKPT="/work/hdd/bdxf/$USER/ldmlog/strain/2026-02-27T15-55-27_dmr-ACCESS/checkpoints/last.ckpt"
OUTROOT="/work/hdd/bdxf/$USER/strain_sweep_step0.04"

mkdir -p "soutgen" "$OUTROOT"

# ---- env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldmgen
cd "/u/$USER/dmr-latent-diffusion"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ---- map array idx -> (repeat, strain_index) ----
IDX=${SLURM_ARRAY_TASK_ID}
REPEAT_IDX=$(( IDX / N_STRAIN ))
STRAIN_IDX=$(( IDX % N_STRAIN ))

export STRAIN_START STRAIN_STEP STRAIN_IDX
STRAIN=$(python - <<'PY'
import os
start = float(os.environ["STRAIN_START"])
step  = float(os.environ["STRAIN_STEP"])
idx   = int(os.environ["STRAIN_IDX"])
v = start + idx * step
print(f"{v:.4f}")
PY
)

BASE_SEED=23
SEED=$(( BASE_SEED + IDX ))

OUTDIR="${OUTROOT}/strain_${STRAIN}"
PREFIX="strain${STRAIN}_r${REPEAT_IDX}_s${SEED}"
mkdir -p "$OUTDIR"

echo "idx=$IDX strain=$STRAIN repeat=$REPEAT_IDX seed=$SEED"

python scripts/strain2img.py \
  --config "$CONFIG" \
  --ckpt   "$CKPT" \
  --strain "$STRAIN" \
  --strain_key "$STRAIN_KEY" \
  --n "$N_PER" \
  --H "$H" --W "$W" \
  --steps "$STEPS" \
  --eta "$ETA" \
  --seed "$SEED" \
  --autocast \
  --scale "$SCALE" \
  --outdir "$OUTDIR" \
  --prefix "$PREFIX"