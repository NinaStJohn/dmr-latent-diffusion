#!/bin/bash
#SBATCH --job-name=porosity
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-104%6
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

set -euo pipefail

# ---- sweep knobs ----
PORO_START=0.999
PORO_STEP=0.005
N_PORO=11          # 1.00 → 0.94 inclusive
REPEATS=20
N_PER=1

H=384
W=576
STEPS=250
ETA=0.0
SCALE=2.0
PORO_KEY="porosity_frac"

# ---- paths ----
CONFIG="/work/hdd/bdxf/nstjohn/ldmlog/porosity/2025-12-17T12-12-41_dmr-ACCESS/configs/2025-12-17T12-12-41-project.yaml"
CKPT="/work/hdd/bdxf/nstjohn/ldmlog/porosity/2025-12-17T12-12-41_dmr-ACCESS/checkpoints/epoch=000848.ckpt"
OUTROOT="/work/hdd/bdxf/$USER/porosity_sweep_step0.01"

mkdir -p "soutgen" "$OUTROOT"

# ---- env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldmgen
cd "/u/$USER/dmr-latent-diffusion"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ---- map array idx -> (repeat, porosity_index) ----
IDX=${SLURM_ARRAY_TASK_ID}
REPEAT_IDX=$(( IDX / N_PORO ))
PORO_IDX=$(( IDX % N_PORO ))

export PORO_START PORO_STEP PORO_IDX
POROSITY=$(python - <<'PY'
import os
v = float(os.environ["PORO_START"]) - int(os.environ["PORO_IDX"]) * float(os.environ["PORO_STEP"])
print(f"{v:.4f}")
PY
)

BASE_SEED=23
SEED=$(( BASE_SEED + IDX ))

OUTDIR="${OUTROOT}/poro_${POROSITY}"
PREFIX="p${POROSITY}_r${REPEAT_IDX}_s${SEED}"
mkdir -p "$OUTDIR"

echo "idx=$IDX porosity=$POROSITY repeat=$REPEAT_IDX seed=$SEED"

python /u/nstjohn/dmr-latent-diffusion/scripts/porosity2img.py \
  --config "$CONFIG" \
  --ckpt   "$CKPT" \
  --porosity "$POROSITY" \
  --porosity_key "$PORO_KEY" \
  --n "$N_PER" \
  --H "$H" --W "$W" \
  --steps "$STEPS" \
  --eta "$ETA" \
  --seed "$SEED" \
  --autocast \
  --scale "$SCALE" \
  --outdir "$OUTDIR" \
  --prefix "$PREFIX"
