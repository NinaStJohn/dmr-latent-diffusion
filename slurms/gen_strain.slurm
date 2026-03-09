#!/bin/bash
#SBATCH --job-name=strain_insiu
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-159%6
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

set -euo pipefail

# ---- sweep knobs ----
STRAIN_START=0.00
STRAIN_STEP=0.02
N_STRAIN=8        # 0 02 04 06 08 10 12 14 
REPEATS=20        # 20 repeats per strain
N_PER=1

H=384
W=576
STEPS=250
ETA=0.0
SCALE=2.0
STRAIN_KEY="strain_frac"

# ---- paths ----
CONFIG="/work/hdd/bdxf/$USER/ldmlog/strain_inst/2026-03-05T21-11-41_insitu-ACCESS/configs/2026-03-05T21-11-41-project.yaml"
CKPT="/work/hdd/bdxf/$USER/ldmlog/strain_inst/2026-03-05T21-11-41_insitu-ACCESS/checkpoints/last.ckpt"
OUTROOT="/work/hdd/bdxf/$USER/cnt_strain_sweep_step0.02"

mkdir -p "soutgen" "$OUTROOT"

# ---- env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldmgen
cd "/u/$USER/dmr-latent-diffusion"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ---- map array idx -> (strain_index, repeat) ----
IDX=${SLURM_ARRAY_TASK_ID}
TOTAL=$(( N_STRAIN * REPEATS ))
if (( IDX >= TOTAL )); then
  echo "IDX $IDX out of range (TOTAL=$TOTAL)"; exit 1
fi

STRAIN_IDX=$(( IDX / REPEATS ))
REPEAT_IDX=$(( IDX % REPEATS ))

export STRAIN_START STRAIN_STEP STRAIN_IDX
STRAIN=$(python - <<'PY'
import os
start = float(os.environ["STRAIN_START"])
step  = float(os.environ["STRAIN_STEP"])
idx   = int(os.environ["STRAIN_IDX"])
print(f"{start + idx * step:.4f}")
PY
)

BASE_SEED=23
SEED=$(( BASE_SEED + REPEAT_IDX ))   # paired noise across strains

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

# python scripts/strain2img.py \
#   --config /work/hdd/bdxf/nstjohn/ldmlog/strain_inst/2026-03-05T21-11-41_insitu-ACCESS/configs/2026-03-05T21-11-41-project.yaml \
#   --ckpt /work/hdd/bdxf/nstjohn/ldmlog/strain_inst/2026-03-05T21-11-41_insitu-ACCESS/checkpoints/last.ckpt   \
#   --strain 0.0 \
#   --strain_key strain_frac \
#   --n 2 \
#   --H 384 --W 250 \
#   --steps 250 \
#   --eta 0.0 \
#   --autocast \
#   --scale 2.0 \
#   --outdir /work/hdd/bdxf/nstjohn/cnt_strain_test0.0
