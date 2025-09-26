#!/bin/bash
#SBATCH --job-name=ldm-gen_cond
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-17%3                         # 6 classes * 3 repeats = 18 tasks; run up to 3 at once
#SBATCH -o soutgen/%x-%j_%a.out
#SBATCH -e soutgen/%x-%j_%a.err

# ---- user knobs ----
NUM_CLASSES=6                                  # classes 0..5
REPEATS=3                                      # repeats per class
N_PER=1                                        # samples per task (keep small for VRAM)
H=512
W=512
CONFIG=/u/$USER/dmr-latent-diffusion/configs/latent-diffusion/dmr-ACCESS.yaml
CKPT=/u/$USER/dmr-latent-diffusion/logs/2025-09-24T12-18-44_dmr-ACCESS/checkpoints/last.ckpt
OUTROOT=/u/$USER/dmr-latent-diffusion/samples_cls4

mkdir -p soutgen

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldm310
cd /u/$USER/dmr-latent-diffusion

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# map 0-based array index -> (class_id 0..5, repeat 0..2)
IDX=${SLURM_ARRAY_TASK_ID}
REPEAT_IDX=$(( IDX / NUM_CLASSES ))
CLASS_ID=$(( IDX % NUM_CLASSES ))

# unique seed per task (stable but different)
# SEED=$(( 23 + CLASS_ID*100 + REPEAT_IDX*10000 + IDX ))
SEED=500

# single line, no commented lines in the middle
python /u/$USER/dmr-latent-diffusion/scripts/class2imgDMR.py \
  --config "$CONFIG" \
  --ckpt   "$CKPT" \
  --class_id "$CLASS_ID" \
  --n "$N_PER" \
  --H "$H" --W "$W" \
  --outdir "$OUTROOT" \
  # --steps 60 --eta 0.1 \
  # --scale 3.0 --sampler ddim \
   --seed "$SEED" \
  # --autocast \
