#!/bin/bash
#SBATCH --job-name=ldmgen_one
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-119%6    
#SBATCH -o soutgen/%x-%j.out
#SBATCH -e soutgen/%x-%j.err

set -euo pipefail

CONFIG=/u/$USER/dmr-latent-diffusion/configs/latent-diffusion/dmr-ACCESS.yaml
CKPT=/work/hdd/bdxf/$USER/ldmlog/2025-10-13T22-16-51_dmr-ACCESS/checkpoints/last.ckpt
# OUTROOT=/u/$USER/dmr-latent-diffusion/sample_cond_FINAL
OUTROOT="/work/hdd/bdxf/$USER/ldmclass0"

mkdir -p soutgen "$OUTROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldmgen
cd /u/$USER/dmr-latent-diffusion

nvidia-smi || true

IDX=${SLURM_ARRAY_TASK_ID}
BASE_SEED=23
SEED=$(( BASE_SEED + IDX ))

python /u/$USER/dmr-latent-diffusion/scripts/class2imgDMR.py \
  --config /work/hdd/bdxf/nstjohn/ldmlog/multi/2025-12-09T23-31-35_dmr-ACCESS/configs/2025-12-09T23-31-35-project.yaml \
  --ckpt   /work/hdd/bdxf/nstjohn/ldmlog/multi/2025-12-09T23-31-35_dmr-ACCESS/checkpoints/epoch=000045.ckpt \
  --class_id 0 \
  --n 1 \
  --H 384 --W 576 \
  --steps 250 \
  --seed "$SEED" \
  --autocast \
  --scale 1.5 \
  --outdir "$OUTROOT"
