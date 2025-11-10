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
#SBATCH -o soutgen/%x-%j.out
#SBATCH -e soutgen/%x-%j.err

set -euo pipefail

CONFIG=/u/$USER/dmr-latent-diffusion/configs/latent-diffusion/dmr-ACCESS.yaml
CKPT=/work/hdd/bdxf/$USER/ldmlog/2025-10-13T22-16-51_dmr-ACCESS/checkpoints/last.ckpt
# OUTROOT=/u/$USER/dmr-latent-diffusion/sample_cond_FINAL
OUTROOT = /work/hdd/bdxf/$USER/ldmTEST

mkdir -p soutgen "$OUTROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldm-pl
cd /u/$USER/dmr-latent-diffusion

nvidia-smi || true

python /u/$USER/dmr-latent-diffusion/scripts/class2imgDMR.py \
  --config /work/hdd/bdxf/nstjohn/ldmlog/2025-10-23T17-18-37_dmr-ACCESS/configs/2025-10-23T17-18-37-project.yaml \
  --ckpt   /work/hdd/bdxf/nstjohn/ldmlog/2025-10-13T22-16-51_dmr-ACCESS/checkpoints/last.ckpt \
  --class_id 0 \
  --n 1 \
  --H 768 --W 1024 \
  --steps 250 \
  --seed 23 \
  --autocast \
  --scale 3.0 \
  --outdir /work/hdd/bdxf/$USER/ldmTEST
