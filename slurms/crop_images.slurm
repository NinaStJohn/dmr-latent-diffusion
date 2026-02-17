#!/bin/bash
#SBATCH --job-name=crop_recursive_1952x1968
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1              # <-- policy compliance (even if unused)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o soutgen/%x-%j.out
#SBATCH -e soutgen/%x-%j.err

set -euo pipefail

# ---- user knobs ----
SRCROOT="/work/hdd/bdxf/nstjohn/in-situ"
DSTROOT="/work/hdd/bdxf/nstjohn/in-situ_crop"

# crop to fixed size (top-left by default, so bottom-right label is excluded)
CROP_W="2952"
CROP_H="1968"
XOFF="0"
YOFF="0"

CROP_SCRIPT="/u/nstjohn/dmr-latent-diffusion/scripts/crop_images.py"

mkdir -p soutgen "$DSTROOT"

# ---- env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate imgtools

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "SRCROOT=$SRCROOT"
echo "DSTROOT=$DSTROOT"
echo "CROP_W=$CROP_W"
echo "CROP_H=$CROP_H"
echo "XOFF=$XOFF"
echo "YOFF=$YOFF"
echo "CROP_SCRIPT=$CROP_SCRIPT"

python -V
python -c "import PIL; print('Pillow:', PIL.__version__)"

# ---- run crop (mirrors tree + keeps filenames) ----
python "$CROP_SCRIPT" \
  -i "$SRCROOT" \
  -o "$DSTROOT" \
  --width "$CROP_W" \
  --height "$CROP_H" \
  --x "$XOFF" \
  --y "$YOFF"

echo "Done."
