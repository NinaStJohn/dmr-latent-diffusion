#!/bin/bash
#SBATCH --job-name=resize_recursive_3072
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
SRCROOT="/work/hdd/bdxf/nstjohn/pre-resize"
DSTROOT="/work/hdd/bdxf/nstjohn/resizedSmallSample"
RESIZE="3072"   # preserves aspect ratio, sets width=3072

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
echo "RESIZE=$RESIZE"

# ---- walk + mirror tree, keep filenames ----
find "$SRCROOT" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 |
while IFS= read -r -d '' src; do
  rel="${src#$SRCROOT/}"
  dst="$DSTROOT/$rel"
  mkdir -p "$(dirname "$dst")"

  echo "Resizing: $rel"
  magick "$src" -resize "$RESIZE" "$dst"
done

echo "Done."
