#!/bin/bash
#SBATCH --job-name=resize_recursive_2952
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o soutgen/%x-%j.out
#SBATCH -e soutgen/%x-%j.err

set -euo pipefail

# ---- user knobs ----
SRCROOT="/work/hdd/bdxf/nstjohn/flip_s2.0_instu_flip"
DSTROOT="/work/hdd/bdxf/nstjohn/resizedSmallSampleInstu"
RESIZE="2952"

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
# FIX 1: Strip trailing slash from SRCROOT so prefix removal is always consistent
SRCROOT="${SRCROOT%/}"

find "$SRCROOT" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 |
while IFS= read -r -d '' src; do
  # FIX 2: Use parameter expansion against the normalized (no trailing slash) SRCROOT
  rel="${src#${SRCROOT}/}"

  # Guard: if rel == src, the strip failed — skip rather than write to wrong path
  if [[ "$rel" == "$src" ]]; then
    echo "WARNING: could not relativize '$src', skipping" >&2
    continue
  fi

  dst="$DSTROOT/$rel"
  mkdir -p "$(dirname "$dst")"

  echo "Resizing: $rel"
  # FIX 3: -colorspace Gray converts to black and white
  magick "$src" -resize "$RESIZE" -colorspace Gray "$dst"
done

echo "Done."