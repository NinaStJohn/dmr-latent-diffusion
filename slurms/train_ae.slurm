#!/bin/bash
#SBATCH --job-name=ae-1g
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1        # 1 GPU 
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o sout/%x-%j.out
#SBATCH -e sout/%x-%j.err

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ldm-pl 
cd /u/$USER/dmr-latent-diffusion

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Good practice for NCCL on multi-GPU/multi-task
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Each task uses 1 GPU; Lightning runs DDP
srun python -u main.py \
  --base configs/autoencoder/autoencoder_ACCESS.yaml \
  -t \
  --gpus 1 \
   --resume /u/nstjohn/dmr-latent-diffusion/logs/2025-08-27T20-09-27_autoencoder_ACCESS/checkpoints/last.ckpt
# (older PL: --gpus 1 --distributed_backend ddp)
