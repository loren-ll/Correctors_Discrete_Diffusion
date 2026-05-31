#!/bin/bash
#SBATCH --job-name=diffusion_mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # adjust to however many cores your node has
#SBATCH --mem=64G                 # adjust to your problem size
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
# Uncomment whichever applies to your cluster setup:
# module load python/3.11
# source ~/venvs/diffusion/bin/activate
# conda activate diffusion

# joblib uses loky workers; tell it how many cores SLURM gave us
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

# Prevent numpy/scipy from spawning their own threads and fighting with joblib
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs

echo "Job $SLURM_JOB_ID running on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

python your_experiment_script.py --n_jobs $SLURM_CPUS_PER_TASK

echo "End: $(date)"
