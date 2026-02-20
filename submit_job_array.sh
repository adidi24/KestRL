#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-8:00:00
#SBATCH --mail-user=<your-email>
#SBATCH --mail-type=ALL
#SBATCH --job-name=kestrl-array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

# --- Configuration ---
WORKDIR="<path-to>/KestRL"
ENV_VARS="HYDRA_FULL_ERROR=1 MUJOCO_GL=egl"
EXTRA_ARGS="algorithm=sac_mujoco environment=mujoco"
# EXTRA_ARGS="algorithm=pbsac_mujoco environment=mujoco"

# --- Seed array (one task per seed) ---
SEEDS=(0 1 2 3 4)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID — seed $CURRENT_SEED"

# --- Execution ---
set -e
cd "$WORKDIR"
mkdir -p logs

# Adjust for your cluster's module system
module load Programming_Languages/python/3.12.2

env $ENV_VARS python -m kestrl.experiments.run \
    experiment.seed="$CURRENT_SEED" \
    $EXTRA_ARGS

echo "Task $SLURM_ARRAY_TASK_ID finished."
