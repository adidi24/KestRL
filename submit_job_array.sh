#!/bin/bash
# KestRL SLURM array job.
#
# Two-level parallelism:
#   - Array tasks  : each task runs one configuration (env/algo combo or sweep point)
#   - Within a task: NUM_SEEDS seeds run in parallel via jax.vmap (CompiledTrainer)
#
# Typical usage:
#   sbatch submit_job_array.sh                     # run all configs
#   sbatch --array=0-1 submit_job_array.sh         # run first two configs only
#   sbatch --array=2 submit_job_array.sh           # single config by index

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=kestrl
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<your-email>
#SBATCH --array=0-1   # adjust to len(CONFIGS)-1

# ── Settings ──────────────────────────────────────────────────────────────────

WORKDIR="$HOME/KestRL"

# Number of seeds run in parallel via jax.vmap within each array task.
# Each seed is a fully independent training run.
NUM_SEEDS=10

# Base seed for this task. Seeds will be base, base+1, ..., base+NUM_SEEDS-1.
# Using TASK_ID * NUM_SEEDS ensures no seed overlap across tasks.
BASE_SEED=$(( SLURM_ARRAY_TASK_ID * NUM_SEEDS ))

# One config per array task. Add as many as needed; bump --array accordingly.
CONFIGS=(
    "environment=brax algorithm=sac_mujoco"
    "environment=cartpole algorithm=sac_cartpole"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# ── Environment ───────────────────────────────────────────────────────────────

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # avoid JAX grabbing all GPU memory upfront

# ── Run ───────────────────────────────────────────────────────────────────────

set -e
cd "$WORKDIR"
mkdir -p logs

echo "========================================"
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Config     : $CONFIG"
echo "Seeds      : $BASE_SEED → $(( BASE_SEED + NUM_SEEDS - 1 )) ($NUM_SEEDS in parallel)"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "========================================"

uv run python -m kestrl.experiments.run \
    $CONFIG \
    experiment.seed=$BASE_SEED \
    experiment.num_seeds=$NUM_SEEDS \
    experiment.track=true

echo "Task $SLURM_ARRAY_TASK_ID done."
