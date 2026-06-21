#!/bin/bash
#SBATCH --job-name=SB_modal_sweep
#SBATCH --output=../logs/modal_sweep_%a.out
#SBATCH --error=../logs/modal_sweep_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-300

# Purpose: Memory-efficient parallel modal sweep (Uncoupled + Coupled)
# This script streams results directly to CSV, bypassing OOM issues.

module load julia

# Pass the SLURM_ARRAY_TASK_ID as the task_id to handle one EI slice
# and 'both' to run uncoupled and coupled sequentially within that task.
julia --project=../ sweep_motor_position_EI.jl $SLURM_ARRAY_TASK_ID both
