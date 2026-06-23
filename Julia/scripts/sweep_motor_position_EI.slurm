#!/bin/bash
#SBATCH --job-name=SB_modal_sweep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-300

# Purpose: Memory-efficient parallel modal sweep (Uncoupled + Coupled)
# This script streams results directly to CSV, bypassing OOM issues.

module load julia

export GKSwstype=nul

# SLURM runs the batch script from the spool dir, not the submission dir.
# Use $SLURM_SUBMIT_DIR (set by sbatch) for all paths.
julia --project="$SLURM_SUBMIT_DIR/../" "$SLURM_SUBMIT_DIR/sweep_motor_position_EI.jl" $SLURM_ARRAY_TASK_ID both
