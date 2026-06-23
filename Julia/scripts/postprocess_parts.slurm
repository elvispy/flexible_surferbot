#!/bin/bash
#SBATCH --job-name=SB_postprocess
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load julia

export GKSwstype=nul

PARTS_DIR="$SLURM_SUBMIT_DIR/../output/parts"
OUTPUT_CSV="$SLURM_SUBMIT_DIR/../output/csv/sweeper_coupled_full_grid.csv"

julia --project="$SLURM_SUBMIT_DIR/../" -t 8 "$SLURM_SUBMIT_DIR/postprocess_parts.jl" "$PARTS_DIR" "$OUTPUT_CSV"
