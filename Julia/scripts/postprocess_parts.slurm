#!/bin/bash
#SBATCH --job-name=SB_postprocess
#SBATCH --output=../logs/postprocess_parts.out
#SBATCH --error=../logs/postprocess_parts.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load julia

PARTS_DIR="$SLURM_SUBMIT_DIR/../output/parts"
OUTPUT_CSV="$SLURM_SUBMIT_DIR/../output/csv/sweeper_coupled_full_grid.csv"

julia --project=../ -t 8 data_postprocess_parts.jl "$PARTS_DIR" "$OUTPUT_CSV"
