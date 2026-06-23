#!/bin/bash
#SBATCH --job-name=SB_regen_caches
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load julia

export GKSwstype=nul

# Regenerate thrust_sweeps.jld2 (plot_thrust_sweeps.jl self-caches via load_or_compute)
julia --project="$SLURM_SUBMIT_DIR/../" --threads 4 "$SLURM_SUBMIT_DIR/plot_thrust_sweeps.jl"

# Regenerate alpha_sweep_kappa_farfield.jld2 (self-caching script)
julia --project="$SLURM_SUBMIT_DIR/../" "$SLURM_SUBMIT_DIR/plot_alpha_sweep_kappa.jl"
