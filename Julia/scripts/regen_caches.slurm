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

# alpha_sweep_kappa_farfield.jld2 runs at nu=0 so it is not stale after the
# kappa edge BC fix. Only thrust_sweeps.jld2 is stale (Re sweep + sp_re use
# nonzero nu). Delete the old cache so load_or_compute recomputes everything.
rm -f "$SLURM_SUBMIT_DIR/../output/jld2/thrust_sweeps.jld2"

julia --project="$SLURM_SUBMIT_DIR/../" --threads 4 "$SLURM_SUBMIT_DIR/plot_thrust_sweeps.jl"
