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

# Staleness analysis: sweep base params have nu=0, sigma=72.2mN/m (water).
# Fix 1 (O(2/Re)) and Fix 2 viscous term vanish exactly at nu=0.
# Fix 2 Lambda/We term is O(6.7e-9) — below double precision for all quantities.
# Therefore: CSV sweeps and alpha_sweep_kappa_farfield.jld2 are NOT stale.
# Only thrust_sweeps.jld2 is stale: Re sweep and sp_re run at nu=nu_water.
# Delete the old cache so load_or_compute recomputes those components.
rm -f "$SLURM_SUBMIT_DIR/../output/jld2/thrust_sweeps.jld2"

julia --project="$SLURM_SUBMIT_DIR/../" --threads 4 "$SLURM_SUBMIT_DIR/plot_thrust_sweeps.jl"
