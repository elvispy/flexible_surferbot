"""
Locally refines the cached kappa-thrust sweep (output/jld2/thrust_sweeps.jld2)
around the three known resonances in figure 6 (plot_kappa_snapshot_grid_flexibility),
which were visibly under-resolved: the base 201-point global grid (~0.025
decades spacing) captured each peak's height correctly but not its
descending flank, producing a faceted rather than smooth apex.

After running this, regenerate the figures with the normal scripts:
  julia --project=. scripts/plot_thrust_sweeps.jl
  julia --project=. scripts/plot_kappa_snapshot.jl --snapshot-grids
"""

include(joinpath(@__DIR__, "..", "scripts", "plot_thrust_sweeps.jl"))

bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params

d = isfile(CACHE_PATH) ? JLD2.load(CACHE_PATH) : Dict{String,Any}()
existing_x = Float64.(d["kap_x"]); existing_T = Float64.(d["kap_T"]); existing_S = Float64.(d["kap_Sxx"])

rho_R = Float64(bp.rho_raft); L = Float64(bp.L_raft); omega = Float64(bp.omega)
xM    = Float64(bp.motor_position)
EI_scale = rho_R * L^4 * omega^2

# Local refinement windows around the three known resonances, each far finer
# than the base 201-point global grid (spacing ~0.025 decades). 20 pts each
# window at ~0.006 decades spacing (~4x finer).
windowA = 10.0 .^ range(log10(3.15e-4), log10(4.35e-4); length = 20)
windowB = 10.0 .^ range(log10(1.55e-3), log10(2.00e-3); length = 20)
windowC = 10.0 .^ range(log10(1.72e-2), log10(2.10e-2); length = 20)

desired_x = sort(vcat(existing_x, windowA, windowB, windowC))

solve_fn(kappa) = solve_one((EI = kappa * EI_scale, motor_position = xM, nu = 0.0), bp)
x, T, Sxx = solve_missing_x(existing_x, existing_T, existing_S, desired_x, solve_fn;
    label = "Local kappa refinement: ",
    checkpoint = (a, b, c) -> checkpoint_sweep("kap_x", "kap_T", "kap_Sxx", a, b, c))

println("Done. Total points now: ", length(x))
