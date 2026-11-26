# Recompute only the alpha-versus-kappa cache (N_COARSE points plus refinement).
using Surferbot, JLD2
module AL; include(joinpath(@__DIR__, "plot_alpha_sweep_kappa.jl")); end
bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
sw = AL.run_sweep(bp)
sp = AL.surferbot_alpha(bp)
mkpath(dirname(AL.CACHE_PATH))
JLD2.save(AL.CACHE_PATH, "log10_kappa", sw.log10_kappa, "kappa", sw.kappa, "alpha", sw.alpha, "sp_kappa", sp.kappa, "sp_alpha", sp.alpha)
println("alpha points: ", length(sw.kappa))
