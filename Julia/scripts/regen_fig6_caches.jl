# Recompute the caches behind the flexibility figure (kappa sweep thrust, F_T_star, alpha vs kappa)
# with whatever grid SURFERBOT_GRID selects. Run from Julia/ with --threads N.
using Surferbot, JLD2
module TS; include(joinpath(@__DIR__, "plot_thrust_sweeps.jl")); end
module AL; include(joinpath(@__DIR__, "plot_alpha_sweep_kappa.jl")); end
bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
println("grid = ", get(ENV, "SURFERBOT_GRID", "legacy"), "  deep_X = ", get(ENV, "SURFERBOT_DEEP_X", "4.5"))
F = TS.compute_F_T_star(bp)
println("F_T_star = ", F)
sw2 = TS.run_sweep_kappa(bp)
d = isfile(TS.CACHE_PATH) ? JLD2.load(TS.CACHE_PATH) : Dict{String,Any}()
d["kap_x"] = sw2.x; d["kap_T"] = sw2.thrust; d["kap_Sxx"] = sw2.Sxx; d["F_T_star"] = F
mkpath(dirname(TS.CACHE_PATH)); JLD2.save(TS.CACHE_PATH, d)
sw = AL.run_sweep(bp)
sp = AL.surferbot_alpha(bp)
mkpath(dirname(AL.CACHE_PATH))
JLD2.save(AL.CACHE_PATH, "log10_kappa", sw.log10_kappa, "kappa", sw.kappa, "alpha", sw.alpha, "sp_kappa", sp.kappa, "sp_alpha", sp.alpha)
println("saved caches")
