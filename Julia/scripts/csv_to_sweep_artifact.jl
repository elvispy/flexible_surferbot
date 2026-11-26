# Build the SweepArtifact files that the modal-figure scripts load for
# `base_params` and the EI axis, from the regenerated 30k CSVs.
# Summaries are filled from the CSV columns that exist (eta at beam/domain ends);
# power/thrust are not stored in the CSV and are set to NaN.
using Surferbot, CSV, DataFrames
using Surferbot.Sweep: SweepArtifact, SweepSummary, save_sweep

out = joinpath(@__DIR__, "..", "output", "jld2"); mkpath(out)
for (name, preset) in (("coupled", Surferbot.default_coupled_motor_position_EI_sweep),
                       ("uncoupled", Surferbot.default_uncoupled_motor_position_EI_sweep))
    df = CSV.read(joinpath(@__DIR__, "..", "output", "csv", "sweeper_$(name)_full_grid.csv"), DataFrame)
    p = preset().base_params
    EI = sort(unique(10 .^ df.log10_EI)); xm = sort(unique(df.xM_over_L)) .* p.L_raft
    ie = Dict(round(l; digits=9) => i for (i, l) in enumerate(sort(unique(df.log10_EI))))
    ix = Dict(round(v; digits=9) => i for (i, v) in enumerate(sort(unique(df.xM_over_L))))
    S = Array{SweepSummary}(undef, length(xm), length(EI))
    for r in eachrow(df)
        c(a, b) = complex(a, b)
        S[ix[round(r.xM_over_L; digits=9)], ie[round(r.log10_EI; digits=9)]] = SweepSummary(
            NaN, NaN, NaN, NaN,
            c(r.eta_1_beam_re, r.eta_1_beam_im), c(r.eta_end_beam_re, r.eta_end_beam_im),
            c(r.eta_1_domain_re, r.eta_1_domain_im), c(r.eta_end_domain_re, r.eta_end_domain_im),
            NaN, NaN, NaN)
    end
    axes_ = (motor_position = xm, EI = EI)
    art = SweepArtifact{2, typeof(axes_)}("$(name)_motor_position_EI_regen", p, axes_, S)
    save_sweep(joinpath(out, "sweep_motor_position_EI_$(name)_from_matlab.jld2"), art)
    println("wrote $name: $(size(S))")
end
