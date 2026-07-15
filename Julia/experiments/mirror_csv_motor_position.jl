# One-off data repair: mirror the cluster-generated sweep CSVs' motor-position
# axis in place (x_M -> -x_M), to match the corrected sign convention (real
# SurferBot: motor 0.6cm LEFT of raft center) now used throughout src/analysis.jl
# and every plot_*.jl script.
#
# Safe to do as a pure column transform, WITHOUT a resimulation, because:
#   - xM_over_L, alpha, and the four eta_1/eta_end x beam/domain endpoint
#     columns are AGGREGATE quantities with verified, robust antisymmetry
#     (T(-xM)=-T(xM), alpha(-xM)=-alpha(xM), confirmed numerically to ~1-3%,
#     consistent with ordinary solver tolerance -- see MILESTONES.md M2).
#   - No embedded paper figure reads the per-mode q_w*/Q_w*/F_w* columns
#     directly from these CSVs (confirmed via grep across all plot_*.jl) --
#     those columns are therefore NOT touched here and are LEFT STALE
#     (still representing the old, un-mirrored xM). Do not rely on them
#     without a real resweep (Phase 2).
#
# Transform per row:
#   xM_over_L  -> -xM_over_L
#   alpha      -> -alpha
#   eta_1_beam_re/im    <-> eta_end_beam_re/im    (swap)
#   eta_1_domain_re/im  <-> eta_end_domain_re/im  (swap)
using CSV
using DataFrames

for fname in ["sweeper_coupled_full_grid.csv", "sweeper_uncoupled_full_grid.csv"]
    path = joinpath(@__DIR__, "..", "output", "csv", fname)
    df = CSV.read(path, DataFrame)

    df.xM_over_L = -df.xM_over_L
    df.alpha     = -df.alpha

    for (a, b) in [(:eta_1_beam_re, :eta_end_beam_re), (:eta_1_beam_im, :eta_end_beam_im),
                   (:eta_1_domain_re, :eta_end_domain_re), (:eta_1_domain_im, :eta_end_domain_im)]
        tmp = copy(df[!, a])
        df[!, a] = df[!, b]
        df[!, b] = tmp
    end

    CSV.write(path, df)
    println("$fname: mirrored $(nrow(df)) rows (xM_over_L, alpha negated; eta_1<->eta_end swapped). ",
            "q_w*/Q_w*/F_w* columns left untouched/stale -- not used by any embedded figure.")
end
