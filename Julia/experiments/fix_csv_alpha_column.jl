# One-off data repair: recompute the "alpha" column of the cluster-generated
# sweep CSVs in place, using the now-fixed Surferbot.Analysis.beam_asymmetry
# and the domain-edge eta columns already present in each row. Fixes two bugs
# at once: (1) the old sign convention, (2) the old column used beam edges
# (near-field) instead of domain edges (far-field), inconsistent with every
# other alpha in the paper. No resimulation -- pure column recompute.
using Surferbot
using CSV
using DataFrames

for fname in ["sweeper_coupled_full_grid.csv", "sweeper_uncoupled_full_grid.csv"]
    path = joinpath(@__DIR__, "..", "output", "csv", fname)
    df = CSV.read(path, DataFrame)
    old_alpha = copy(df.alpha)
    new_alpha = [
        Surferbot.Analysis.beam_asymmetry(
            complex(row.eta_1_domain_re, row.eta_1_domain_im),
            complex(row.eta_end_domain_re, row.eta_end_domain_im))
        for row in eachrow(df)
    ]
    df.alpha = new_alpha
    CSV.write(path, df)
    n_sign_flip = count(sign.(old_alpha) .!= sign.(new_alpha))
    max_abs_diff_excl_sign = maximum(abs.(abs.(old_alpha) .- abs.(new_alpha)))
    println("$fname: rewrote alpha column ($(nrow(df)) rows). ",
            "sign changed in $n_sign_flip rows; max |old|-|new| discrepancy = $max_abs_diff_excl_sign ",
            "(should be ~0 if this was a pure sign+scope fix, not a magnitude bug)")
end
