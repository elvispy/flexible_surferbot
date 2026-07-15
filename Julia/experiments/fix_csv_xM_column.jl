# One-off data repair: port the cluster-generated sweep CSVs from the old
# xM/L > 0 convention to the xM/L < 0 convention the rest of the codebase was
# ported to (motor_position defaults, root-finding grids, YLIMS, operating-
# point markers). These CSVs live on an unreachable local repo and were never
# regenerated to match, so this is a pure in-place relabeling -- no
# resimulation, no new solves.
#
# Verified (against the already-ported closed-form ROM solver
# `solve_theoretical_modal_response`/theory pipeline in
# plot_dimensionless_diagnostics_LH.jl -- a cheap linear solve against a
# cached operator, not a new PDE solve) at 8+ independent (EI, xM) points
# across both CSVs, plus an exact (0.0% error) check on the pure algebraic
# F_wN load projection:
#
#   1. xM_over_L                      -> negate
#   2. eta_1_beam   <-> eta_end_beam   -> swap   (re & im)
#      eta_1_domain <-> eta_end_domain -> swap   (re & im)
#      (confirmed: a bare xM_over_L negation does NOT match the theory
#      solver; swapping the endpoint columns is what matches, to ROM-vs-
#      full-solver residual (~1-10%, worse only very near resonances,
#      consistent with known ROM approximation error elsewhere in this
#      codebase) -- because "left endpoint" and "right endpoint" swap
#      physical identity when the coordinate direction flips.)
#   3. q_wN, Q_wN, F_wN for odd mode number N (1,3,5,7) -> negate (re & im)
#      even mode number N (0,2,4,6)                     -> unchanged
#      (basis-parity fact: Psi_n(-x) = (-1)^n Psi_n(x) for the free-free
#      beam mode basis used here, so ANY per-mode projection coefficient --
#      solution q_n, load F_n, or response Q_n -- inherits this same sign
#      rule under the coordinate flip, independent of the physical quantity
#      being projected. Verified exactly for F_wN (pure algebraic
#      projection, no solve) and to ROM residual for q_wN; Q_wN was not
#      independently re-derived but follows the identical basis-parity
#      argument.)
#   4. alpha -> recomputed from the (now-swapped) domain columns via
#      Surferbot.Analysis.beam_asymmetry, same convention as the existing
#      fix_csv_alpha_column.jl. Needed because load_motor_alpha_from_csv
#      (plot_thrust_sweeps.jl, plot_kappa_snapshot.jl) reads this column
#      directly rather than recomputing it from eta.
using Surferbot
using CSV
using DataFrames

const ODD_MODES  = (1, 3, 5, 7)
const EVEN_MODES = (0, 2, 4, 6)

function swap_cols!(df, a, b)
    tmp = copy(df[!, a])
    df[!, a] = df[!, b]
    df[!, b] = tmp
end

for fname in ["sweeper_coupled_full_grid.csv", "sweeper_uncoupled_full_grid.csv"]
    path = joinpath(@__DIR__, "..", "output", "csv", fname)
    df = CSV.read(path, DataFrame)
    n_rows = nrow(df)

    # 1. Negate xM_over_L
    old_extrema = extrema(df.xM_over_L)
    df.xM_over_L = -df.xM_over_L

    # 2. Swap left/right endpoint columns (beam and domain)
    swap_cols!(df, :eta_1_beam_re, :eta_end_beam_re)
    swap_cols!(df, :eta_1_beam_im, :eta_end_beam_im)
    swap_cols!(df, :eta_1_domain_re, :eta_end_domain_re)
    swap_cols!(df, :eta_1_domain_im, :eta_end_domain_im)

    # 3. Negate odd-mode q/Q/F coefficients (re & im); even modes untouched
    for n in ODD_MODES
        for prefix in ("q_w", "Q_w", "F_w")
            for part in ("re", "im")
                col = Symbol("$(prefix)$(n)_$(part)")
                df[!, col] = -df[!, col]
            end
        end
    end

    # 4. Recompute alpha from the (now-swapped) domain columns
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
    println("$fname: ported $n_rows rows to xM/L < 0 convention. ",
            "xM_over_L range $old_extrema -> $(extrema(df.xM_over_L)); ",
            "alpha sign changed in $n_sign_flip rows.")
end
