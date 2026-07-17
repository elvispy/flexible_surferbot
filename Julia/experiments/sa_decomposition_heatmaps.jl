using Surferbot
using JLD2
using LinearAlgebra
using Plots
using LaTeXStrings
using Printf

const JULIA_DIR = joinpath(@__DIR__, "..")
const OUTPUT_DIR = joinpath(JULIA_DIR, "output")
include(joinpath(JULIA_DIR, "scripts", "plot_dimensionless_diagnostics_LH.jl"))

"""
sa_decomposition_heatmaps.jl

Diagnostic heatmaps of the even/odd (S/A) endpoint decomposition used in the
appendix's α = -2Re(SA*)/(|S|²+|A|²) derivation (see main.tex, `eq:alpha_SA_app`):
log10|S|, log10|A|, cos(arg(S)-arg(A)), and α itself for reference, over the
(x_M/L, κ) plane. Confirms visually that a net thrust requires both parities
at once (α ≈ 0 exactly where the cos(arg S - arg A) sign flips or one of
|S|/|A| collapses), rather than testing anything not already covered by the
production sweeps -- kept here as a standalone diagnostic, not a paper figure.

Usage:
  julia --project=. experiments/sa_decomposition_heatmaps.jl
"""

function build_sa_grid(; n_xM::Int = 200, n_k::Int = 200)
    art_cpl = load_sweep(joinpath(OUTPUT_DIR, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    ctx = theoretical_modal_context_LH(art_cpl.base_params; output_dir = OUTPUT_DIR)
    shift = log10(ctx.params.rho_raft * ctx.derived.L_c^4 * ctx.params.omega^2)

    xM_grid   = collect(range(-0.5, 0.0; length = n_xM))
    logK_grid = collect(range(-4.0, 1.0; length = n_k))

    absS   = zeros(n_xM, n_k)
    absA   = zeros(n_xM, n_k)
    cosdphi = zeros(n_xM, n_k)
    alpha  = zeros(n_xM, n_k)

    for (j, logK) in enumerate(logK_grid)
        EI = 10.0^(logK + shift)
        for (i, xM) in enumerate(xM_grid)
            q = solve_theoretical_modal_response(EI, xM, ctx)
            d = theoretical_endpoint_diagnostics_LH(q, ctx)
            absS[i, j] = abs(d.S)
            absA[i, j] = abs(d.A)
            cosdphi[i, j] = cos(angle(d.S) - angle(d.A))
            # alpha = (|eta_left|^2 - |eta_right|^2) / denom (left minus right,
            # matching Surferbot.Analysis.beam_asymmetry's convention). Do NOT
            # copy the "right-minus-left" shape seen elsewhere in this file
            # (plot_dimensionless_diagnostics_LH.jl's internal alpha_col) --
            # that's a pre-existing, deliberately-tolerated quirk that's
            # harmless ONLY because those call sites consume abs(alpha_col),
            # never its sign. This heatmap displays the signed value, so the
            # sign must be right: eta_LH_1 is the LEFT end, eta_LH_end is the
            # RIGHT end (see this file's own docstring above).
            alpha[i, j] = (abs2(d.eta_LH_1) - abs2(d.eta_LH_end)) / (abs2(d.eta_LH_1) + abs2(d.eta_LH_end))
        end
    end

    return (; xM_grid, logK_grid, absS, absA, cosdphi, alpha)
end

function plot_sa_heatmaps(grid)
    common = (
        xlabel = L"x_M/L",
        ylabel = L"\log_{10}\kappa",
        size = (600, 450),
        framestyle = :box,
        bottom_margin = 8Plots.mm,
        left_margin = 10Plots.mm,
        top_margin = 4Plots.mm,
    )

    p1 = heatmap(grid.xM_grid, grid.logK_grid, log10.(grid.absS)';
        title = L"\log_{10}|S|", colormap = :viridis, common...)
    p2 = heatmap(grid.xM_grid, grid.logK_grid, log10.(grid.absA)';
        title = L"\log_{10}|A|", colormap = :viridis, common...)
    p3 = heatmap(grid.xM_grid, grid.logK_grid, grid.cosdphi';
        title = L"\cos(\arg S - \arg A)", colormap = cgrad(:RdBu, rev = true), clims = (-1, 1), common...)
    p4 = heatmap(grid.xM_grid, grid.logK_grid, grid.alpha';
        title = L"\alpha \;\mathrm{(for\ reference)}", colormap = cgrad(:RdBu, rev = true), clims = (-1, 1), common...)

    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 900))
end

function main()
    grid = build_sa_grid()

    csv_path = joinpath(OUTPUT_DIR, "csv", "sa_decomposition_grid.csv")
    mkpath(dirname(csv_path))
    open(csv_path, "w") do io
        println(io, "xM,logK,absS,absA,cosdphi,alpha")
        for (j, logK) in enumerate(grid.logK_grid), (i, xM) in enumerate(grid.xM_grid)
            @printf(io, "%.6f,%.6f,%.6e,%.6e,%.6f,%.6f\n",
                xM, logK, grid.absS[i, j], grid.absA[i, j], grid.cosdphi[i, j], grid.alpha[i, j])
        end
    end
    println("Saved $csv_path")

    fig_path = joinpath(OUTPUT_DIR, "figures", "sa_decomposition_heatmaps.pdf")
    mkpath(dirname(fig_path))
    p = plot_sa_heatmaps(grid)
    savefig(p, fig_path)
    println("Saved $fig_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
