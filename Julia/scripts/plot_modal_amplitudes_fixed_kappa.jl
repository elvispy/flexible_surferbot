"""
plot_modal_amplitudes_fixed_kappa.jl

Publication-grade figure: modal amplitudes |q_n| vs x_M/L at a fixed κ.

Instead of tracking the |η_end|=0 curve through (κ, x_M) space, this script
takes a vertical slice at a single log₁₀(κ) value and sweeps x_M/L across
the full range available in the sweep data.

Output filename encodes κ so no title is needed in the figure.
"""

using Surferbot, JLD2, Plots, LaTeXStrings, Printf, CSV, DataFrames, Statistics

# ── Configuration ─────────────────────────────────────────────────────────────
const N_MODES       = 5           # modes 0 … N_MODES-1
const LOG10_KAPPA   = -2.7        # target log₁₀(κ) slice
const KAPPA_TOL     = 0.06        # half-width of slice (grid step is ~0.1)
const XM_MIN        = 0.01        # xM/L lower limit
const XM_MAX        = 0.50        # xM/L upper limit
const SMOOTH_WINDOW = 5           # moving average half-width

const JLD2_PATH = joinpath(@__DIR__, "..", "output", "jld2",
                            "sweep_motor_position_EI_coupled_from_matlab.jld2")
const CSV_PATH  = joinpath(@__DIR__, "..", "output", "csv",
                            "sweeper_coupled_full_grid.csv")
const FIG_DIR   = joinpath(@__DIR__, "..", "output", "figures")

# Okabe-Ito colorblind-safe palette (black first, then colours)
const OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2",
                   "#D55E00", "#CC79A7"]

# ── Data loading ──────────────────────────────────────────────────────────────
function load_fixed_kappa_slice(csv_path, jld2_path)
    artifact = Surferbot.Sweep.load_sweep(jld2_path)
    params   = artifact.base_params
    shift    = log10(Float64(params.rho_raft) *
                     Float64(params.L_raft)^4 *
                     Float64(params.omega)^2)

    df      = CSV.read(csv_path, DataFrame)
    logK    = df.log10_EI .- shift
    mask    = (abs.(logK .- LOG10_KAPPA) .< KAPPA_TOL) .&
              (df.xM_over_L .>= XM_MIN) .&
              (df.xM_over_L .< XM_MAX)

    sub  = df[mask, :]
    @assert nrow(sub) > 0 "No rows found for log₁₀(κ) ≈ $(LOG10_KAPPA)"

    actual_logK = mean(df.log10_EI[mask] .- shift)
    @info @sprintf("Slice: log₁₀(κ) = %.3f  (%d points, xM/L ∈ [%.3f, %.3f])",
                   actual_logK, nrow(sub),
                   minimum(sub.xM_over_L), maximum(sub.xM_over_L))

    order  = sortperm(sub.xM_over_L)
    xM_vec = Float64.(sub.xM_over_L)[order]
    Q_mat  = zeros(Float64, length(xM_vec), N_MODES)
    for (i, row_i) in enumerate(order)
        for n in 0:(N_MODES-1)
            re = Float64(sub[row_i, Symbol("q_w$(n)_re")])
            im = Float64(sub[row_i, Symbol("q_w$(n)_im")])
            Q_mat[i, n+1] = abs(complex(re, im))
        end
    end

    return xM_vec, Q_mat, actual_logK
end

# ── Smoothing ─────────────────────────────────────────────────────────────────
function moving_average(v, w)
    n   = length(v)
    out = similar(v)
    for i in 1:n
        lo       = max(1, i - w ÷ 2)
        hi       = min(n, i + w ÷ 2)
        out[i]   = mean(v[lo:hi])
    end
    return out
end

# ── Figure ────────────────────────────────────────────────────────────────────
function make_figure(xM_vec, Q_mat, actual_logK)
    mode_colors  = OKABE_ITO[1:N_MODES]
    mode_markers = [:circle, :rect, :diamond, :utriangle, :dtriangle]

    p = plot(;
        xlabel      = L"$x_M \, / \, L$",
        ylabel      = L"$|q_n|$",
        legend      = :bottomleft,
        background_color_legend = RGBA(1, 1, 1, 0.85),
        foreground_color_legend = :black,
        size        = (540, 380),       # ~single-column width
        margin      = 5Plots.mm,
        dpi         = 300,
        guidefontsize  = 14,
        tickfontsize   = 12,
        legendfontsize = 9,
        fontfamily  = "Computer Modern",
        framestyle  = :box,
        grid        = true,
        gridalpha   = 0.20,
        yscale      = :log10,
        yticks      = 10.0 .^ collect(-6:0),
    )

    for n in 0:(N_MODES-1)
        y_smooth = moving_average(Q_mat[:, n+1], SMOOTH_WINDOW)
        plot!(p, xM_vec, y_smooth;
              label     = latexstring("n = $n"),
              color     = mode_colors[n+1],
              linewidth = 2.0)
    end

    return p
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    xM_vec, Q_mat, actual_logK = load_fixed_kappa_slice(CSV_PATH, JLD2_PATH)
    p = make_figure(xM_vec, Q_mat, actual_logK)

    mkpath(FIG_DIR)
    stem    = @sprintf("modal_amplitudes_kappa_%+.2f", actual_logK)
    stem    = replace(stem, "+" => "p", "-" => "m", "." => "p")  # e.g. m2p70
    out_pdf = joinpath(FIG_DIR, stem * ".pdf")
    out_png = joinpath(FIG_DIR, stem * ".png")
    savefig(p, out_pdf)
    savefig(p, out_png)
    println("Saved $out_pdf")
    println("Saved $out_png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
