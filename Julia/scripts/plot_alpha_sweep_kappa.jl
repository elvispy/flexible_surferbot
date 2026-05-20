"""
plot_alpha_sweep_kappa.jl

Beam asymmetry α vs flexural parameter κ (log scale), mirroring the style
of thrust_sweep_kappa.pdf.

  α = (|η̂(ℓ)|² − |η̂(−ℓ)|²) / (|η̂(ℓ)|² + |η̂(−ℓ)|²)

where ±ℓ are the raft beam endpoints (ν = 0).

Output: output/figures/alpha_sweep_kappa.{pdf,png}
Cache:  output/jld2/alpha_sweep_kappa.jld2

Usage:
  julia --project=. scripts/plot_alpha_sweep_kappa.jl
"""

using Surferbot
using JLD2
using Plots
using LaTeXStrings
using Printf

const CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "alpha_sweep_kappa.jld2")
const FIG_DIR    = joinpath(@__DIR__, "..", "output", "figures")
const N_SWEEP    = 50
const NU_WATER   = 1e-6

# ─── Per-solve extraction ─────────────────────────────────────────────────────

function solve_alpha(bp_overrides, bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, bp_overrides)
    res = Surferbot.flexible_solver(p)
    m   = Surferbot.Analysis.beam_edge_metrics(res)
    return Surferbot.Analysis.beam_asymmetry(m.eta_left_beam, m.eta_right_beam)
end

# ─── Sweep ────────────────────────────────────────────────────────────────────

function run_sweep(bp)
    rho_R    = Float64(bp.rho_raft)
    L        = Float64(bp.L_raft)
    omega    = Float64(bp.omega)
    xM       = Float64(bp.motor_position)
    EI_scale = rho_R * L^4 * omega^2

    log10_kappa = collect(range(-4.0, 1.0; length = N_SWEEP))
    alpha = Vector{Float64}(undef, N_SWEEP)

    println("Sweeping κ ($N_SWEEP points, ν = 0) …")
    for (i, lk) in enumerate(log10_kappa)
        EI_i     = 10.0^lk * EI_scale
        alpha[i] = solve_alpha((EI = EI_i, motor_position = xM, nu = 0.0), bp)
        @printf "  [%2d/%d]  log₁₀κ = %+.2f   α = %+.4f\n" i N_SWEEP lk alpha[i]
    end
    return (; log10_kappa, kappa = 10.0 .^ log10_kappa, alpha)
end

function surferbot_alpha(bp)
    rho_R = Float64(bp.rho_raft)
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    EI    = Float64(bp.EI)
    kappa = EI / (rho_R * L^4 * omega^2)
    alpha = solve_alpha((nu = NU_WATER,), bp)
    return (; kappa, alpha)
end

# ─── Cache ────────────────────────────────────────────────────────────────────

function load_or_compute(bp)
    if isfile(CACHE_PATH)
        println("Loading cache from $CACHE_PATH …")
        d  = JLD2.load(CACHE_PATH)
        sw = (; log10_kappa = d["log10_kappa"], kappa = d["kappa"], alpha = d["alpha"])
        sp = (; kappa = d["sp_kappa"], alpha = d["sp_alpha"])
        return sw, sp
    end

    sw = run_sweep(bp)
    sp = surferbot_alpha(bp)

    mkpath(dirname(CACHE_PATH))
    JLD2.save(CACHE_PATH,
        "log10_kappa", sw.log10_kappa,
        "kappa",       sw.kappa,
        "alpha",       sw.alpha,
        "sp_kappa",    sp.kappa,
        "sp_alpha",    sp.alpha)
    println("Saved cache → $CACHE_PATH")
    return sw, sp
end

# ─── Plot ─────────────────────────────────────────────────────────────────────

const BASE_OPTS = (
    legend     = :bottomright,
    background_color_legend = RGBA(1, 1, 1, 0.85),
    foreground_color_legend = :black,
    size       = (1094, 380),
    dpi        = 220,
    bottom_margin = 12Plots.mm,
    left_margin   = 10Plots.mm,
    top_margin    =  5Plots.mm,
    right_margin  =  5Plots.mm,
    framestyle = :box,
    grid       = false,
    guidefontsize  = 21,
    tickfontsize   = 18,
    legendfontsize = 17,
    fontfamily = "Computer Modern",
)

function make_plot(sw, sp)
    pad  = 0.10 * (maximum(sw.alpha) - minimum(sw.alpha))
    ylim = (minimum(sw.alpha) - pad, maximum(sw.alpha) + pad)

    p = plot(sw.kappa, sw.alpha;
        label     = L"\alpha",
        color     = :royalblue,
        linewidth = 2.5,
        xlabel    = L"$\kappa$",
        ylabel    = L"$\alpha$",
        xscale    = :log10,
        xticks    = 10.0 .^ collect(-4:1),
        xlims     = (10.0^-4, 10.0^1),
        ylims     = ylim,
        BASE_OPTS...,
    )

    hline!(p, [0.0]; color = :black, linewidth = 0.8, linestyle = :dot, label = false)

    scatter!(p, [sp.kappa], [sp.alpha];
        marker            = :star5,
        markersize        = 14,
        color             = RGB(0.95, 0.75, 0.05),
        markerstrokecolor = :black,
        markerstrokewidth = 1,
        label             = "SurferBot",
    )

    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    bp      = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    sw, sp  = load_or_compute(bp)
    p       = make_plot(sw, sp)

    mkpath(FIG_DIR)
    for ext in ("pdf", "png")
        fname = joinpath(FIG_DIR, "alpha_sweep_kappa.$ext")
        savefig(p, fname)
        println("Saved $fname")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
