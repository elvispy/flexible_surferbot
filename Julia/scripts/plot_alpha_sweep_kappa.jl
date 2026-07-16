# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_alpha_sweep_kappa.jl

Far-field asymmetry α vs flexural parameter κ (log scale), mirroring the style
of thrust_sweep_kappa.pdf.

  α = (|η̂(+L/2)|² − |η̂(−L/2)|²) / (|η̂(+L/2)|² + |η̂(−L/2)|²)

where ±L/2 are the domain endpoints (far-field amplitudes, ν = 0).

Output: output/figures/alpha_sweep_kappa.{pdf,png}
Cache:  output/jld2/alpha_sweep_kappa_farfield.jld2

Usage:
  julia --project=. scripts/plot_alpha_sweep_kappa.jl
"""

using Surferbot
using JLD2
using Plots
using LaTeXStrings
using Printf
using Base.Threads: @threads, ReentrantLock

const PRINT_LOCK  = ReentrantLock()
const MAX_WORKERS = min(Threads.nthreads(), 8)
const SOLVE_SEM   = Base.Semaphore(MAX_WORKERS)
const CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "alpha_sweep_kappa_farfield.jld2")
const FIG_DIR    = joinpath(@__DIR__, "..", "output", "figures")
const N_COARSE   = 50
const REFINE_THRESHOLD  = 0.25  # |Δα| between adjacent coarse points that triggers refinement
const N_PER_INTERVAL    = 12    # extra points inserted into each flagged interval
const NU_WATER   = 1e-6
const RIGID_INVISCID_OVERRIDES = (nu = 0.0, EI = Inf)

# ─── Per-solve extraction ─────────────────────────────────────────────────────

function solve_alpha(bp_overrides, bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, bp_overrides)
    res = Surferbot.flexible_solver(p)
    m   = Surferbot.Analysis.beam_edge_metrics(res)
    return Surferbot.Analysis.beam_asymmetry(m.eta_left_domain, m.eta_right_domain)
end

# ─── Sweep ────────────────────────────────────────────────────────────────────
#
# Two-pass adaptive grid: a coarse log-uniform pass over the full range, then
# extra points inserted only inside intervals where α changes sharply between
# adjacent coarse points (i.e. resonance-driven features), rather than
# uniformly densifying the whole (mostly smooth) range. Keeps the total point
# count in the ~50-100 range instead of paying for fine resolution everywhere.

function run_sweep(bp)
    rho_R    = Float64(bp.rho_raft)
    L        = Float64(bp.L_raft)
    omega    = Float64(bp.omega)
    xM       = Float64(bp.motor_position)
    EI_scale = rho_R * L^4 * omega^2

    solve_at(lk) = solve_alpha((EI = 10.0^lk * EI_scale, motor_position = xM, nu = 0.0), bp)

    log10_kappa = collect(range(-4.0, 1.0; length = N_COARSE))
    alpha = Vector{Float64}(undef, N_COARSE)

    println("Sweeping κ ($N_COARSE coarse points, ν = 0, $MAX_WORKERS workers) …")
    @threads for i in eachindex(log10_kappa)
        lk = log10_kappa[i]
        Base.acquire(SOLVE_SEM)
        try
            alpha[i] = solve_at(lk)
        finally
            Base.release(SOLVE_SEM)
        end
        lock(PRINT_LOCK) do
            @printf "  [%2d/%d]  log₁₀κ = %+.2f   α = %+.4f\n" i N_COARSE lk alpha[i]
        end
    end

    steep_flagged = [i for i in 1:(N_COARSE - 1) if abs(alpha[i + 1] - alpha[i]) > REFINE_THRESHOLD]
    # Also refine the interval immediately following the steep interval that
    # ends just before the third root (kappa ~ 3.64e-4, counting from the
    # kappa~10 end): alpha settles into a local maximum right after that
    # root, but its neighboring coarse-point delta falls under
    # REFINE_THRESHOLD even though the extremum itself sits just past the
    # flagged interval, right where dense refined spacing abruptly reverts to
    # the original coarse spacing -- producing a visible kink there. Only
    # this specific neighbor is added (not a general rule for every steep
    # interval) to avoid pulling in unrelated, expensive-to-solve regions.
    third_root_interval = findlast(i -> log10_kappa[i] < log10(3.64e-4) < log10_kappa[i + 1], steep_flagged)
    steep = if third_root_interval === nothing
        steep_flagged
    else
        i0 = steep_flagged[third_root_interval]
        sort(unique(vcat(steep_flagged, i0 + 1 <= N_COARSE - 1 ? [i0 + 1] : Int[])))
    end
    if !isempty(steep)
        extra_lk = Float64[]
        for i in steep
            append!(extra_lk, range(log10_kappa[i], log10_kappa[i + 1]; length = N_PER_INTERVAL + 2)[2:end-1])
        end
        println("Refining $(length(extra_lk)) points across $(length(steep)) steep interval(s), $MAX_WORKERS workers …")
        extra_alpha = Vector{Float64}(undef, length(extra_lk))
        @threads for i in eachindex(extra_lk)
            lk = extra_lk[i]
            Base.acquire(SOLVE_SEM)
            try
                extra_alpha[i] = solve_at(lk)
            finally
                Base.release(SOLVE_SEM)
            end
            lock(PRINT_LOCK) do
                @printf "  refine [%2d/%d]  log₁₀κ = %+.4f   α = %+.4f\n" i length(extra_lk) lk extra_alpha[i]
            end
        end
        log10_kappa = vcat(log10_kappa, extra_lk)
        alpha       = vcat(alpha, extra_alpha)
        order       = sortperm(log10_kappa)
        log10_kappa = log10_kappa[order]
        alpha       = alpha[order]
    end

    return (; log10_kappa, kappa = 10.0 .^ log10_kappa, alpha)
end

function surferbot_alpha(bp)
    alpha = solve_alpha(RIGID_INVISCID_OVERRIDES, bp)
    kappa = Inf
    return (; kappa, alpha)
end

# ─── Cache ────────────────────────────────────────────────────────────────────

function load_or_compute(bp)
    if isfile(CACHE_PATH)
        println("Loading cache from $CACHE_PATH …")
        d  = JLD2.load(CACHE_PATH)
        sw = (; log10_kappa = d["log10_kappa"], kappa = d["kappa"], alpha = d["alpha"])
        sp = (; kappa = d["sp_kappa"], alpha = d["sp_alpha"])
        if !isinf(Float64(sp.kappa))
            println("Updating cached Surferbot marker to rigid-inviscid reference …")
            sp = surferbot_alpha(bp)
            JLD2.save(CACHE_PATH,
                "log10_kappa", sw.log10_kappa,
                "kappa",       sw.kappa,
                "alpha",       sw.alpha,
                "sp_kappa",    sp.kappa,
                "sp_alpha",    sp.alpha)
        end
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
    legendfontsize = 16,
    fontfamily = "Computer Modern",
)

function make_plot(sw, sp)
    pad  = 0.10 * (maximum(sw.alpha) - minimum(sw.alpha))
    ylim = (minimum(sw.alpha) - pad, maximum(sw.alpha) + pad)
    transform(x) = log10(x + 1.0)
    tick_values = [0.0; 10.0 .^ collect(0:4)]
    tick_labels = [L"0", L"10^0", L"10^1", L"10^2", L"10^3", L"10^4"]
    chi = 1.0 ./ sw.kappa
    order = sortperm(chi)

    p = plot(transform.(chi[order]), sw.alpha[order];
        label     = L"\alpha",
        color     = :royalblue,
        linewidth = 2.5,
        xlabel    = L"$1/\kappa$",
        ylabel    = L"$\alpha$",
        xticks    = (transform.(tick_values), tick_labels),
        xlims     = (-0.12, transform(1e4)),
        ylims     = ylim,
        BASE_OPTS...,
        left_margin = 9.0Plots.mm,
        legend    = :bottomleft,
    )

    hline!(p, [0.0]; color = :black, linewidth = 0.8, linestyle = :dot, label = false)

    scatter!(p, [0.0], [sp.alpha];
        marker           = :star5,
        markersize       = 14,
        color            = RGB(0.95, 0.75, 0.05),
        markerstrokecolor = :black,
        markerstrokewidth = 1,
        label            = "Surferbot")

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
