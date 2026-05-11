"""
plot_thrust_sweeps.jl

Three separate figures, each with two curves (Numerics, Longuet-Higgins) and
a star marking the surferbot operating point:
  1. Motor-position sweep  x = xM/L        (ν = 0)
  2. Stiffness sweep       x = κ  (log)    (ν = 0)
  3. Reynolds sweep        x = Re (log)    (ν swept; y also log)

Output: output/figures/thrust_sweep_{xM,kappa,Re}.{pdf,png}
Cache:  output/jld2/thrust_sweeps.jld2

Usage:
  julia --project=. scripts/plot_thrust_sweeps.jl
"""

using Surferbot
using JLD2
using Plots
using LaTeXStrings
using Printf

const CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "thrust_sweeps.jld2")
const FIG_DIR    = joinpath(@__DIR__, "..", "output", "figures")
const N_SWEEP    = 50
const NU_WATER   = 1e-6

# ─── Per-solve extraction ─────────────────────────────────────────────────────
function compute_Sxx(result)
    args = result.metadata.args
    k    = Float64(real(args.k))
    pref = Float64(args.rho) * Float64(args.g) / 4 +
           3/4 * Float64(args.sigma) * k^2
    m    = Surferbot.Analysis.beam_edge_metrics(result)
    return pref * (abs2(m.eta_left_domain) - abs2(m.eta_right_domain))
end

function solve_one(bp_overrides, bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, bp_overrides)
    res = Surferbot.flexible_solver(p)
    d   = Float64(res.metadata.args.d)
    return res.thrust / d, compute_Sxx(res)
end

# ─── Three sweeps ─────────────────────────────────────────────────────────────
function run_sweep_xM(bp)
    L  = Float64(bp.L_raft)
    xs = collect(range(0.0, 0.48; length = N_SWEEP))
    T   = Vector{Float64}(undef, N_SWEEP)
    Sxx = Vector{Float64}(undef, N_SWEEP)
    println("Sweep 1/3: motor position ($N_SWEEP points) …")
    for (i, xM_norm) in enumerate(xs)
        T[i], Sxx[i] = solve_one((motor_position = xM_norm * L, nu = 0.0), bp)
        @printf "  [%2d/%d]  xM/L=%.3f   T/d=%+.3e   Sxx=%+.3e\n" i N_SWEEP xM_norm T[i] Sxx[i]
    end
    return (; x = xs, thrust = T, Sxx)
end

function run_sweep_kappa(bp)
    rho_R    = Float64(bp.rho_raft)
    L        = Float64(bp.L_raft)
    omega    = Float64(bp.omega)
    xM       = Float64(bp.motor_position)
    EI_scale = rho_R * L^4 * omega^2

    log10_kappa = collect(range(-4.0, 1.0; length = N_SWEEP))
    kappa_vals  = 10.0 .^ log10_kappa
    T   = Vector{Float64}(undef, N_SWEEP)
    Sxx = Vector{Float64}(undef, N_SWEEP)
    println("Sweep 2/3: stiffness κ ($N_SWEEP points) …")
    for (i, lk) in enumerate(log10_kappa)
        EI_i = 10.0^lk * EI_scale
        T[i], Sxx[i] = solve_one((EI = EI_i, motor_position = xM, nu = 0.0), bp)
        @printf "  [%2d/%d]  log10(κ)=%.2f   T/d=%+.3e   Sxx=%+.3e\n" i N_SWEEP lk T[i] Sxx[i]
    end
    return (; x = kappa_vals, thrust = T, Sxx)
end

function run_sweep_Re(bp)
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    xM    = Float64(bp.motor_position)
    EI    = Float64(bp.EI)

    log10_nu = collect(range(log10(NU_WATER / 100), log10(NU_WATER * 100); length = N_SWEEP))
    Re_vals  = (omega * L^2) ./ (10.0 .^ log10_nu)

    T   = Vector{Float64}(undef, N_SWEEP)
    Sxx = Vector{Float64}(undef, N_SWEEP)
    println("Sweep 3/3: Reynolds ($N_SWEEP points) …")
    for (i, lnu) in enumerate(log10_nu)
        nu_i = 10.0^lnu
        T[i], Sxx[i] = solve_one((EI = EI, motor_position = xM, nu = nu_i), bp)
        @printf "  [%2d/%d]  Re=%.2e   T/d=%+.3e   Sxx=%+.3e\n" i N_SWEEP Re_vals[i] T[i] Sxx[i]
    end
    return (; x = Re_vals, thrust = T, Sxx)
end

# ─── Surferbot operating point ────────────────────────────────────────────────
function surferbot_point(bp)
    T, Sxx  = solve_one((nu = NU_WATER,), bp)
    rho_R   = Float64(bp.rho_raft)
    L       = Float64(bp.L_raft)
    omega   = Float64(bp.omega)
    EI      = Float64(bp.EI)
    kappa   = EI / (rho_R * L^4 * omega^2)
    Re      = omega * L^2 / NU_WATER
    xM_norm = Float64(bp.motor_position) / L
    return (; xM_norm, kappa, Re, thrust = T, Sxx)
end

# ─── Cache ────────────────────────────────────────────────────────────────────
function load_or_compute(bp)
    if isfile(CACHE_PATH)
        println("Loading cache from $CACHE_PATH …")
        d   = JLD2.load(CACHE_PATH)
        sw1 = (; x = d["xM_x"],  thrust = d["xM_T"],  Sxx = d["xM_Sxx"])
        sw2 = (; x = d["kap_x"], thrust = d["kap_T"], Sxx = d["kap_Sxx"])
        sw3 = (; x = d["re_x"],  thrust = d["re_T"],  Sxx = d["re_Sxx"])
        sp  = (; xM_norm = d["sp_xM"], kappa = d["sp_kap"], Re = d["sp_Re"],
                thrust = d["sp_T"], Sxx = d["sp_Sxx"])
        return sw1, sw2, sw3, sp
    end

    sw1 = run_sweep_xM(bp)
    sw2 = run_sweep_kappa(bp)
    sw3 = run_sweep_Re(bp)
    sp  = surferbot_point(bp)

    mkpath(dirname(CACHE_PATH))
    JLD2.save(CACHE_PATH,
        "xM_x",  sw1.x,  "xM_T",  sw1.thrust, "xM_Sxx",  sw1.Sxx,
        "kap_x", sw2.x,  "kap_T", sw2.thrust, "kap_Sxx", sw2.Sxx,
        "re_x",  sw3.x,  "re_T",  sw3.thrust, "re_Sxx",  sw3.Sxx,
        "sp_xM", sp.xM_norm, "sp_kap", sp.kappa, "sp_Re", sp.Re,
        "sp_T",  sp.thrust,  "sp_Sxx", sp.Sxx)
    println("Saved cache → $CACHE_PATH")
    return sw1, sw2, sw3, sp
end

# ─── Plot style ───────────────────────────────────────────────────────────────
const BASE_OPTS = (
    legend     = :topright,
    background_color_legend = RGBA(1, 1, 1, 0.85),
    foreground_color_legend = :black,
    size       = (1094, 380),
    dpi        = 220,
    bottom_margin = 12Plots.mm,
    left_margin   = 16Plots.mm,
    top_margin    = 5Plots.mm,
    right_margin  = 5Plots.mm,
    framestyle = :box,
    grid       = false,
    guidefontsize  = 21,
    tickfontsize   = 18,
    legendfontsize = 17,
    fontfamily = "Computer Modern",
)

function make_panel(sw, xlabel_str, sp_x, sp_T, sp_S, norm;
                    log_x = false, xticks = :auto)
    yt         = sw.thrust .* norm .* 1e6
    yS         = sw.Sxx    .* norm .* 1e6
    ylabel_str = L"$F_T\,/\,(\rho_R L^2 \omega^2)$"
    sp_y       = sp_T * norm * 1e6

    p = plot(sw.x, yt;
             label      = "Numerics",
             color      = :royalblue, linewidth = 2.5,
             xlabel     = xlabel_str,
             ylabel     = ylabel_str,
             xscale     = log_x ? :log10 : :identity,
             xticks     = xticks,
             BASE_OPTS...)

    plot!(p, sw.x, yS;
          label     = "Longuet-Higgins",
          color     = :crimson, linewidth = 2.5, linestyle = :dash)

    hline!(p, [0.0]; color = :black, linewidth = 0.8, linestyle = :dot, label = false)

    scatter!(p, [sp_x], [sp_y];
             marker           = :star5, markersize = 14,
             color            = RGB(0.95, 0.75, 0.05),
             markerstrokecolor = :black, markerstrokewidth = 1,
             label            = "Surferbot")

    # ×10⁻⁶ offset label at top of y-axis, mimicking matplotlib's exponent convention
    xl = xlims(p); yl = ylims(p)
    annotate!(p, xl[1], yl[2],
        text(L"$\times\!10^{-6}$", :left, :bottom, 13, "Computer Modern"))

    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    sw1, sw2, sw3, sp = load_or_compute(bp)

    d     = Float64(bp.d)
    rho_R = Float64(bp.rho_raft)
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    norm  = d / (rho_R * L^2 * omega^2)   # F_T = sw.thrust * d  →  F_T/(ρ_R L² ω²)

    p1 = make_panel(sw1,
        L"$x_M / L$",
        sp.xM_norm, sp.thrust, sp.Sxx, norm)

    p2 = make_panel(sw2,
        L"$\kappa$",
        sp.kappa, sp.thrust, sp.Sxx, norm;
        log_x = true, xticks = 10.0 .^ collect(-4:1))

    p3 = make_panel(sw3,
        L"$Re$",
        sp.Re, sp.thrust, sp.Sxx, norm;
        log_x = true, xticks = 10.0 .^ collect(4:8))

    mkpath(FIG_DIR)
    for (fig, name) in [(p1, "thrust_sweep_xM"), (p2, "thrust_sweep_kappa"), (p3, "thrust_sweep_Re")]
        savefig(fig, joinpath(FIG_DIR, name * ".pdf"))
        savefig(fig, joinpath(FIG_DIR, name * ".png"))
        println("Saved $(joinpath(FIG_DIR, name)).{pdf,png}")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
