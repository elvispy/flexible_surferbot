# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_thrust_sweeps.jl

Four separate figures, each with two curves (Numerics, Longuet-Higgins) and
a star marking the surferbot operating point where applicable:
  1. Motor-position sweep  x = xM/L        (ν = 0, finite EI = κ-fixed)
  2. Stiffness sweep       x = κ  (log)    (ν = 0)
  3. Reynolds sweep        x = Re (log)    (ν swept; y also log)
  4. Motor-position sweep  x = xM/L        (ν = 0, EI = Inf / rigid limit)

Output: output/figures/plot_thrust_sweeps_{xM,kappa,Re,xM_rigid}.{pdf,png}
Cache:  output/jld2/thrust_sweeps.jld2
Scale:  F_T^* is cached from the inviscid rigid Surferbot reference case
        (ν = 0, EI = Inf; all other parameters at the Surferbot point).

Usage:
  julia --project=. --threads auto scripts/plot_thrust_sweeps.jl
"""

using Surferbot
using JLD2
using CairoMakie
using LaTeXStrings
import Plots
using Printf
using CSV
using DataFrames
using Base.Threads: @threads, ReentrantLock

const PRINT_LOCK   = ReentrantLock()
const MAX_WORKERS  = min(Threads.nthreads(), 8)
const SOLVE_SEM    = Base.Semaphore(MAX_WORKERS)
const CACHE_PATH   = joinpath(@__DIR__, "..", "output", "jld2", "thrust_sweeps.jld2")
const ALPHA_CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "alpha_sweep_kappa_farfield.jld2")
const GRID_ALPHA_CSV = joinpath(@__DIR__, "..", "output", "csv", "sweeper_coupled_full_grid.csv")
const FIG_DIR    = joinpath(@__DIR__, "..", "output", "figures")
const N_SWEEP    = 50
const NU_WATER   = 1e-6
const RIGID_INVISCID_OVERRIDES = (nu = 0.0, EI = Inf)
const BLUE = RGBf(0.10, 0.30, 0.80)
const RED = RGBf(0.78, 0.12, 0.18)
const ALPHA_COLOR = RGBf(0.00, 0.45, 0.25)
const GOLD = RGBf(0.84, 0.55, 0.10)
const GRAY = RGBf(0.25, 0.25, 0.25)
# The paper body (jfm.cls) uses plain LaTeX default Computer Modern, and
# Plots.jl/GR's fontfamily="Computer Modern" (see plot_fig4_Aguero2026.jl)
# resolves to the same family. "Latin Modern" is a distinct, only visually-
# similar redesign -- using it here made every Makie-generated label/tick
# subtly mismatched against the rest of the paper. cm-unicode ships true
# Computer Modern as OTF, so text glyphs now match exactly. MathTeXEngine
# still needs a math font with an OpenType MATH table for symbol layout;
# cm-unicode ships no such file, so math symbols keep Latin Modern Math
# (harder to notice a mismatch on symbols than on the italic and digit
# glyphs of the tick labels/legend text most of these figures show).
const NEWCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
const LM_FONT = joinpath(NEWCM_DIR, "NewCM10-Regular.otf")
const NEWCM_MATH = joinpath(NEWCM_DIR, "NewCMMath-Regular.otf")
const XM_HIGHLIGHTS = [-0.12, -0.1885, -0.272]

function setup_lm_mathfonts()
    MTE_ID = Base.PkgId(Base.UUID("0a4f8689-d25c-4efe-a92b-7142dfc1aa53"), "MathTeXEngine")
    MTE = get(Base.loaded_modules, MTE_ID, nothing)
    MTE === nothing && return
    isfile(NEWCM_MATH) || return
    try
        MTE.set_texfont_family!(
            regular    = joinpath(NEWCM_DIR, "NewCM10-Regular.otf"),
            italic     = joinpath(NEWCM_DIR, "NewCM10-Italic.otf"),
            bold       = joinpath(NEWCM_DIR, "NewCM10-Bold.otf"),
            bolditalic = joinpath(NEWCM_DIR, "NewCM10-BoldItalic.otf"),
            math       = NEWCM_MATH,
        )
    catch
    end
end
const KAPPA_HIGHLIGHTS = [2.1209508879201904e-3, 6.8665e-3, 1.698244e-2]
const RIGID_XM_VALUES = collect(range(-0.48, 0.48; length = 2 * N_SWEEP - 1))

# ─── Per-solve extraction ─────────────────────────────────────────────────────
"""
    compute_Sxx(result) -> Float64

Far-field radiation-stress thrust measure `DeltaS_xx` (2D, force per unit width).

SIGN CONVENTION (matches the paper, Sec. 2.6):

    DeltaS_xx = (rho*g/4 + 3*sigma*k^2/4) * ( |eta_left|^2 - |eta_right|^2 )

with `eta_left` = eta(-l) the LEFT (-x) domain-edge amplitude and `eta_right` =
eta(+l) the RIGHT one. By momentum balance this equals the mean horizontal thrust
on the raft, `F_T = DeltaS_xx`, positive in +x. The left-minus-right ordering
encodes recoil: the raft is propelled opposite to its stronger radiation. Note
this is MINUS the Longuet-Higgins radiation-stress difference (which is the force
needed to hold the raft fixed, not the thrust). Same sign as `beam_asymmetry`
(`alpha`) and the `calculate_surferbot_outputs` thrust.
"""
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

function solve_one_with_alpha(bp_overrides, bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, bp_overrides)
    res = Surferbot.flexible_solver(p)
    d   = Float64(res.metadata.args.d)
    m   = Surferbot.Analysis.beam_edge_metrics(res)
    alpha = Surferbot.Analysis.beam_asymmetry(m.eta_left_domain, m.eta_right_domain)
    return res.thrust / d, compute_Sxx(res), alpha
end

function solve_alpha(bp_overrides, bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, bp_overrides)
    res = Surferbot.flexible_solver(p)
    m   = Surferbot.Analysis.beam_edge_metrics(res)
    return Surferbot.Analysis.beam_asymmetry(m.eta_left_domain, m.eta_right_domain)
end

function compute_F_T_star(bp)
    p   = Surferbot.Sweep.apply_parameter_overrides(bp, RIGID_INVISCID_OVERRIDES)
    res = Surferbot.flexible_solver(p)
    F_T_star = Float64(res.thrust)
    isfinite(F_T_star) || error("Rigid-inviscid reference thrust F_T^* is not finite")
    F_T_star != 0.0 || error("Rigid-inviscid reference thrust F_T^* is zero")
    return F_T_star
end

# ─── Three sweeps ─────────────────────────────────────────────────────────────
const XM_SWEEP_KAPPA = 6.8665e-3   # κ value for the motor-position sweep (Fig 5b snapshot)

# Lazy/incremental helper shared by the xM and kappa sweeps: any point in
# `existing_x` that already matches a `desired_x` value (to within `rtol`) is
# reused as-is; only genuinely new points get solved. Without this, widening
# a sweep (e.g. adding a few points to resolve a feature) means re-solving
# the entire grid, and some points near resonances take ~1 min instead of
# ~1 s -- the difference between a few minutes and hours.
function solve_missing_x(existing_x, existing_thrust, existing_Sxx, desired_x, solve_fn; label = "")
    new_x = missing_re_values(existing_x, desired_x)
    if isempty(new_x)
        return existing_x, existing_thrust, existing_Sxx
    end
    println("$(label)Solving $(length(new_x)) new point(s) of $(length(desired_x)) desired, $MAX_WORKERS workers …")
    new_T   = Vector{Float64}(undef, length(new_x))
    new_Sxx = Vector{Float64}(undef, length(new_x))
    @threads for i in eachindex(new_x)
        Base.acquire(SOLVE_SEM)
        try
            new_T[i], new_Sxx[i] = solve_fn(new_x[i])
        finally
            Base.release(SOLVE_SEM)
        end
        lock(PRINT_LOCK) do
            @printf "  [%2d/%d]  x=%+.4e   T/d=%+.3e   Sxx=%+.3e\n" i length(new_x) new_x[i] new_T[i] new_Sxx[i]
        end
    end
    x_all      = vcat(existing_x, new_x)
    thrust_all = vcat(existing_thrust, new_T)
    Sxx_all    = vcat(existing_Sxx, new_Sxx)
    order      = sortperm(x_all)
    return x_all[order], thrust_all[order], Sxx_all[order]
end

function run_sweep_xM(bp; existing_x = Float64[], existing_thrust = Float64[], existing_Sxx = Float64[])
    L        = Float64(bp.L_raft)
    rho_R    = Float64(bp.rho_raft)
    omega    = Float64(bp.omega)
    EI_xM    = XM_SWEEP_KAPPA * rho_R * L^4 * omega^2
    xs = collect(range(-0.48, 0.0; length = N_SWEEP))
    println("Sweep 1/4: motor position ($N_SWEEP points) …")
    solve_fn(xM_norm) = solve_one((motor_position = xM_norm * L, nu = 0.0, EI = EI_xM), bp)
    x, T, Sxx = solve_missing_x(existing_x, existing_thrust, existing_Sxx, xs, solve_fn)
    return (; x, thrust = T, Sxx)
end

function run_sweep_kappa(bp; existing_x = Float64[], existing_thrust = Float64[], existing_Sxx = Float64[])
    rho_R    = Float64(bp.rho_raft)
    L        = Float64(bp.L_raft)
    omega    = Float64(bp.omega)
    xM       = Float64(bp.motor_position)
    EI_scale = rho_R * L^4 * omega^2

    log10_kappa = collect(range(-4.0, 1.0; length = N_SWEEP))
    kappa_vals  = 10.0 .^ log10_kappa
    println("Sweep 2/4: stiffness κ ($N_SWEEP points) …")
    solve_fn(kappa) = solve_one((EI = kappa * EI_scale, motor_position = xM, nu = 0.0), bp)
    x, T, Sxx = solve_missing_x(existing_x, existing_thrust, existing_Sxx, kappa_vals, solve_fn)
    return (; x, thrust = T, Sxx)
end

function desired_Re_values(bp)
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    log10_nu = collect(range(log10(NU_WATER / 100), log10(NU_WATER * 100); length = N_SWEEP))
    base_Re  = (omega * L^2) ./ (10.0 .^ log10_nu)
    return sort(unique(base_Re))
end

function run_sweep_Re(bp, Re_vals = desired_Re_values(bp))
    L     = Float64(bp.L_raft)
    omega = Float64(bp.omega)
    xM    = Float64(bp.motor_position)

    T   = Vector{Float64}(undef, length(Re_vals))
    Sxx = Vector{Float64}(undef, length(Re_vals))
    println("Sweep 3/4: Reynolds ($(length(Re_vals)) points) …")
    @threads for i in eachindex(Re_vals)
        nu_i = omega * L^2 / Re_vals[i]
        Base.acquire(SOLVE_SEM)
        try
            T[i], Sxx[i] = solve_one((EI = Inf, motor_position = xM, nu = nu_i), bp)
        finally
            Base.release(SOLVE_SEM)
        end
        lock(PRINT_LOCK) do
            @printf "  [%2d/%d]  Re=%.2e   T/d=%+.3e   Sxx=%+.3e\n" i length(Re_vals) Re_vals[i] T[i] Sxx[i]
        end
    end
    return (; x = Re_vals, thrust = T, Sxx)
end

function merge_rigid_xM_sweeps(existing, new)
    x = vcat(Float64.(existing.x), Float64.(new.x))
    thrust = vcat(Float64.(existing.thrust), Float64.(new.thrust))
    Sxx = vcat(Float64.(existing.Sxx), Float64.(new.Sxx))
    alpha = vcat(Float64.(existing.alpha), Float64.(new.alpha))
    order = sortperm(x)
    return (; x = x[order], thrust = thrust[order], Sxx = Sxx[order], alpha = alpha[order])
end

function missing_re_values(existing_x, desired_x)
    out = Float64[]
    for x in desired_x
        any(isapprox(x, y; rtol = 1e-10, atol = 0.0) for y in existing_x) || push!(out, x)
    end
    return out
end

function same_grid(existing_x, desired_x)
    length(existing_x) == length(desired_x) || return false
    return all(isapprox(Float64(x), Float64(y); rtol = 1e-10, atol = 1e-12) for (x, y) in zip(existing_x, desired_x))
end

function run_sweep_xM_rigid(bp, xs = RIGID_XM_VALUES)
    L     = Float64(bp.L_raft)
    T     = Vector{Float64}(undef, length(xs))
    Sxx   = Vector{Float64}(undef, length(xs))
    alpha = Vector{Float64}(undef, length(xs))
    println("Sweep 4/4: motor position rigid EI=Inf ($(length(xs)) points) …")
    @threads for i in eachindex(xs)
        xM_norm = xs[i]
        Base.acquire(SOLVE_SEM)
        try
            T[i], Sxx[i], alpha[i] = solve_one_with_alpha((motor_position = xM_norm * L, nu = 0.0, EI = Inf), bp)
        finally
            Base.release(SOLVE_SEM)
        end
        lock(PRINT_LOCK) do
            @printf "  [%2d/%d]  xM/L=%.3f   T/d=%+.3e   Sxx=%+.3e   α=%+.4f\n" i length(xs) xM_norm T[i] Sxx[i] alpha[i]
        end
    end
    return (; x = xs, thrust = T, Sxx, alpha)
end

function run_alpha_xM(bp)
    L        = Float64(bp.L_raft)
    rho_R    = Float64(bp.rho_raft)
    omega    = Float64(bp.omega)
    EI_xM    = XM_SWEEP_KAPPA * rho_R * L^4 * omega^2
    xs = collect(range(-0.48, 0.0; length = N_SWEEP))
    alpha = Vector{Float64}(undef, N_SWEEP)
    println("Alpha sweep: flexible motor position ($N_SWEEP points) …")
    @threads for i in 1:N_SWEEP
        xM_norm = xs[i]
        Base.acquire(SOLVE_SEM)
        try
            alpha[i] = solve_alpha((motor_position = xM_norm * L, nu = 0.0, EI = EI_xM), bp)
        finally
            Base.release(SOLVE_SEM)
        end
        lock(PRINT_LOCK) do
            @printf "  [%2d/%d]  xM/L=%.3f   α=%+.4f\n" i length(xs) xM_norm alpha[i]
        end
    end
    return (; x = xs, alpha)
end


# ─── Surferbot operating point ────────────────────────────────────────────────
function surferbot_point(bp; nu = 0.0)
    T, Sxx  = solve_one((nu = nu, EI = Inf), bp)
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
function save_cache(sw1, sw2, sw3, sw4, sp, F_T_star, sp_re)
    mkpath(dirname(CACHE_PATH))
    JLD2.save(CACHE_PATH,
        "xM_x",       sw1.x,  "xM_T",       sw1.thrust, "xM_Sxx",       sw1.Sxx,
        "kap_x",      sw2.x,  "kap_T",       sw2.thrust, "kap_Sxx",      sw2.Sxx,
        "re_x",       sw3.x,  "re_T",        sw3.thrust, "re_Sxx",       sw3.Sxx,
        "xM_rig_x",   sw4.x,  "xM_rig_T",    sw4.thrust, "xM_rig_Sxx",   sw4.Sxx,
        "xM_rig_alpha", sw4.alpha,
        "sp_xM", sp.xM_norm, "sp_kap", sp.kappa, "sp_Re", sp.Re,
        "sp_T",  sp.thrust,  "sp_Sxx", sp.Sxx,
        "F_T_star", F_T_star,
        "sp_re_T", sp_re.thrust, "sp_re_Sxx", sp_re.Sxx)
end

function load_or_compute(bp)
    d = isfile(CACHE_PATH) ? (println("Loading cache from $CACHE_PATH …"); JLD2.load(CACHE_PATH)) : Dict{String,Any}()
    changed = false

    existing1 = all(k -> haskey(d, k), ["xM_x", "xM_T", "xM_Sxx"]) ?
        (Float64.(d["xM_x"]), Float64.(d["xM_T"]), Float64.(d["xM_Sxx"])) : (Float64[], Float64[], Float64[])
    sw1 = run_sweep_xM(bp; existing_x = existing1[1], existing_thrust = existing1[2], existing_Sxx = existing1[3])
    if length(sw1.x) != length(existing1[1])
        GC.gc(); changed = true
        d["xM_x"] = sw1.x; d["xM_T"] = sw1.thrust; d["xM_Sxx"] = sw1.Sxx
    end

    existing2 = all(k -> haskey(d, k), ["kap_x", "kap_T", "kap_Sxx"]) ?
        (Float64.(d["kap_x"]), Float64.(d["kap_T"]), Float64.(d["kap_Sxx"])) : (Float64[], Float64[], Float64[])
    sw2 = run_sweep_kappa(bp; existing_x = existing2[1], existing_thrust = existing2[2], existing_Sxx = existing2[3])
    if length(sw2.x) != length(existing2[1])
        GC.gc(); changed = true
        d["kap_x"] = sw2.x; d["kap_T"] = sw2.thrust; d["kap_Sxx"] = sw2.Sxx
    end

    if all(k -> haskey(d, k), ["re_x", "re_T", "re_Sxx"])
        sw3 = (; x = d["re_x"], thrust = d["re_T"], Sxx = d["re_Sxx"])
    else
        sw3 = run_sweep_Re(bp); GC.gc(); changed = true
        d["re_x"] = sw3.x; d["re_T"] = sw3.thrust; d["re_Sxx"] = sw3.Sxx
    end

    if all(k -> haskey(d, k), ["sp_xM", "sp_kap", "sp_Re", "sp_T", "sp_Sxx"])
        sp = (; xM_norm = d["sp_xM"], kappa = d["sp_kap"], Re = d["sp_Re"],
               thrust = d["sp_T"], Sxx = d["sp_Sxx"])
    else
        sp = surferbot_point(bp; nu = 0.0); changed = true
        d["sp_xM"] = sp.xM_norm; d["sp_kap"] = sp.kappa; d["sp_Re"] = sp.Re
        d["sp_T"]  = sp.thrust;  d["sp_Sxx"] = sp.Sxx
    end

    if haskey(d, "F_T_star")
        F_T_star = Float64(d["F_T_star"])
    else
        F_T_star = compute_F_T_star(bp); changed = true
        d["F_T_star"] = F_T_star
    end

    if haskey(d, "sp_re_T")
        sp_re = (; sp.Re, thrust = Float64(d["sp_re_T"]), Sxx = Float64(d["sp_re_Sxx"]))
    else
        sp_re = surferbot_point(bp; nu = NU_WATER); changed = true
        d["sp_re_T"] = sp_re.thrust; d["sp_re_Sxx"] = sp_re.Sxx
    end

    if all(k -> haskey(d, k), ["xM_rig_x", "xM_rig_T", "xM_rig_Sxx"])
        haskey(d, "xM_rig_alpha") || error("Rigid motor-position cache lacks alpha values. Regenerate the rigid motor-position sweep.")
        sw4 = (; x = d["xM_rig_x"], thrust = d["xM_rig_T"], Sxx = d["xM_rig_Sxx"], alpha = d["xM_rig_alpha"])
        missing_xM = missing_re_values(sw4.x, RIGID_XM_VALUES)
        if !isempty(missing_xM)
            println("Computing $(length(missing_xM)) missing rigid motor-position points …")
            sw4 = merge_rigid_xM_sweeps(sw4, run_sweep_xM_rigid(bp, missing_xM))
            GC.gc(); changed = true
            d["xM_rig_x"] = sw4.x; d["xM_rig_T"] = sw4.thrust; d["xM_rig_Sxx"] = sw4.Sxx
            d["xM_rig_alpha"] = sw4.alpha
        end
    else
        sw4 = run_sweep_xM_rigid(bp); GC.gc(); changed = true
        d["xM_rig_x"] = sw4.x; d["xM_rig_T"] = sw4.thrust; d["xM_rig_Sxx"] = sw4.Sxx
        d["xM_rig_alpha"] = sw4.alpha
    end

    if changed
        save_cache(sw1, sw2, sw3, sw4, sp, F_T_star, sp_re)
        println("Cache updated → $CACHE_PATH")
    end
    return sw1, sw2, sw3, sw4, sp, F_T_star, sp_re
end

# ─── Plot style ───────────────────────────────────────────────────────────────
function load_alpha_sweep()
    isfile(ALPHA_CACHE_PATH) || error("Missing alpha sweep cache: $ALPHA_CACHE_PATH. Run Julia/scripts/plot_alpha_sweep_kappa.jl first.")
    d = JLD2.load(ALPHA_CACHE_PATH)
    return (; kappa = d["kappa"], alpha = d["alpha"])
end

function load_motor_alpha_from_csv(bp; target_kappa)
    isfile(GRID_ALPHA_CSV) || error("Missing alpha grid CSV: $GRID_ALPHA_CSV")
    df = CSV.read(GRID_ALPHA_CSV, DataFrame)
    shift = log10(Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2)
    logk = df.log10_EI .- shift
    target = isnothing(target_kappa) ? maximum(logk) : log10(target_kappa)
    nearest = sort(unique(logk))[argmin(abs.(sort(unique(logk)) .- target))]
    mask = abs.(logk .- nearest) .< 1e-10
    rows = df[mask, :]
    order = sortperm(rows.xM_over_L)
    label = isnothing(target_kappa) ? @sprintf("stiffest grid κ=%.3g", 10.0^nearest) :
                                      @sprintf("grid κ=%.3g", 10.0^nearest)
    println("Loaded motor-position alpha from $label")
    return (; x = Float64.(rows.xM_over_L[order]), alpha = Float64.(rows.alpha[order]))
end


function panel_limits(y1, y2; include_zero=true)
    vals = vcat(y1, y2)
    if include_zero
        vals = vcat(vals, 0.0)
    end
    lo, hi = minimum(vals), maximum(vals)
    pad = 0.08 * max(hi - lo, eps())
    return (lo - pad, hi + pad)
end

function makie_figure()
    set_theme!(Theme(
        fonts = (; regular = LM_FONT),
        fontsize = 26,
        Axis = (;
            xlabelsize = 29,
            ylabelsize = 29,
            xticklabelsize = 26,
            yticklabelsize = 26,
            xticklabelfont = LM_FONT,
            yticklabelfont = LM_FONT,
            xlabelfont = LM_FONT,
            ylabelfont = LM_FONT,
            xgridvisible = false,
            ygridvisible = false,
            topspinevisible = true,
            rightspinevisible = true,
            bottomspinevisible = true,
            leftspinevisible = true,
        ),
        Legend = (;
            labelsize = 28,
            framevisible = true,
            patchsize = (55, 23),
        ),
    ))
    return Figure(size = (1500, 400), backgroundcolor = :white)
end

function add_dual_axis!(fig, sw, alpha_sw, d, F_T_star; xlabel, xscale=identity,
                        xticks=Makie.automatic, ylims=nothing, show_Sxx=true,
                        show_zero=true, highlight_x=Float64[], legend_position=:rb,
                        xlims=nothing)
    yt = sw.thrust .* d ./ abs(F_T_star)
    yS = sw.Sxx .* d ./ abs(F_T_star)
    ylim = isnothing(ylims) ? panel_limits(yt, show_Sxx ? yS : yt) : ylims
    xlim = isnothing(xlims) ? (minimum(sw.x), maximum(sw.x)) : xlims
    order = sortperm(sw.x)
    alpha_order = sortperm(alpha_sw.x)

    ax = Axis(fig[1, 1];
        xlabel,
        xscale, xticks, ytickformat = x -> [@sprintf("%.0f", v) for v in x],
        limits = (xlim, ylim),
        alignmode = Makie.Mixed(left = Makie.Protrusion(125), right = Makie.Protrusion(90)))

    Label(fig[1, 1, Makie.Left()], L"\text{Normalized thrust}", rotation = pi/2, fontsize = 29, font = LM_FONT)
    l1 = lines!(ax, sw.x[order], yt[order]; color = BLUE, linewidth = 3)
    handles = [l1]
    labels = [L"F_T/F_T^\ast"]
    if show_Sxx
        l2 = lines!(ax, sw.x[order], yS[order]; color = RED, linewidth = 3, linestyle = :dash)
        push!(handles, l2); push!(labels, L"\Delta S_{xx}/F_T^\ast")
    end
    if show_zero
        hlines!(ax, [0.0]; color = (:black, 0.55), linewidth = 1)
    end
    if !isempty(highlight_x)
        vlines!(ax, highlight_x; color = (GRAY, 0.75), linestyle = :dash, linewidth = 1.5)
    end

    axr = Axis(fig[1, 1];
        xscale, xticks,
        yaxisposition = :right,
        ylabel = L"\alpha",
        ylabelcolor = ALPHA_COLOR,
        yticklabelcolor = ALPHA_COLOR,
        rightspinecolor = ALPHA_COLOR,
        ytickcolor = ALPHA_COLOR,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = :transparent,
        limits = (xlim, (-1.1, 1.1)),
        ytickformat = vals -> [@sprintf("%.1f", v) for v in vals],
        alignmode = Makie.Mixed(right = Makie.Protrusion(90)))
    hidespines!(axr, :l, :b, :t)
    hidexdecorations!(axr; grid = false)
    l3 = lines!(axr, alpha_sw.x[alpha_order], alpha_sw.alpha[alpha_order];
        color = ALPHA_COLOR, linewidth = 3)
    push!(handles, l3); push!(labels, L"\alpha")

    Makie.colsize!(fig.layout, 1, Makie.Fixed(1285))
    Makie.rowsize!(fig.layout, 1, Makie.Fixed(355))

    axislegend(ax, handles, labels; position = legend_position,
        backgroundcolor = (:white, 0.86), framecolor = (:black, 0.45))
    return fig
end

function make_sweep_panel(sw, alpha_sw, d, F_T_star; xlabel, outfile,
                          xscale=identity, xticks=Makie.automatic,
                          ylims=nothing, show_Sxx=true, show_zero=true,
                          highlight_x=Float64[], legend_position=:rb,
                          grid_style=false, xlims=nothing)
    fig = makie_figure()
    add_dual_axis!(fig, sw, alpha_sw, d, F_T_star; xlabel, xscale, xticks,
        ylims, show_Sxx, show_zero, highlight_x, legend_position, xlims)
    Makie.resize_to_layout!(fig)
    save(outfile * ".pdf", fig)
    save(outfile * ".png", fig; px_per_unit = 2)
    println("Saved $outfile.{pdf,png}")
end

function make_single_axis_panel(sw, d, F_T_star; xlabel, outfile,
                                xscale=identity, xticks=Makie.automatic,
                                ylims=nothing, show_zero=true,
                                marker_point=nothing)
    yt = sw.thrust .* d ./ abs(F_T_star)
    order = sortperm(sw.x)
    ylim = isnothing(ylims) ? panel_limits(yt, yt) : ylims
    
    Plots.gr()
    
    p = Plots.plot(sw.x[order], yt[order];
        color          = Plots.RGB(0.10, 0.30, 0.80),
        linewidth      = 4.0,
        label          = false,
        xlabel         = xlabel,
        ylabel         = L"F_T/F_T^\ast",
        xscale         = xscale == log10 ? :log10 : :identity,
        xticks         = xticks,
        ylims          = ylim,
        grid           = true,
        framestyle     = :box,
        fontfamily     = "Computer Modern",
        guidefontsize  = 39,
        tickfontsize   = 29,
        size           = (1100, 530),
        dpi            = 220,
        left_margin    = 16Plots.mm,
        bottom_margin  = 14Plots.mm,
        top_margin     = 4Plots.mm,
        right_margin   = 5Plots.mm,
    )

    if !isnothing(marker_point)
        Plots.scatter!(p, [marker_point.x], [marker_point.y];
            color = Plots.RGB(0.84, 0.55, 0.10),
            markerstrokecolor = :black,
            markerstrokewidth = 1.6,
            markersize = 14,
            marker = :star5,
            label = "SurferBot",
            legend = :topright,
            legendfontsize = 29
        )
    end
    
    if show_zero
        Plots.hline!(p, [0.0]; color = :black, alpha = 0.55, linewidth = 1, label = false)
    end
    
    Plots.savefig(p, outfile * ".pdf")
    Plots.savefig(p, outfile * ".png")
    println("Saved $outfile.{pdf,png} (via Plots.jl)")
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    setup_lm_mathfonts()
    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    sw1, sw2, sw3, sw4, sp, F_T_star, sp_re = load_or_compute(bp)
    alpha_kappa = load_alpha_sweep()
    alpha_kappa_sw = (; x = alpha_kappa.kappa, alpha = alpha_kappa.alpha)
    alpha_xM = load_motor_alpha_from_csv(bp; target_kappa = XM_SWEEP_KAPPA)
    alpha_xM_rigid = (; x = sw4.x, alpha = sw4.alpha)

    d     = Float64(bp.d)
    @printf "Using F_T^* = %+.6e N from rigid-inviscid Surferbot reference\n" F_T_star

    mkpath(FIG_DIR)
    make_sweep_panel(sw1, alpha_xM, d, F_T_star;
        xlabel = L"x_M/L",
        outfile = joinpath(FIG_DIR, "plot_thrust_sweeps_xM"),
        highlight_x = XM_HIGHLIGHTS)
    make_sweep_panel(sw2, alpha_kappa_sw, d, F_T_star;
        xlabel = L"\kappa",
        outfile = joinpath(FIG_DIR, "plot_thrust_sweeps_kappa"),
        xscale = log10,
        xticks = (10.0 .^ collect(-4:1),
                  [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
        ylims = (-32.0, 32.0),
        highlight_x = KAPPA_HIGHLIGHTS,
        legend_position = :rb)
    yt3  = sw3.thrust .* d ./ abs(F_T_star)
    pad3 = 0.08 * (maximum(yt3) - minimum(yt3))
    make_single_axis_panel(sw3, d, F_T_star;
        xlabel = L"Re",
        outfile = joinpath(FIG_DIR, "plot_thrust_sweeps_Re"),
        xscale = log10,
        xticks = (10.0 .^ collect(4:8),
                  [L"10^{4}", L"10^{5}", L"10^{6}", L"10^{7}", L"10^{8}"]),
        ylims = (minimum(yt3) - pad3, maximum(yt3) + pad3),
        marker_point = (; x = sp_re.Re, y = sp_re.thrust * d / abs(F_T_star)),
        show_zero = false)
    make_sweep_panel(sw4, alpha_xM_rigid, d, F_T_star;
        xlabel = L"x_M/L",
        outfile = joinpath(FIG_DIR, "plot_thrust_sweeps_xM_rigid"),
        xticks = ([-0.5, -0.25, 0.0, 0.25, 0.5],
                  [L"-0.5", L"-0.25", L"0", L"0.25", L"0.5"]),
        xlims = (-0.5, 0.5),
        highlight_x = Float64[],
        grid_style = true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
