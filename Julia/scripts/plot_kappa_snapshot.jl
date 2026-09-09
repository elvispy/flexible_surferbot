# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_kappa_snapshot.jl

Two modes:

1. Single snapshot (default / --EI / --kappa):
   Two-panel figure for one operating point:
     (a) Free-surface profile η(x) with a scatter marker at the motor position
     (b) Modal amplitude |q_n|^2 in each free-free beam mode on a log axis

2. Paper snapshot series (--paper-snapshots):
   The five two-panel figures used in the manuscript, generated with a common
   modal-amplitude scale across panels.

3. Five-panel snapshot figure (--5panel):
   One-row figure with panels (a)–(e) covering three κ values and five
   distinct (κ, xM/L) operating points for comparison:
     (a) κ = 1.82e-3,  xM = Surferbot
     (b) κ = 6.8665e-3,  xM = Surferbot
     (c) κ = 6.8665e-3,  xM/L = -0.1885  (α ≈ 0)
     (d) κ = 6.8665e-3,  xM/L = -0.272  (|α| ≈ 1)
     (e) κ = 1.94e-2,  xM = Surferbot

Usage:
  julia --project=. scripts/plot_kappa_snapshot.jl [--EI VALUE] [--kappa VALUE] [--outdir DIR]
  julia --project=. scripts/plot_kappa_snapshot.jl --paper-snapshots [--outdir DIR]
  julia --project=. scripts/plot_kappa_snapshot.jl --5panel [--outdir DIR]
"""

using Surferbot
using Plots
using LaTeXStrings
using Printf
using JLD2
using CSV
using DataFrames
import CairoMakie as CM

include(joinpath(@__DIR__, "paper_plot_theme.jl"))
using .PaperPlotTheme

# ─── Parameters ──────────────────────────────────────────────────────────────

function build_params(; EI=nothing, xM_norm=nothing)
    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    if isnothing(EI) && isnothing(xM_norm)
        return Surferbot.Sweep.apply_parameter_overrides(bp, (L_domain = 0.10,))
    elseif isnothing(xM_norm)
        return Surferbot.Sweep.apply_parameter_overrides(bp, (L_domain = 0.10, EI = EI))
    elseif isnothing(EI)
        return Surferbot.Sweep.apply_parameter_overrides(bp,
            (L_domain = 0.10, motor_position = xM_norm * Float64(bp.L_raft)))
    else
        return Surferbot.Sweep.apply_parameter_overrides(bp,
            (L_domain = 0.10, EI = EI, motor_position = xM_norm * Float64(bp.L_raft)))
    end
end

# ─── Phase convention ────────────────────────────────────────────────────────

function phase_and_profile(result)
    x_cm    = result.x .* 1e2
    contact = Bool.(result.metadata.args.x_contact)
    contact_idx = findall(contact)
    beam_ends   = [contact_idx[1], contact_idx[end]]
    dom_end     = beam_ends[argmax(abs.(result.eta[beam_ends]))]
    theta       = -angle(result.eta[dom_end])
    eta_m       = real.(result.eta .* exp(im * theta))
    motor_x_cm  = Float64(result.metadata.args.motor_position) * 1e2
    return x_cm, contact, eta_m, motor_x_cm
end

# ─── Wave-profile panel ──────────────────────────────────────────────────────

const FIG1_FREE_SURFACE = "#1A4DCC"
const FIG1_RAFT = "#000000"
const FIG1_LOAD = "#A86E0D"
const MAKIE_BLUE = CM.RGBf(0.10, 0.30, 0.80)
const MAKIE_RED = CM.RGBf(0.78, 0.12, 0.18)
const MAKIE_ALPHA = CM.RGBf(0.00, 0.45, 0.25)
const MAKIE_LOAD = CM.RGBf(0.66, 0.43, 0.05)
const MAKIE_GRAY = CM.RGBf(0.25, 0.25, 0.25)
const COLUMN_COLORS = [CM.RGBf(0.58, 0.58, 0.58), CM.RGBf(0.35, 0.35, 0.35), CM.RGBf(0.10, 0.10, 0.10)]
const COLUMN_LINEWIDTHS = [1.5, 3.0, 4.5]
const COLUMN_SPINEWIDTHS = [1.0, 2.0, 3.0]
# See plot_thrust_sweeps.jl setup_lm_mathfonts() for why this is cm-unicode
# (true Computer Modern, matching the paper body and STYLE.fontfamily above)
# rather than "Latin Modern Roman".
function _kpsewhich(fname, fallback)
    p = try
        strip(read(`kpsewhich $fname`, String))
    catch
        ""
    end
    isempty(p) ? fallback : p
end
const CMU_DIR = dirname(_kpsewhich("cmunrm.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/cm-unicode/cmunrm.otf"))
const LM_FONT = PaperPlotTheme.REGULAR

function setup_lm_mathfonts()
    PaperPlotTheme.setup_mathfonts!()
end
const THRUST_CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "thrust_sweeps.jld2")
const ALPHA_CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "alpha_sweep_kappa_farfield.jld2")
const GRID_ALPHA_CSV = joinpath(@__DIR__, "..", "output", "csv", "sweeper_coupled_full_grid.csv")
# The plotted Fig 5 top-row curve interpolates between the sweep's own sample
# points, so a highlight that is not itself a sample reads as a much shallower
# dip than the true minimum. These values are sample points of the refined
# 201-point sweep. The middle one is not an extremum, it is the alpha ~ -1 case.
const KAPPA_HIGHLIGHTS = PaperPlotTheme.KAPPA_HIGHLIGHTS
const XM_HIGHLIGHTS = [-0.12, -0.1885, -0.272]
const SNAPSHOT_CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "kappa_snapshots_cache.jld2")

const STYLE = (
    framestyle    = :box,
    fontfamily    = "Computer Modern",
    guidefontsize = 21,
    tickfontsize  = 18,
)

function paper_snapshot_ops()
    bp       = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    EI_scale = Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2
    xM_sb    = Float64(bp.motor_position) / Float64(bp.L_raft)

    ops = [
        (kappa=KAPPA_HIGHLIGHTS[1], xM=xM_sb,  file_xM=nothing, label="(a)"),
        (kappa=KAPPA_HIGHLIGHTS[2], xM=xM_sb,  file_xM=nothing, label="(b)"),
        (kappa=KAPPA_HIGHLIGHTS[2], xM=-0.1885,  file_xM=-0.1885,   label="(c)"),
        (kappa=KAPPA_HIGHLIGHTS[2], xM=-0.272,  file_xM=-0.272,   label="(d)"),
        (kappa=KAPPA_HIGHLIGHTS[3], xM=xM_sb,  file_xM=nothing, label="(e)"),
    ]
    return bp, EI_scale, xM_sb, ops
end

function get_cached_snapshot(op, EI_scale)
    d = isfile(SNAPSHOT_CACHE_PATH) ? JLD2.load(SNAPSHOT_CACHE_PATH) : Dict{String,Any}()
    key_res = @sprintf("res_k%.3e_x%.3f", op.kappa, op.xM)
    key_mod = @sprintf("mod_k%.3e_x%.3f", op.kappa, op.xM)
    
    if haskey(d, key_res) && haskey(d, key_mod)
        println("Loaded snapshot from cache: κ=$(op.kappa) xM=$(op.xM)")
        return d[key_res], d[key_mod]
    else
        @info @sprintf("Solving snapshot κ=%.2e  xM/L=%.3f", op.kappa, op.xM)
        EI = op.kappa * EI_scale
        p = build_params(; EI, xM_norm=op.xM)
        result = Surferbot.flexible_solver(p)
        modal = Surferbot.decompose_raft_freefree_modes(result; num_modes=10, verbose=false)
        
        d[key_res] = result
        d[key_mod] = modal
        mkpath(dirname(SNAPSHOT_CACHE_PATH))
        JLD2.save(SNAPSHOT_CACHE_PATH, d)
        
        return result, modal
    end
end

function make_wave_panel(result; ylim=1500, small=false,
                          show_xlabel=true, show_ylabel=true)
    x_cm, contact, eta_m, motor_x_cm = phase_and_profile(result)
    L_raft = result.metadata.args.L_raft
    x_over_L = x_cm ./ (L_raft * 100)
    motor_x_over_L = motor_x_cm / (L_raft * 100)
    eta_um = eta_m .* 1e6
    motor_idx = argmin(abs.(x_over_L .- motor_x_over_L))

    fs_guide = small ? 15 : 21
    fs_tick  = small ? 12 : 18
    b_margin = small ?  6Plots.mm : 11Plots.mm
    l_margin = small ?  8Plots.mm : 14Plots.mm

    p = plot(x_over_L, eta_um;
        color     = FIG1_FREE_SURFACE,
        linewidth = 1.5,
        label     = false,
        xlabel    = show_xlabel ? L"x/L" : "",
        ylabel    = show_ylabel ? L"h\;(\mu\mathrm{m})" : "",
        xlims     = (-1.0, 1.0),
        ylims     = (-ylim, ylim),
        yticks    = show_ylabel ? (-ylim:500:ylim) : [],
        grid      = true,
        bottom_margin = b_margin,
        left_margin   = l_margin,
        top_margin    =  4Plots.mm,
        right_margin  =  2Plots.mm,
        framestyle    = :box,
        fontfamily    = "Computer Modern",
        guidefontsize = fs_guide,
        tickfontsize  = fs_tick,
    )
    plot!(p, x_over_L[contact], eta_um[contact];
        color = FIG1_RAFT, linewidth = 2.5, label = false)
    scatter!(p, [x_over_L[motor_idx]], [eta_um[motor_idx]];
        markershape       = :circle,
        markersize        = 9,
        color             = FIG1_LOAD,
        markerstrokecolor = FIG1_LOAD,
        markerstrokewidth = 1,
        label             = false)
    return p
end

# ─── Two-panel snapshot figure (original single-snapshot mode) ────────────────

function common_modal_energy_limits(modals; decades=6.0)
    all_energy = reduce(vcat, (abs.(modal.q) for modal in modals))
    positive = all_energy[all_energy .> 0]
    isempty(positive) && return (10.0^(-decades), 1.0)

    ymax = ceil(log10(maximum(positive)))
    ymin_from_data = floor(log10(minimum(positive)))
    ymin_from_span = ymax - decades
    ymin = max(ymin_from_data, ymin_from_span)
    return (10.0^ymin, 10.0^ymax)
end

function make_figure(result, modal, kappa_val, fig_dir; xM_norm=nothing,
                     modal_energy_ylims=nothing)
    p1 = make_wave_panel(result)

    mode_energy = abs.(modal.q)
    ylims = isnothing(modal_energy_ylims) ? common_modal_energy_limits([modal]) : modal_energy_ylims
    mode_energy = max.(mode_energy, ylims[1])
    p2  = bar(modal.n, mode_energy;
        xticks        = modal.n,
        xlabel        = L"n",
        ylabel        = L"|\bar{q}_n|",
        label         = false,
        fillcolor     = :steelblue,
        linecolor     = :steelblue,
        linewidth     = 0.5,
        grid          = :y,
        ylims         = ylims,
        yscale        = :log10,
        fillrange     = ylims[1],
        bottom_margin = 11Plots.mm,
        left_margin   = 12Plots.mm,
        top_margin    =  4Plots.mm,
        right_margin  =  5Plots.mm,
        STYLE...,
    )

    fig = plot(p1, p2;
        layout = grid(1, 2, widths = [0.65, 0.35]),
        size   = (1420, 300),
        dpi    = 220,
    )

    fname = if isnothing(xM_norm)
        joinpath(fig_dir, @sprintf("plot_kappa_snapshot_%.2e.pdf", kappa_val))
    else
        joinpath(fig_dir, @sprintf("plot_kappa_snapshot_%.2e_xM%.3f.pdf", kappa_val, xM_norm))
    end
    savefig(fig, fname)
    println("Saved $fname")
end

function main_paper_snapshots(fig_dir)
    _, EI_scale, _, ops = paper_snapshot_ops()

    results = []
    modals = []
    for op in ops
        result, modal = get_cached_snapshot(op, EI_scale)
        push!(results, result)
        push!(modals, modal)
    end

    modal_energy_ylims = common_modal_energy_limits(modals)
    @info @sprintf("Shared modal-amplitude y-limits: %.3e to %.3e",
                   modal_energy_ylims[1], modal_energy_ylims[2])
    for (result, modal, op) in zip(results, modals, ops)
        make_figure(result, modal, op.kappa, fig_dir;
                    xM_norm=op.file_xM, modal_energy_ylims=modal_energy_ylims)
    end
end

# ─── Five-panel figure ────────────────────────────────────────────────────────

function main_5panel(fig_dir)
    _, EI_scale, _, ops = paper_snapshot_ops()

    results = []
    for op in ops
        result, _ = get_cached_snapshot(op, EI_scale)
        push!(results, result)
    end

    # Common y-axis limit across all panels
    ylim = 0
    for res in results
        _, _, eta_m, _ = phase_and_profile(res)
        eta_um = eta_m .* 1e6
        ylim = max(ylim, maximum(abs.(eta_um)))
    end
    ylim = ceil(ylim / 500) * 500   # round up to nearest 500 µm

    panels = []
    for (i, (res, op)) in enumerate(zip(results, ops))
        p = make_wave_panel(res; ylim=ylim, small=true,
                             show_xlabel=true, show_ylabel=(i == 1))
        # Panel label in top-left corner
        annotate!(p, -4.6, ylim * 0.88, text(op.label, :left, 14, "Computer Modern"))
        push!(panels, p)
    end

    fig = plot(panels...;
        layout = grid(1, 5),
        size   = (2200, 440),
        dpi    = 300,
    )

    fname = joinpath(fig_dir, "plot_kappa_snapshot_5panel.pdf")
    savefig(fig, fname)
    println("Saved $fname")
    savefig(fig, replace(fname, ".pdf" => ".png"))
    println("Saved $(replace(fname, ".pdf" => ".png"))")
end

# ─── Composite snapshot grids ────────────────────────────────────────────────

function makie_snapshot_theme!()
    # Print scale = textwidth / native_fig_width = 468 / 1500 = 0.312.
    # Column panel labels: 26 × 0.312 = 8.1 pt
    # Column panel ticks:  22 × 0.312 = 6.9 pt
    # Sweep row uses explicit overrides in draw_sweep_axis! (29 → 9.1 pt, 26 → 8.1 pt).
    CM.set_theme!(CM.Theme(
        fonts = (; regular = LM_FONT),
        fontsize = 22,
        Axis = (;
            xlabelsize = 29,
            ylabelsize = 29,
            xticklabelsize = 26,
            yticklabelsize = 26,
            titlesize = 29,
            xticklabelfont = LM_FONT,
            yticklabelfont = LM_FONT,
            ylabelfont = LM_FONT,
            ylabelrotation = 0,
            ylabelpadding = 25,
            titlefont = LM_FONT,
            xgridvisible = false,
            ygridvisible = false,
            topspinevisible = true,
            rightspinevisible = true,
            bottomspinevisible = true,
            leftspinevisible = true,
        ),
        Legend = (;
            labelsize = 26,
            framevisible = true,
            patchsize = (44, 20),
        ),
    ))
end

function load_sweep_cache_for_grid(kind::Symbol)
    isfile(THRUST_CACHE_PATH) || error("Missing thrust cache: $THRUST_CACHE_PATH. Run scripts/plot_thrust_sweeps.jl first.")
    d = JLD2.load(THRUST_CACHE_PATH)
    scale = Float64(d["F_T_star"])
    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    depth = Float64(bp.d)
    if kind == :kappa
        isfile(ALPHA_CACHE_PATH) || error("Missing alpha cache: $ALPHA_CACHE_PATH. Run scripts/plot_alpha_sweep_kappa.jl first.")
        a = JLD2.load(ALPHA_CACHE_PATH)
        return (; x = d["kap_x"], y = d["kap_T"] .* depth ./ scale,
                ylh = d["kap_Sxx"] .* depth ./ scale,
                alpha_x = a["kappa"], alpha = a["alpha"],
                xlabel = L"\kappa",
                xscale = log10,
                xticks = (10.0 .^ collect(-4:1),
                          [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
                highlights = KAPPA_HIGHLIGHTS, F_T_star = scale)
    elseif kind == :xM
        alpha_xM = load_motor_alpha_from_csv(bp; target_kappa = 6.8665e-3)
        return (; x = d["xM_x"], y = d["xM_T"] .* depth ./ scale,
                ylh = d["xM_Sxx"] .* depth ./ scale,
                alpha_x = alpha_xM.x, alpha = alpha_xM.alpha,
                xlabel = L"x_M/L",
                xscale = identity,
                xticks = -0.5:0.1:0.0,
                highlights = XM_HIGHLIGHTS, F_T_star = scale)
    else
        error("Unknown sweep kind: $kind")
    end
end

function load_motor_alpha_from_csv(bp; target_kappa)
    isfile(GRID_ALPHA_CSV) || error("Missing alpha grid CSV: $GRID_ALPHA_CSV")
    df = CSV.read(GRID_ALPHA_CSV, DataFrame)
    shift = log10(Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2)
    logk = df.log10_EI .- shift
    target = log10(target_kappa)
    vals = sort(unique(logk))
    nearest = vals[argmin(abs.(vals .- target))]
    rows = df[abs.(logk .- nearest) .< 1e-10, :]
    order = sortperm(rows.xM_over_L)
    println(@sprintf("Loaded grid alpha for xM sweep at κ=%.3g", 10.0^nearest))
    return (; x = Float64.(rows.xM_over_L[order]), alpha = Float64.(rows.alpha[order]))
end

function draw_sweep_axis!(figpos, labelpos, sweep; legend_position = :rb,
                           sweep_labelsize = 29, sweep_ticksize = 26,
                           legend_labelsize = 28, legend_patchsize = (55, 23),
                           highlight_colors = fill(MAKIE_GRAY, length(sweep.highlights)),
                           highlight_linewidths = fill(1.5, length(sweep.highlights)),
                           ylim_override = nothing, xlim_override = nothing)
    # sweep_labelsize/sweep_ticksize override the column-panel theme defaults
    # for this full-width row (scale 0.416): 22→9.2pt, 19→7.9pt
    # `ylim_override` clips the vertical range instead of letting it autoscale.
    # The refined kappa sweep resolves a narrow resonance near the low-kappa end
    # whose true peak is ~2.3x the tallest previously sampled point; letting it
    # set the axis compresses every other feature. Clipping keeps the range the
    # rest of the paper's figures were drawn against. The clipped peak must be
    # acknowledged in the caption, since the curve then leaves the axis.
    # Horizontal restriction is resolved first, because the vertical autoscale
    # below must see only the data that is actually plotted.  Restricting the
    # low-kappa end is what lets the y axis autoscale sanely: the narrow
    # resonance below kappa ~ 1.5e-4 peaks at ~65, and if it is included the
    # axis stretches to fit it and flattens everything else.
    xlo, xhi = isnothing(xlim_override) ?
        (minimum(sweep.x), maximum(sweep.x)) : xlim_override
    vis = (sweep.x .>= xlo) .& (sweep.x .<= xhi)
    ylim = isnothing(ylim_override) ?
        1.08 * maximum(abs.(vcat(sweep.y[vis], sweep.ylh[vis], 0.0))) : ylim_override
    xt = sweep.xticks
    if !isnothing(xlim_override) && xt isa Tuple
        keep = findall(v -> xlo <= v <= xhi, xt[1])
        xt = (xt[1][keep], xt[2][keep])
    end

    ax = CM.Axis(figpos;
        xlabel = sweep.xlabel,
        xlabelsize = sweep_labelsize,
        xticklabelsize = sweep_ticksize,
        yticklabelsize = sweep_ticksize,
        ytickformat = vals -> [@sprintf("%.1f", v) for v in vals],
        xscale = sweep.xscale,
        xticks = xt,
        limits = ((xlo, xhi), (-ylim, ylim)),
        alignmode = CM.Mixed(left = CM.Protrusion(150), right = CM.Protrusion(90)))
        
    CM.Label(labelpos, L"\text{Normalized thrust}", rotation = pi/2, fontsize = sweep_labelsize, font = LM_FONT)
        
    order = sortperm(sweep.x)
    l1 = CM.lines!(ax, sweep.x[order], sweep.y[order]; color = MAKIE_BLUE, linewidth = 2.6)
    l2 = CM.lines!(ax, sweep.x[order], sweep.ylh[order]; color = MAKIE_RED, linewidth = 2.6, linestyle = :dash)
    CM.hlines!(ax, [0.0]; color = (:black, 0.55), linewidth = 0.9)
    for (xh, col, lw) in zip(sweep.highlights, highlight_colors, highlight_linewidths)
        CM.vlines!(ax, [xh]; color = (col, 0.85), linewidth = lw, linestyle = :dash)
    end

    axr = CM.Axis(figpos;
        xscale = sweep.xscale,
        xticks = xt,
        yaxisposition = :right,
        ylabel = L"\alpha",
        ylabelsize = sweep_labelsize,
        yticklabelsize = sweep_ticksize,
        ylabelfont = LM_FONT,
        yticklabelfont = LM_FONT,
        ylabelcolor = MAKIE_ALPHA,
        yticklabelcolor = MAKIE_ALPHA,
        rightspinecolor = MAKIE_ALPHA,
        ytickcolor = MAKIE_ALPHA,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = :transparent,
        limits = ((xlo, xhi), (-1.1, 1.1)),
        ytickformat = vals -> [@sprintf("%.1f", v) for v in vals],
        alignmode = CM.Mixed(right = CM.Protrusion(90)))
    CM.hidespines!(axr, :l, :b, :t)
    CM.hidexdecorations!(axr; grid = false)
    aorder = sortperm(sweep.alpha_x)
    l3 = CM.lines!(axr, sweep.alpha_x[aorder], sweep.alpha[aorder]; color = MAKIE_ALPHA, linewidth = 2.6)

    CM.axislegend(ax, [l1, l2, l3], [L"F_T/F_T^\ast", L"\Delta S_{xx}/F_T^\ast", L"\alpha"];
        position = legend_position, backgroundcolor = (:white, 0.86), framecolor = (:black, 0.45),
        labelsize = legend_labelsize, patchsize = legend_patchsize)
    return ax
end

function draw_wave_axis!(ax, result; ylim, show_ylabel, title, F_T_ratio, F_T_raw)
    x_cm, contact, eta_m, motor_x_cm = phase_and_profile(result)
    L_raft = result.metadata.args.L_raft
    x_over_L = x_cm ./ (L_raft * 100)
    motor_x_over_L = motor_x_cm / (L_raft * 100)
    eta_over_L = eta_m ./ L_raft
    motor_idx = argmin(abs.(x_over_L .- motor_x_over_L))
    CM.lines!(ax, x_over_L, eta_over_L; color = MAKIE_BLUE, linewidth = 1.8)
    CM.lines!(ax, x_over_L[contact], eta_over_L[contact]; color = :black, linewidth = 3.0)
    CM.scatter!(ax, [x_over_L[motor_idx]], [eta_over_L[motor_idx]];
        color = MAKIE_LOAD, strokecolor = MAKIE_LOAD, markersize = 15)
    ax.title = title
    ax.xlabel = L"x/L"
    ax.ylabel = show_ylabel ? L"\bar{\eta}" : ""
    CM.xlims!(ax, -1.0, 1.0)
    CM.ylims!(ax, -ylim, ylim)
    ax.yticks = [-ylim, 0.0, ylim]

    # The raft sits at |x|<2.5 with much smaller amplitude than the free-surface
    # ripples beyond it (which approach +-ylim near the domain edges), so the
    # region directly below the raft, well inside -ylim, is empty. Placed below
    # (not above) the raft: the tikz panel-tag overlay in main.tex ((b)-(g))
    # sits right above this row, so an above-raft placement collided with it.
    #
    # Direction comes from the raw (unnormalized) thrust, not sign(F_T_ratio).
    # F_T_ratio = F_T_raw/abs(F_T_star): F_T_star (the rigid-inviscid reference)
    # can itself be negative depending on which side its own reference motor
    # position sits on, so dividing by the signed value (rather than its
    # absolute value) would silently flip the displayed ratio's sign relative
    # to the true physical push direction -- making the arrow and the printed
    # F_T/F_T^* value disagree. Verified against the Fig 3b validated case
    # (motor left of centre -> raw thrust positive, pushes right, away from
    # the bigger left-side wave, matching ordinary recoil) that sign(F_T_raw)
    # is the physically meaningful one; dividing by abs(F_T_star) keeps the
    # displayed ratio's sign consistent with it too.
    dir = sign(F_T_raw)
    arrow_y = -0.45 * ylim
    label_y = -0.58 * ylim
    # Round first, then decide.  A case that displays as 0.0 has no meaningful
    # direction, so the arrow is dropped rather than pointing at rounding noise.
    # `+ 0.0` also normalises -0.0, which would otherwise print as "-0.0".
    shown = round(F_T_ratio; digits = 1) + 0.0
    if shown != 0
        arrow_x = -dir * 1.1 / (L_raft * 100)
        arrow_u = dir * 2.2 / (L_raft * 100)
        CM.arrows2d!(ax, [arrow_x], [arrow_y], [arrow_u], [0.0];
            shaftcolor = :black, tipcolor = :black, shaftwidth = 3.0, tipwidth = 18, tiplength = 18)
    end
    CM.text!(ax, 0.0, label_y;
        text = LaTeXString(@sprintf("\$F_T/F_T^\\ast = %.1f\$", shown)),
        align = (:center, :top), fontsize = 20, color = :black)
end

_mixc(a, b, w) = CM.RGBf((1-w)*a.r + w*b.r, (1-w)*a.g + w*b.g, (1-w)*a.b + w*b.b)

# Colour the modal bars by arg(q_n), measured in the same phase convention as
# the wave panel above (the dominant raft edge is taken real).  The map is
# cyclic and built from the paper's own accents: MAKIE_BLUE in phase, through
# neutral grey at +-90 deg, to MAKIE_RED anti-phase.  Amplitude sets opacity,
# because arg(q_n) is meaningless for a mode of negligible amplitude.
function phase_bar_colors(modal, result)
    L = result.metadata.args.L_raft
    ci = findall(abs.(result.x) .<= L/2 + 1e-12)
    e = [ci[1], ci[end]]
    th = -angle(result.eta[e[argmax(abs.(result.eta[e]))]])
    q = modal.q .* exp(im * th)
    amp = abs.(q); amax = maximum(amp)
    neutral = CM.RGBf(0.72, 0.72, 0.72)
    map(eachindex(q)) do i
        t = abs(angle(q[i])) / pi
        base = t <= 0.5 ? _mixc(MAKIE_BLUE, neutral, 2t) : _mixc(neutral, MAKIE_RED, 2t - 1)
        a = 0.35 + 0.65 * clamp(amp[i] / amax, 0, 1)^0.45
        CM.RGBAf(base.r, base.g, base.b, a)
    end
end

# Slim phase key, drawn inside the modal axis in data coordinates.  Modes n>=6
# are negligible in every panel of both grids, so the upper-right of the axis is
# dead space and the key costs no layout.
function draw_phase_key!(ax, ylims_scaled)
    x0, x1 = 5.5, 9.45
    lo, hi = ylims_scaled
    y0 = lo + 0.60 * (hi - lo)
    y1 = lo + 0.71 * (hi - lo)
    neutral = CM.RGBf(0.72, 0.72, 0.72)
    N = 96
    xs = range(x0, x1, length = N + 1)
    for i in 1:N
        t = abs(-pi + 2pi * (i - 0.5) / N) / pi
        c = t <= 0.5 ? _mixc(MAKIE_BLUE, neutral, 2t) : _mixc(neutral, MAKIE_RED, 2t - 1)
        CM.poly!(ax, CM.Point2f[(xs[i], y0), (xs[i+1], y0), (xs[i+1], y1), (xs[i], y1)];
                 color = c, strokewidth = 0)
    end
    CM.lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0];
              color = MAKIE_GRAY, linewidth = 0.8)
    for (xx, lab, al) in ((x0, "-180", :left), (x1, "180", :right))
        CM.text!(ax, xx, y0; text = lab, fontsize = 24, align = (al, :top),
                 offset = (0, -2), color = MAKIE_GRAY)
    end
    CM.text!(ax, (x0+x1)/2, y1; text = L"\arg\bar{q}_n", fontsize = 26,
             align = (:center, :bottom), offset = (0, 3), color = MAKIE_GRAY)
end

function draw_modal_axis!(ax, modal, L_raft; ylims, show_ylabel, result = nothing, show_phase_key = false)
    mode_energy = abs.(modal.q) ./ L_raft
    ylims_scaled = (ylims[1] / L_raft, ylims[2] / L_raft)
    barcol = isnothing(result) ? MAKIE_BLUE : phase_bar_colors(modal, result)
    CM.barplot!(ax, modal.n, mode_energy;
        color = barcol, strokecolor = barcol, fillto = ylims_scaled[1])
    ax.xlabel = L"n"
    ax.ylabel = show_ylabel ? L"|\bar{q}_n|" : ""
    ax.xticks = modal.n
    CM.ylims!(ax, ylims_scaled...)
    ax.yticks = [0.0, ylims_scaled[2]/2, ylims_scaled[2]]
    show_phase_key && draw_phase_key!(ax, ylims_scaled)
end

function solve_snapshot_ops(ops)
    _, EI_scale, _, _ = paper_snapshot_ops()
    results = []
    modals = []
    for op in ops
        result, modal = get_cached_snapshot(op, EI_scale)
        push!(results, result)
        push!(modals, modal)
    end
    return results, modals
end

function wave_ylim(results)
    ylim = 0.0
    for res in results
        _, _, eta_m, _ = phase_and_profile(res)
        L_raft = res.metadata.args.L_raft
        eta_over_L = eta_m ./ L_raft
        ylim = max(ylim, maximum(abs.(eta_over_L)))
    end
    return ceil(ylim / 0.01) * 0.01
end

function make_snapshot_grid(fig_dir; kind::Symbol, op_indices, filename, column_titles,
                            modal_energy_ylims = (0.0, 2e-5), global_ylim = nothing,
                            sweep_ylim = nothing, sweep_xlim = nothing)
    setup_lm_mathfonts()
    makie_snapshot_theme!()
    _, _, _, all_ops = paper_snapshot_ops()
    ops = all_ops[op_indices]
    results, modals = solve_snapshot_ops(ops)
    ylim = isnothing(global_ylim) ? wave_ylim(results) : global_ylim
    sweep = load_sweep_cache_for_grid(kind)

    fig = CM.Figure(size = (1500, 995), backgroundcolor = :white)
    CM.rowsize!(fig.layout, 1, CM.Fixed(355))  # keep sweep row at original height
    # `XM_HIGHLIGHTS` is stored in the historical right-to-left order.  The
    # snapshot columns are left-to-right, so reverse styles only for this grid.
    highlight_colors = kind == :xM ? reverse(COLUMN_COLORS) : COLUMN_COLORS
    highlight_linewidths = kind == :xM ? reverse(COLUMN_LINEWIDTHS) : COLUMN_LINEWIDTHS
    draw_sweep_axis!(fig[1, 1:3], fig[1, 1:3, CM.Left()], sweep; legend_position = :rt,
        highlight_colors, highlight_linewidths, ylim_override = sweep_ylim,
        xlim_override = sweep_xlim)

    for j in 1:3
        col = COLUMN_COLORS[j]
        sw  = COLUMN_SPINEWIDTHS[j]
        am  = j == 1 ? CM.Mixed(left = CM.Protrusion(150)) :
              j == 3 ? CM.Mixed(right = CM.Protrusion(90)) :
              CM.Mixed()
              
        axw = CM.Axis(fig[2, j];
            alignmode = am,
            title = column_titles[j], titlecolor = col,
            leftspinecolor = col, rightspinecolor = col,
            topspinecolor = col, bottomspinecolor = col, spinewidth = sw)
        F_T_ratio = results[j].thrust / abs(sweep.F_T_star)
        draw_wave_axis!(axw, results[j]; ylim, show_ylabel = (j == 1), title = column_titles[j],
            F_T_ratio, F_T_raw = results[j].thrust)
            
        axm = CM.Axis(fig[3, j];
            alignmode = am,
            leftspinecolor = col, rightspinecolor = col,
            topspinecolor = col, bottomspinecolor = col, spinewidth = sw)
        L_raft = results[j].metadata.args.L_raft
        draw_modal_axis!(axm, modals[j], L_raft; ylims = modal_energy_ylims, show_ylabel = (j == 1), result = results[j], show_phase_key = (j == 3))
    end

    CM.rowgap!(fig.layout, 18)
    CM.colgap!(fig.layout, 18)

    fname = joinpath(fig_dir, filename)
    CM.save(fname * ".pdf", fig)
    CM.save(fname * ".png", fig; px_per_unit = 2)
    println("Saved $fname.{pdf,png}")
end

function main_snapshot_grids(fig_dir)
    _, _, _, all_ops = paper_snapshot_ops()
    results_kappa, _ = solve_snapshot_ops(all_ops[[1, 2, 5]])
    results_xM, _    = solve_snapshot_ops(all_ops[[4, 3, 2]])
    global_ylim = max(wave_ylim(results_kappa), wave_ylim(results_xM))

    make_snapshot_grid(fig_dir;
        kind = :kappa,
        op_indices = [1, 2, 5],
        # The refined sweep resolves a narrow resonance near kappa = 1e-4 that
        # peaks at 65, well outside the vertical scale the rest of the paper was
        # drawn against. Restricting the horizontal range excludes it instead of
        # clipping it, and the autoscale then returns to 30.5, matching the
        # earlier figures without hiding any data inside the plotted window.
        sweep_xlim = (2e-4, 1e0),
        # The refined first highlight sits in a deeper minimum, so its modal
        # response is larger: max|q| = 2.87e-5 against the old 2e-5 ceiling,
        # which clipped the mode-3 bar. Raised with ~15% headroom.
        modal_energy_ylims = (0.0, 3.3e-5),
        filename = "plot_kappa_snapshot_grid_flexibility",
        column_titles = [L"\kappa=2.00\times10^{-3}",
                         L"\kappa=6.87\times10^{-3}",
                         L"\kappa=1.76\times10^{-2}"],
        global_ylim = global_ylim)
    make_snapshot_grid(fig_dir;
        kind = :xM,
        # Match the sweep's left-to-right thin-to-thick highlighted lines.
        # This swaps the former (b,e) and (d,g) snapshot columns.
        op_indices = [4, 3, 2],
        filename = "plot_kappa_snapshot_grid_motor_position",
        column_titles = [L"x_M/L=-0.272",
                         L"x_M/L=-0.1885",
                         L"x_M/L=-0.12"],
        modal_energy_ylims = (0.0, 3e-6),
        global_ylim = global_ylim)
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    EI        = nothing
    kappa     = nothing
    xM_norm   = nothing
    do_5panel = false
    do_paper_snapshots = false
    do_snapshot_grids = false
    fig_dir   = joinpath(@__DIR__, "..", "output", "figures")

    i = 1
    while i <= length(ARGS)
        if     ARGS[i] == "--EI";     EI      = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--kappa";  kappa   = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--xM";     xM_norm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--outdir"; fig_dir = ARGS[i+1];                 i += 2
        elseif ARGS[i] == "--paper-snapshots"; do_paper_snapshots = true;   i += 1
        elseif ARGS[i] == "--5panel"; do_5panel = true;                    i += 1
        elseif ARGS[i] == "--snapshot-grids"; do_snapshot_grids = true;     i += 1
        else   error("Unknown argument: $(ARGS[i])")
        end
    end
    !isnothing(EI) && !isnothing(kappa) && error("Provide --EI or --kappa, not both.")
    mkpath(fig_dir)

    if do_5panel
        main_5panel(fig_dir)
        return
    end
    if do_paper_snapshots
        main_paper_snapshots(fig_dir)
        return
    end
    if do_snapshot_grids
        main_snapshot_grids(fig_dir)
        return
    end

    if !isnothing(kappa) && isnothing(EI)
        bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
        EI = kappa * bp.rho_raft * bp.L_raft^4 * bp.omega^2
    end

    params  = build_params(; EI, xM_norm)
    derived = Surferbot.derive_params(params)
    kappa   = real(derived.nd_groups.kappa)
    @info @sprintf("EI = %.3e  κ = %.3e", Float64(params.EI), kappa)

    result = Surferbot.flexible_solver(params)
    modal  = Surferbot.decompose_raft_freefree_modes(result; num_modes=10, verbose=false)

    make_figure(result, modal, kappa, fig_dir; xM_norm)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
