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
     (b) κ = 5.43e-3,  xM = Surferbot
     (c) κ = 5.43e-3,  xM/L = 0.183  (α ≈ 0)
     (d) κ = 5.43e-3,  xM/L = 0.272  (|α| ≈ 1)
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
    eta_um      = real.(result.eta .* exp(im * theta)) .* 1e6
    motor_x_cm  = Float64(result.metadata.args.motor_position) * 1e2
    return x_cm, contact, eta_um, motor_x_cm
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
const LM_FONT = "Latin Modern Roman"
const THRUST_CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "thrust_sweeps.jld2")
const ALPHA_CACHE_PATH = joinpath(@__DIR__, "..", "output", "jld2", "alpha_sweep_kappa_farfield.jld2")
const GRID_ALPHA_CSV = joinpath(@__DIR__, "..", "output", "csv", "sweeper_coupled_full_grid.csv")
const KAPPA_HIGHLIGHTS = [1.71103172e-3, 5.43e-3, 2.22e-2]
const XM_HIGHLIGHTS = [0.12, 0.183, 0.272]

const STYLE = (
    framestyle    = :box,
    fontfamily    = "Computer Modern",
    guidefontsize = 21,
    tickfontsize  = 18,
)

function paper_snapshot_ops()
    bp       = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    EI_scale = Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2
    xM_sb    = abs(Float64(bp.motor_position)) / Float64(bp.L_raft)

    ops = [
        (kappa=1.71103172e-3, xM=xM_sb,  file_xM=nothing, label="(a)"),
        (kappa=5.43e-3, xM=xM_sb,  file_xM=nothing, label="(b)"),
        (kappa=5.43e-3, xM=0.183,  file_xM=0.183,   label="(c)"),
        (kappa=5.43e-3, xM=0.272,  file_xM=0.272,   label="(d)"),
        (kappa=2.22e-2, xM=xM_sb,  file_xM=nothing, label="(e)"),
    ]
    return bp, EI_scale, xM_sb, ops
end

function make_wave_panel(result; ylim=1500, small=false,
                          show_xlabel=true, show_ylabel=true)
    x_cm, contact, eta_um, motor_x_cm = phase_and_profile(result)
    motor_idx = argmin(abs.(x_cm .- motor_x_cm))

    fs_guide = small ? 15 : 21
    fs_tick  = small ? 12 : 18
    b_margin = small ?  6Plots.mm : 11Plots.mm
    l_margin = small ?  8Plots.mm : 14Plots.mm

    p = plot(x_cm, eta_um;
        color     = FIG1_FREE_SURFACE,
        linewidth = 1.5,
        label     = false,
        xlabel    = show_xlabel ? L"x\;(\mathrm{cm})" : "",
        ylabel    = show_ylabel ? L"h\;(\mu\mathrm{m})" : "",
        xlims     = (-5, 5),
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
    plot!(p, x_cm[contact], eta_um[contact];
        color = FIG1_RAFT, linewidth = 2.5, label = false)
    scatter!(p, [x_cm[motor_idx]], [eta_um[motor_idx]];
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
    all_energy = reduce(vcat, (abs2.(modal.q) for modal in modals))
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

    mode_energy = abs2.(modal.q)
    ylims = isnothing(modal_energy_ylims) ? common_modal_energy_limits([modal]) : modal_energy_ylims
    mode_energy = max.(mode_energy, ylims[1])
    p2  = bar(modal.n, mode_energy;
        xticks        = modal.n,
        xlabel        = L"n",
        ylabel        = L"|q_n|^2",
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
        @info @sprintf("Solving κ=%.2e  xM/L=%.3f", op.kappa, op.xM)
        EI = op.kappa * EI_scale
        p = build_params(; EI, xM_norm=op.xM)
        result = Surferbot.flexible_solver(p)
        modal = Surferbot.decompose_raft_freefree_modes(result; num_modes=10, verbose=false)
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
        @info @sprintf("Solving κ=%.2e  xM/L=%.3f", op.kappa, op.xM)
        EI = op.kappa * EI_scale
        p  = build_params(; EI, xM_norm=op.xM)
        push!(results, Surferbot.flexible_solver(p))
    end

    # Common y-axis limit across all panels
    ylim = 0
    for res in results
        _, _, eta_um, _ = phase_and_profile(res)
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
            xlabelsize = 26,
            ylabelsize = 26,
            xticklabelsize = 22,
            yticklabelsize = 22,
            titlesize = 26,
            xgridvisible = false,
            ygridvisible = false,
            topspinevisible = true,
            rightspinevisible = true,
            bottomspinevisible = true,
            leftspinevisible = true,
        ),
        Legend = (;
            labelsize = 22,
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
                highlights = KAPPA_HIGHLIGHTS)
    elseif kind == :xM
        alpha_xM = load_motor_alpha_from_csv(bp; target_kappa = 5.43e-3)
        return (; x = d["xM_x"], y = d["xM_T"] .* depth ./ scale,
                ylh = d["xM_Sxx"] .* depth ./ scale,
                alpha_x = alpha_xM.x, alpha = alpha_xM.alpha,
                xlabel = L"x_M/L",
                xscale = identity,
                xticks = 0.0:0.1:0.5,
                highlights = XM_HIGHLIGHTS)
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

function draw_sweep_axis!(figpos, sweep; legend_position = :rb,
                           sweep_labelsize = 29, sweep_ticksize = 26,
                           legend_labelsize = 26, legend_patchsize = (55, 23),
                           highlight_colors = fill(MAKIE_GRAY, length(sweep.highlights)),
                           highlight_linewidths = fill(1.5, length(sweep.highlights)))
    # sweep_labelsize/sweep_ticksize override the column-panel theme defaults
    # for this full-width row (scale 0.416): 22→9.2pt, 19→7.9pt
    ylim = 1.08 * maximum(abs.(vcat(sweep.y, sweep.ylh, 0.0)))
    ax = CM.Axis(figpos;
        xlabel = sweep.xlabel,
        ylabel = L"F_T/F_T^\ast",
        xlabelsize = sweep_labelsize,
        ylabelsize = sweep_labelsize,
        xticklabelsize = sweep_ticksize,
        yticklabelsize = sweep_ticksize,
        xscale = sweep.xscale,
        xticks = sweep.xticks,
        limits = ((minimum(sweep.x), maximum(sweep.x)), (-ylim, ylim)))
    order = sortperm(sweep.x)
    l1 = CM.lines!(ax, sweep.x[order], sweep.y[order]; color = MAKIE_BLUE, linewidth = 2.6)
    l2 = CM.lines!(ax, sweep.x[order], sweep.ylh[order]; color = MAKIE_RED, linewidth = 2.6, linestyle = :dash)
    CM.hlines!(ax, [0.0]; color = (:black, 0.55), linewidth = 0.9)
    for (xh, col, lw) in zip(sweep.highlights, highlight_colors, highlight_linewidths)
        CM.vlines!(ax, [xh]; color = (col, 0.85), linewidth = lw, linestyle = :dash)
    end

    axr = CM.Axis(figpos;
        xscale = sweep.xscale,
        xticks = sweep.xticks,
        yaxisposition = :right,
        ylabel = L"\alpha",
        ylabelsize = sweep_labelsize,
        yticklabelsize = sweep_ticksize,
        ylabelcolor = MAKIE_ALPHA,
        yticklabelcolor = MAKIE_ALPHA,
        rightspinecolor = MAKIE_ALPHA,
        ytickcolor = MAKIE_ALPHA,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = :transparent,
        limits = ((minimum(sweep.x), maximum(sweep.x)), (-1.1, 1.1)),
        ytickformat = vals -> [latexstring(@sprintf("%.1f", v)) for v in vals])
    CM.hidespines!(axr, :l, :b, :t)
    CM.hidexdecorations!(axr; grid = false)
    aorder = sortperm(sweep.alpha_x)
    l3 = CM.lines!(axr, sweep.alpha_x[aorder], sweep.alpha[aorder]; color = MAKIE_ALPHA, linewidth = 2.6)
    CM.axislegend(ax, [l1, l2, l3], [L"Numerics", L"Longuet-Higgins", L"\alpha"];
        position = legend_position, backgroundcolor = (:white, 0.86), framecolor = (:black, 0.45),
        labelsize = legend_labelsize, patchsize = legend_patchsize)
    return ax
end

function draw_wave_axis!(ax, result; ylim, show_ylabel, title)
    x_cm, contact, eta_um, motor_x_cm = phase_and_profile(result)
    motor_idx = argmin(abs.(x_cm .- motor_x_cm))
    CM.lines!(ax, x_cm, eta_um; color = MAKIE_BLUE, linewidth = 1.8)
    CM.lines!(ax, x_cm[contact], eta_um[contact]; color = :black, linewidth = 3.0)
    CM.scatter!(ax, [x_cm[motor_idx]], [eta_um[motor_idx]];
        color = MAKIE_LOAD, strokecolor = MAKIE_LOAD, markersize = 10)
    ax.title = title
    ax.xlabel = L"x\;(\mathrm{cm})"
    ax.ylabel = show_ylabel ? L"h\;(\mu\mathrm{m})" : ""
    CM.xlims!(ax, -5, 5)
    CM.ylims!(ax, -ylim, ylim)
end

function draw_modal_axis!(ax, modal; ylims, show_ylabel)
    mode_energy = abs2.(modal.q)
    mode_energy = max.(mode_energy, ylims[1])
    CM.barplot!(ax, modal.n, mode_energy;
        color = MAKIE_BLUE, strokecolor = MAKIE_BLUE, fillto = ylims[1])
    ax.xlabel = L"n"
    ax.ylabel = show_ylabel ? L"|q_n|^2" : ""
    ax.xticks = modal.n
    ax.yticks = CM.LogTicks(CM.WilkinsonTicks(4))
    CM.ylims!(ax, ylims...)
end

function solve_snapshot_ops(ops)
    _, EI_scale, _, _ = paper_snapshot_ops()
    results = []
    modals = []
    for op in ops
        @info @sprintf("Solving grid snapshot κ=%.2e  xM/L=%.3f", op.kappa, op.xM)
        EI = op.kappa * EI_scale
        p = build_params(; EI, xM_norm=op.xM)
        result = Surferbot.flexible_solver(p)
        modal = Surferbot.decompose_raft_freefree_modes(result; num_modes=10, verbose=false)
        push!(results, result)
        push!(modals, modal)
    end
    return results, modals
end

function wave_ylim(results)
    ylim = 0.0
    for res in results
        _, _, eta_um, _ = phase_and_profile(res)
        ylim = max(ylim, maximum(abs.(eta_um)))
    end
    return ceil(ylim / 500) * 500
end

function make_snapshot_grid(fig_dir; kind::Symbol, op_indices, filename, column_titles)
    makie_snapshot_theme!()
    _, _, _, all_ops = paper_snapshot_ops()
    ops = all_ops[op_indices]
    results, modals = solve_snapshot_ops(ops)
    ylim = wave_ylim(results)
    modal_energy_ylims = (1e-14, 1e-8)
    sweep = load_sweep_cache_for_grid(kind)

    fig = CM.Figure(size = (1500, 995), backgroundcolor = :white)
    CM.rowsize!(fig.layout, 1, CM.Fixed(355))  # keep sweep row at original height
    draw_sweep_axis!(fig[1, 1:3], sweep; legend_position = :rb,
        highlight_colors = COLUMN_COLORS, highlight_linewidths = COLUMN_LINEWIDTHS)

    for j in 1:3
        col = COLUMN_COLORS[j]
        sw  = COLUMN_SPINEWIDTHS[j]
        axw = CM.Axis(fig[2, j];
            title = column_titles[j], titlecolor = col,
            leftspinecolor = col, rightspinecolor = col,
            topspinecolor = col, bottomspinecolor = col, spinewidth = sw)
        draw_wave_axis!(axw, results[j]; ylim, show_ylabel = (j == 1), title = column_titles[j])
        axm = CM.Axis(fig[3, j]; yscale = log10,
            leftspinecolor = col, rightspinecolor = col,
            topspinecolor = col, bottomspinecolor = col, spinewidth = sw)
        draw_modal_axis!(axm, modals[j]; ylims = modal_energy_ylims, show_ylabel = (j == 1))
    end

    CM.rowgap!(fig.layout, 18)
    CM.colgap!(fig.layout, 18)

    fname = joinpath(fig_dir, filename)
    CM.save(fname * ".pdf", fig)
    CM.save(fname * ".png", fig; px_per_unit = 2)
    println("Saved $fname.{pdf,png}")
end

function main_snapshot_grids(fig_dir)
    make_snapshot_grid(fig_dir;
        kind = :kappa,
        op_indices = [1, 2, 5],
        filename = "plot_kappa_snapshot_grid_flexibility",
        column_titles = [L"\kappa=1.71\times10^{-3}",
                         L"\kappa=5.43\times10^{-3}",
                         L"\kappa=2.22\times10^{-2}"])
    make_snapshot_grid(fig_dir;
        kind = :xM,
        op_indices = [2, 3, 4],
        filename = "plot_kappa_snapshot_grid_motor_position",
        column_titles = [L"x_M/L=0.12",
                         L"x_M/L=0.183",
                         L"x_M/L=0.272"])
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
