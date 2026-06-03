# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_kappa_snapshot.jl

Two modes:

1. Single snapshot (default / --EI / --kappa):
   Two-panel figure for one operating point:
     (a) Free-surface profile η(x) with a scatter marker at the motor position
     (b) Spectral energy fraction in each free-free beam mode W_n

2. Five-panel snapshot figure (--5panel):
   One-row figure with panels (a)–(e) covering three κ values and five
   distinct (κ, xM/L) operating points for comparison:
     (a) κ = 1.82e-3,  xM = Surferbot
     (b) κ = 5.43e-3,  xM = Surferbot
     (c) κ = 5.43e-3,  xM/L = 0.183  (α ≈ 0)
     (d) κ = 5.43e-3,  xM/L = 0.272  (|α| ≈ 1)
     (e) κ = 1.94e-2,  xM = Surferbot

Usage:
  julia --project=. scripts/plot_kappa_snapshot.jl [--EI VALUE] [--kappa VALUE] [--outdir DIR]
  julia --project=. scripts/plot_kappa_snapshot.jl --5panel [--outdir DIR]
"""

using Surferbot
using Plots
using LaTeXStrings
using Printf

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

const STYLE = (
    framestyle    = :box,
    fontfamily    = "Computer Modern",
    guidefontsize = 21,
    tickfontsize  = 18,
)

function make_wave_panel(result; ylim=1500, small=false,
                          show_xlabel=true, show_ylabel=true)
    x_cm, contact, eta_um, motor_x_cm = phase_and_profile(result)
    motor_idx = argmin(abs.(x_cm .- motor_x_cm))

    fs_guide = small ? 15 : 21
    fs_tick  = small ? 12 : 18
    b_margin = small ?  6Plots.mm : 10Plots.mm
    l_margin = small ?  8Plots.mm : 14Plots.mm

    p = plot(x_cm, eta_um;
        color     = :red,
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
        color = :blue, linewidth = 2.5, label = false)
    scatter!(p, [x_cm[motor_idx]], [eta_um[motor_idx]];
        markershape       = :circle,
        markersize        = 9,
        color             = :black,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
        label             = false)
    return p
end

# ─── Two-panel snapshot figure (original single-snapshot mode) ────────────────

function make_figure(result, modal, kappa_val, fig_dir; xM_norm=nothing)
    p1 = make_wave_panel(result)

    pct = modal.energy_frac .* 100
    p2  = bar(modal.n, pct;
        xticks        = modal.n,
        xlabel        = L"n",
        ylabel        = "mode energy (%)",
        label         = false,
        fillcolor     = :steelblue,
        linecolor     = :steelblue,
        linewidth     = 0.5,
        grid          = :y,
        ylims         = (0, max(maximum(pct) * 1.2, 5.0)),
        bottom_margin = 10Plots.mm,
        left_margin   = 10Plots.mm,
        top_margin    =  4Plots.mm,
        right_margin  =  5Plots.mm,
        STYLE...,
    )

    fig = plot(p1, p2;
        layout = grid(1, 2, widths = [0.65, 0.35]),
        size   = (1152, 380),
        dpi    = 220,
    )

    fname = if isnothing(xM_norm)
        joinpath(fig_dir, @sprintf("kappa_snapshot_%.2e.pdf", kappa_val))
    else
        joinpath(fig_dir, @sprintf("kappa_snapshot_%.2e_xM%.3f.pdf", kappa_val, xM_norm))
    end
    savefig(fig, fname)
    println("Saved $fname")
end

# ─── Five-panel figure ────────────────────────────────────────────────────────

function main_5panel(fig_dir)
    bp       = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    EI_scale = Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2
    xM_sb    = abs(Float64(bp.motor_position)) / Float64(bp.L_raft)

    # Five operating points
    ops = [
        (kappa=1.82e-3, xM=xM_sb,  label="(a)"),
        (kappa=5.43e-3, xM=xM_sb,  label="(b)"),
        (kappa=5.43e-3, xM=0.183,  label="(c)"),   # α ≈ 0
        (kappa=5.43e-3, xM=0.272,  label="(d)"),   # |α| ≈ 1
        (kappa=1.94e-2, xM=xM_sb,  label="(e)"),
    ]

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

    fname = joinpath(fig_dir, "kappa_snapshot_5panel.pdf")
    savefig(fig, fname)
    println("Saved $fname")
    savefig(fig, replace(fname, ".pdf" => ".png"))
    println("Saved $(replace(fname, ".pdf" => ".png"))")
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    EI        = nothing
    kappa     = nothing
    xM_norm   = nothing
    do_5panel = false
    fig_dir   = joinpath(@__DIR__, "..", "output", "figures")

    i = 1
    while i <= length(ARGS)
        if     ARGS[i] == "--EI";     EI      = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--kappa";  kappa   = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--xM";     xM_norm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--outdir"; fig_dir = ARGS[i+1];                 i += 2
        elseif ARGS[i] == "--5panel"; do_5panel = true;                    i += 1
        else   error("Unknown argument: $(ARGS[i])")
        end
    end
    !isnothing(EI) && !isnothing(kappa) && error("Provide --EI or --kappa, not both.")
    mkpath(fig_dir)

    if do_5panel
        main_5panel(fig_dir)
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
