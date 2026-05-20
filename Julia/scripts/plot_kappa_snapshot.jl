"""
plot_kappa_snapshot.jl

Two-panel snapshot figure for one operating point:
  (a) Free-surface profile η(x) — wave field across the domain
  (b) Spectral energy fraction in each free-free beam mode W_n

Default parameters match the SurferBot (Rhee et al. 2022 / Benham et al. 2024).
Pass --EI to override the bending stiffness and select a different κ.

Usage:
  julia --project=. scripts/plot_kappa_snapshot.jl [--EI VALUE] [--outdir DIR]
"""

using Surferbot
using Plots
using LaTeXStrings
using Printf

# ─── Parameters ──────────────────────────────────────────────────────────────

function build_params(; EI=nothing)
    bp       = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    overrides = isnothing(EI) ?
        (L_domain = 0.10,) :
        (L_domain = 0.10, EI = EI)
    return Surferbot.Sweep.apply_parameter_overrides(bp, overrides)
end

# ─── Figure ──────────────────────────────────────────────────────────────────

const STYLE = (
    framestyle    = :box,
    fontfamily    = "Computer Modern",
    guidefontsize = 21,
    tickfontsize  = 18,
)

function make_figure(result, modal, kappa_val, fig_dir)
    x_cm    = result.x .* 1e2
    contact = Bool.(result.metadata.args.x_contact)
    nm      = length(modal.q_w)

    # Choose t so that the larger beam endpoint is at its positive maximum:
    #   Re(η(beam_end) · exp(iθ)) = |η(beam_end)|  with θ = -∠η(beam_end).
    # This pins the phase to the raft edge that dominates (relevant for |α|≈1).
    contact_idx = findall(contact)
    beam_ends   = [contact_idx[1], contact_idx[end]]
    dom_end     = beam_ends[argmax(abs.(result.eta[beam_ends]))]
    theta       = -angle(result.eta[dom_end])
    eta_um      = real.(result.eta .* exp(im * theta)) .* 1e6

    # ── Panel (a): wave profile ───────────────────────────────────────────────
    p1 = plot(x_cm, eta_um;
        color     = :red,
        linewidth = 1.5,
        label     = false,
        xlabel    = L"x\;(\mathrm{cm})",
        ylabel    = L"h\;(\mu\mathrm{m})",
        xlims     = (-5, 5),
        ylims     = (-1500, 1500),
        yticks    = -1500:500:1500,
        grid      = true,
        bottom_margin = 10Plots.mm,
        left_margin   = 14Plots.mm,
        top_margin    =  4Plots.mm,
        right_margin  =  2Plots.mm,
        STYLE...,
    )
    plot!(p1, x_cm[contact], eta_um[contact];
        color = :blue, linewidth = 2.5, label = false)

    # ── Panel (b): modal energy distribution ─────────────────────────────────
    pct = modal.energy_frac .* 100

    p2 = bar(modal.n, pct;
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

    fname = joinpath(fig_dir, @sprintf("kappa_snapshot_%.2e.pdf", kappa_val))
    savefig(fig, fname)
    println("Saved $fname")
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    EI      = nothing
    kappa   = nothing
    fig_dir = joinpath(@__DIR__, "..", "output", "figures")

    i = 1
    while i <= length(ARGS)
        if     ARGS[i] == "--EI";     EI      = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--kappa";  kappa   = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--outdir"; fig_dir = ARGS[i+1];                 i += 2
        else   error("Unknown argument: $(ARGS[i])")
        end
    end
    !isnothing(EI) && !isnothing(kappa) && error("Provide --EI or --kappa, not both.")
    mkpath(fig_dir)

    if !isnothing(kappa) && isnothing(EI)
        bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
        EI = kappa * bp.rho_raft * bp.L_raft^4 * bp.omega^2
    end

    params  = build_params(; EI)
    derived = Surferbot.derive_params(params)
    kappa   = real(derived.nd_groups.kappa)
    @info @sprintf("EI = %.3e  κ = %.3e", Float64(params.EI), kappa)

    result = Surferbot.flexible_solver(params)
    modal  = Surferbot.decompose_raft_freefree_modes(result; num_modes=10, verbose=false)

    make_figure(result, modal, kappa, fig_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
