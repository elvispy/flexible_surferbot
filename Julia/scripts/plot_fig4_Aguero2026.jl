"""
plot_fig4_Aguero2026.jl

Rigid-raft validation figure (Fig 4, Aguero 2026):
  Wave profile η(x) across the full 14 cm domain.
  Blue segment: raft contact region.
  Red segment: free water on either side.

EI = Inf  (rigid limit), ν = 1e-6 m²/s (water), motor at −3 mm left of centre.
motor_inertia doubled (×2.0) vs default to match Benham 2024 amplitude A = 150 µm.

Ports MATLAB/utils/plot_one.m to Julia.
"""

using Surferbot
using Plots
using LaTeXStrings

function main()
    fig_dir = joinpath(@__DIR__, "..", "output", "figures")
    mkpath(fig_dir)

    params = Surferbot.FlexibleParams(
        sigma          = 0.0722,
        rho            = 1000.0,
        omega          = 2π * 80.0,
        nu             = 1e-6,
        g              = 9.81,
        L_raft         = 0.05,
        motor_position = -0.003,
        d              = 0.03,
        EI             = Inf,
        rho_raft       = 0.052,
        motor_inertia  = 0.13e-3 * 2.5e-3 * 2.0,
        L_domain       = 0.14,
    )

    result  = Surferbot.flexible_solver(params)
    contact = Bool.(result.metadata.args.x_contact)

    contact_idx = findall(contact)
    beam_ends   = [contact_idx[1], contact_idx[end]]
    dom_end     = beam_ends[argmax(abs.(result.eta[beam_ends]))]
    theta       = -angle(result.eta[dom_end])

    x_cm   = result.x .* 1e2
    eta_um = real.(result.eta .* exp(im * theta)) .* 1e6

    p = plot(x_cm, eta_um;
        color          = :red,
        linewidth      = 1.2,
        label          = false,
        xlabel         = L"x\;(\mathrm{cm})",
        ylabel         = L"h\;(\mu\mathrm{m})",
        xlims          = (-7, 7),
        ylims          = (-300, 300),
        yticks         = -300:100:300,
        grid           = true,
        framestyle     = :box,
        fontfamily     = "Computer Modern",
        guidefontsize  = 20,
        tickfontsize   = 15,
        size           = (1100, 420),
        dpi            = 220,
        left_margin    = 12Plots.mm,
        bottom_margin  = 10Plots.mm,
        top_margin     =  4Plots.mm,
        right_margin   =  4Plots.mm,
    )
    plot!(p, x_cm[contact], eta_um[contact];
        color = :blue, linewidth = 2.0, label = false)

    fname = joinpath(fig_dir, "fig4_Aguero2026.pdf")
    savefig(p, fname)
    println("Saved $fname")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
