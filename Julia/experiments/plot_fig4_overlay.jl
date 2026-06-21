"""
plot_fig4_overlay.jl

Overlay fig4_Aguero2026 (Julia) and fig4_wavescodeV2 (MATLAB old_code/waves_code)
on the same axes, using the same phase conventions as each original figure:
  - Julia:        theta = -angle(eta[dominant beam end])  (same as plot_fig4_Aguero2026.jl)
  - waves_code:   sign flip  eta_M -> -eta_M              (same as plot_fig4_wavescodeV2.m)
"""

using Surferbot, Plots, LaTeXStrings, DelimitedFiles

function main()
    fig_dir = joinpath(@__DIR__, "..", "output", "figures")
    mkpath(fig_dir)

    # --- Julia ---
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

    x_J_cm   = result.x .* 1e2
    eta_J_um = real.(result.eta .* exp(im * theta)) .* 1e6

    # --- MATLAB waves_code (nondim CSV) ---
    L = 0.05
    ml = readdlm("/tmp/fig4_wavescodeV2_eta.csv", ',')
    x_M_cm   = ml[:, 1] .* L .* 100          # nondim -> cm
    eta_M_um = -real.(ml[:, 2]) .* L .* 1e6  # sign flip + nondim -> µm
    x_M_contact = abs.(ml[:, 1]) .<= 0.5

    # --- Plot ---
    p = plot(x_J_cm, eta_J_um;
        color         = :blue,
        linewidth     = 1.5,
        label         = "Julia",
        xlabel        = L"x\;(\mathrm{cm})",
        ylabel        = L"h\;(\mu\mathrm{m})",
        xlims         = (-7, 7),
        ylims         = (-300, 300),
        yticks        = -300:100:300,
        grid          = true,
        framestyle    = :box,
        fontfamily    = "Computer Modern",
        guidefontsize = 20,
        tickfontsize  = 15,
        legendfontsize= 13,
        size          = (1100, 420),
        dpi           = 220,
        left_margin   = 12Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin    =  4Plots.mm,
        right_margin   = 10Plots.mm,
    )
    plot!(p, x_M_cm, eta_M_um;
        color     = :red,
        linewidth = 1.5,
        linestyle = :dash,
        label     = "waves\\_code (MATLAB)",
    )

    fname = joinpath(fig_dir, "fig4_overlay.pdf")
    savefig(p, fname)
    println("Saved $fname")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
