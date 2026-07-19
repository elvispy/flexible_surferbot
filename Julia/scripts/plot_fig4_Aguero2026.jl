# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_fig4_Aguero2026.jl

Rigid-raft validation figure (Fig 4, Aguero 2026):
  Wave profile η(x) across the full 14 cm domain.
  Blue: free water surface on either side.
  Black: raft contact region.

EI = Inf  (rigid limit), ν = 1e-6 m²/s (water), motor at −6 mm left of centre
(matches the real SurferBot: raft 5×3 cm, motor 0.6 cm left of center).
motor_inertia doubled (×2.0) vs default to match Benham 2024 amplitude A = 150 µm
(re-verify this scale factor still hits A≈150µm at the corrected -6mm position).

Ports MATLAB/utils/plot_one.m to Julia.
"""

using Surferbot
using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "paper_plot_theme.jl"))
using .PaperPlotTheme

const FIG1_FREE_SURFACE = "#1A4DCC"
const FIG1_RAFT = "#000000"
const FIG1_MOTOR = CairoMakie.RGBf(0.66, 0.43, 0.05)

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
        motor_position = -0.006,
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

    motor_x_cm = params.motor_position * 1e2
    motor_idx  = argmin(abs.(x_cm .- motor_x_cm))

    fname = joinpath(fig_dir, "plot_fig4_Aguero2026_1.pdf")
    PaperPlotTheme.with_theme() do
        fig = Figure(size = (1100, 530), backgroundcolor = :white,
            figure_padding = (28, 8, 8, 18))
        ax = Axis(fig[1, 1]; xlabel = L"x\;(\mathrm{cm})", ylabel = L"h\;(\mu\mathrm{m})",
            xlabelsize = 39, ylabelsize = 39, xticklabelsize = 29, yticklabelsize = 29,
            xticks = -6:2:6, yticks = -300:100:300, xgridvisible = true, ygridvisible = true)
        xlims!(ax, -7, 7)
        ylims!(ax, -300, 300)
        lines!(ax, x_cm, eta_um; color = FIG1_FREE_SURFACE, linewidth = 2.8)
        lines!(ax, x_cm[contact], eta_um[contact]; color = FIG1_RAFT, linewidth = 5.6)
        scatter!(ax, [x_cm[motor_idx]], [eta_um[motor_idx]];
            color = FIG1_MOTOR, strokecolor = FIG1_MOTOR, markersize = 18)
        save(fname, fig)
    end
    println("Saved $fname")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
