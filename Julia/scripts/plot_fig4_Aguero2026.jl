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

const FIG1_FREE_SURFACE = "#194CCC"
const FIG1_RAFT = "#000000"
const FIG1_MOTOR = CairoMakie.RGBf(0.66, 0.43, 0.05)
const FIG3_LABELSIZE = 56.6
const FIG3_TICKSIZE = 48.1
# Panel (a) is a MATLAB figure: text, spines and ticks are all MATLAB's default
# axis colour 0.15 grey (#252525), and the grid is that colour at alpha 0.15
# (#dedede on white). Matched here so the two panels read as one figure.
const FIG3_TEXT_COLOR = RGBf(0.15, 0.15, 0.15)
const FIG3_AXIS_COLOR = RGBf(0.15, 0.15, 0.15)
const FIG3_GRID_COLOR = RGBAf(0.15, 0.15, 0.15, 0.15)

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
        # Panel (b) is compared side by side with panel (a), which is reproduced from
        # Benham et al. (2024) and cannot be regenerated. The canvas aspect and the
        # plot-box placement below are therefore locked to fractions measured directly
        # off that figure, so that when both are included at the same \textwidth the
        # two data areas coincide exactly. Do not replace these with layout padding.
        CANVAS_W, CANVAS_H = 1100, 532          # aspect 2.0677 (panel (a): 788/381)
        BOX_L, BOX_R = 0.15127, 0.99716         # fractions of canvas width
        BOX_T, BOX_B = 0.03989, 0.75247         # fractions of canvas height, from top
        fig = Figure(size = (CANVAS_W, CANVAS_H), backgroundcolor = :white,
            figure_padding = 0)
        ax = Axis(fig; bbox = BBox(BOX_L * CANVAS_W, BOX_R * CANVAS_W,
                                   (1 - BOX_B) * CANVAS_H, (1 - BOX_T) * CANVAS_H),
            xlabel = L"x\;(\mathrm{cm})", ylabel = L"h\;(𝜇\mathrm{m})",
            xlabelsize = FIG3_LABELSIZE, ylabelsize = FIG3_LABELSIZE,
            xticklabelsize = FIG3_TICKSIZE, yticklabelsize = FIG3_TICKSIZE,
            xlabelcolor = FIG3_TEXT_COLOR, ylabelcolor = FIG3_TEXT_COLOR,
            xticklabelcolor = FIG3_TEXT_COLOR, yticklabelcolor = FIG3_TEXT_COLOR,
            xticksize = 4.4, xtickalign = 0,
            xticklabelpad = -1.4, xlabelpadding = 5.8,
            # panel (a) prints ASCII hyphens, not the Unicode minus Makie
            # defaults to; match it so the two tick rows read identically
            xtickformat = vs -> [string(round(Int, v)) for v in vs],
            ytickformat = vs -> [string(round(Int, v)) for v in vs],
            bottomspinecolor = FIG3_AXIS_COLOR, topspinecolor = FIG3_AXIS_COLOR,
            leftspinecolor = FIG3_AXIS_COLOR, rightspinecolor = FIG3_AXIS_COLOR,
            xtickcolor = FIG3_AXIS_COLOR, ytickcolor = FIG3_AXIS_COLOR,
            xgridcolor = FIG3_GRID_COLOR, ygridcolor = FIG3_GRID_COLOR,
            xticks = -6:2:6, yticks = -300:100:300, xgridvisible = true, ygridvisible = true)
        xlims!(ax, -7, 7)
        ylims!(ax, -300, 300)
        lines!(ax, x_cm, eta_um; color = FIG1_FREE_SURFACE, linewidth = 2.8)
        lines!(ax, x_cm[contact], eta_um[contact]; color = FIG1_RAFT, linewidth = 5.6)
        scatter!(ax, [x_cm[motor_idx]], [eta_um[motor_idx]];
            color = FIG1_MOTOR, strokecolor = FIG1_MOTOR, markersize = 30)
        save(fname, fig)
    end
    println("Saved $fname")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
