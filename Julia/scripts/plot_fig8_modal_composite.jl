# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_fig8_modal_composite.jl

Four-panel replacement for figure 8 (the domain-end asymmetry factor map).
Panel (a) is the existing full-finite-difference map with its root/resonance
overlays (plot_dimensionless_diagnostics_LH.jl's build_LH_plot / draw_LH_axis!).
Panels (b)-(d) are the N=8, N=4, and N=2 modal-truncation (ROM) reconstructions
of the same quantity (far-field asymmetry, coupled, Lambda!=0), stacked to its
right and sharing one colorbar. N=2 keeps only the two rigid-body modes
(heave, pitch); its featureless map is the direct illustration that the model
reduces to rigid-body mechanics as kappa -> infinity (see the theoretical
result: at kappa=infinity, the response depends only on the n=0,1 projections
of the forcing, so a rigid-only truncation is exact in that limit and
necessarily flat/wrong away from it).

Layout notes:
  - Panel (a) and each of (b)/(c)/(d) share the same measured box aspect
    ratio, R = 1.236 (panel (a)'s own aspect from an earlier render), enforced
    via Fixed (not Relative) GridLayout sizes solved as one self-consistent
    system -- see the comments below for why Relative sizing and AxisAspect()
    both failed here (stray blank columns, or panel/column mismatches).
  - Font sizes and curve linewidths are scaled from figures 7/8/10's
    established fontsize/canvas_width and linewidth/canvas_width ratios
    (~0.0193, ~0.0171, ~0.00488), since this figure's canvas is much wider
    than those, then bumped an additional +10% except for panel (a)'s legend
    and the "ROM N=X" subtitles specifically (kept at the unscaled reference
    size).

Output: output/figures/plot_fig8_modal_composite.pdf
"""

include(joinpath(@__DIR__, "plot_fig10_modal_maps_3x3.jl"))

function main()
    fig_dir = joinpath(OUTPUT_DIR, "figures")
    mkpath(fig_dir)

    art_cpl = Surferbot.Sweep.load_sweep(joinpath(
        OUTPUT_DIR, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))

    data_a = compute_LH_plot_data(art_cpl,
        joinpath(OUTPUT_DIR, "csv", "sweeper_coupled_full_grid.csv"),
        OUTPUT_DIR; xlim_min = -4.0)

    farfield_spec = ROW_SPECS[1]
    ref_bc = csv_alpha_map(joinpath(OUTPUT_DIR, "csv", farfield_spec.csv_file),
        art_cpl, farfield_spec; xlim_min = -4.0)
    map_b = rom_alpha_map(art_cpl, ref_bc, farfield_spec; num_modes = 8)
    map_c = rom_alpha_map(art_cpl, ref_bc, farfield_spec; num_modes = 4)
    map_d = rom_alpha_map(art_cpl, ref_bc, farfield_spec; num_modes = 2)

    PaperPlotTheme.with_theme() do
        # Fixed (not Relative) sizes throughout, solved so that:
        #  (1) panel (b)/(c)/(d) width = 363 units, the width already proven
        #      wide enough for the "ROM N = 8"-style header text without
        #      overlap;
        #  (2) all four panels share panel (a)'s measured aspect ratio,
        #      R = 1345/1088 = 1.236 (measured from a prior render, not
        #      assumed -- AxisAspect() enforced this cleanly per-axis but
        #      fought colsize! fractions that didn't also account for it,
        #      producing large blank margins inside the (b)/(c)/(d) column);
        #  (3) column 1 (panel a) and column 2 (b/c/d stack) total heights
        #      are equal by construction, both Fixed to values derived from
        #      the same underlying system rather than independent guesses
        #      (which repeatedly left stray blank space or overlapping text).
        R = 1.236
        PANEL_BCD_W = 363
        PANEL_BCD_H = PANEL_BCD_W / R
        # HEADER_H, COLORBAR_W, LABELSIZE, TICKSIZE solved as a self-consistent
        # fixed point: fontsize/canvas_width must equal the ~0.0193 (label) /
        # ~0.0171 (tick) ratio measured from figures 7/8/10, while HEADER_H and
        # COLORBAR_W scale proportionally with fontsize (they were tuned at
        # LABELSIZE=24). Letting PANEL_BCD_W also scale in that same loop
        # diverges (positive feedback: canvas width grows faster than the
        # loop's own target shrinks it back) -- it stays fixed here, and only
        # gl_a's geometry is solved around it.
        LABELSIZE = 34.66
        TICKSIZE = 30.71
        # +10% everywhere except panel (a)'s legend and the "ROM N=X"
        # subtitles, per explicit request -- those two keep the unscaled
        # values.
        TAGSIZE = LABELSIZE * 1.1
        TICKSIZE_BIG = TICKSIZE * 1.1
        LEGEND_SIZE = TICKSIZE          # unscaled
        ROM_TITLE_SIZE = LABELSIZE      # unscaled
        HEADER_H = 46.22 * 1.1  # scaled to match the bigger tag/title text
        HEADER_GAP = 4
        INTERNAL_GAP = 10
        # gl_bcd's TRUE total height includes each panel's own header+gap,
        # not just the 3 axis boxes (an earlier omission let panel (b)'s
        # header and box top overflow past the canvas edge).
        PER_BLOCK_H = HEADER_H + HEADER_GAP + PANEL_BCD_H
        COL2_TOTAL_H = 3 * PER_BLOCK_H + 2 * INTERNAL_GAP
        AX_A_BOX_H = COL2_TOTAL_H - HEADER_H - HEADER_GAP
        AX_A_BOX_W = AX_A_BOX_H * R
        COLORBAR_W = 101.1
        COLGAP_OUTER = 14
        COLGAP_CBAR = 10
        LEFT_PAD = 95   # room for panel (a)'s y-axis label + tick labels
        TOP_PAD = 70    # room for the colorbar's top ("1.0") tick label
        BOTTOM_PAD = 85

        CANVAS_W = round(Int, LEFT_PAD + AX_A_BOX_W + COLGAP_OUTER + PANEL_BCD_W + COLGAP_CBAR + COLORBAR_W + 160)
        CANVAS_H = round(Int, TOP_PAD + COL2_TOTAL_H + BOTTOM_PAD)

        fig = CairoMakie.Figure(size = (CANVAS_W, CANVAS_H), backgroundcolor = :white,
            figure_padding = (LEFT_PAD, 4, BOTTOM_PAD, TOP_PAD))

        # Each panel gets a thin header row with two separate Labels overlaid
        # in the SAME single-column cell (not split into a narrow tag column
        # + a wide subtitle column -- splitting left the "centered" subtitle
        # centered only within the remaining space, offset right of the
        # panel's true center by half the excluded tag column's width): the
        # panel letter left-aligned, and a descriptive subtitle centered,
        # both independently aligned relative to the full panel width below.
        gl_a = fig[1, 1] = CairoMakie.GridLayout()
        CairoMakie.Label(gl_a[1, 1], "(a)"; halign = :left, tellwidth = false, fontsize = TAGSIZE)
        CairoMakie.Label(gl_a[1, 1], "Full model"; halign = :center, tellwidth = false, fontsize = TAGSIZE)
        ax_a = CairoMakie.Axis(gl_a[2, 1]; xlabel = L"x_M / L", ylabel = L"\kappa",
            xlabelsize = TAGSIZE, ylabelsize = TAGSIZE,
            xticklabelsize = TICKSIZE_BIG, yticklabelsize = TICKSIZE_BIG,
            yticks = kappa_exp_xticks(data_a.XLIMS),
            xgridvisible = false, ygridvisible = false)
        hm_a = draw_LH_axis!(ax_a, data_a; legend_labelsize = LEGEND_SIZE,
            linewidth_scale = 9.7188 / 4.0)
        CairoMakie.rowgap!(gl_a, HEADER_GAP)
        CairoMakie.rowsize!(gl_a, 1, CairoMakie.Fixed(HEADER_H))
        CairoMakie.rowsize!(gl_a, 2, CairoMakie.Fixed(AX_A_BOX_H))
        CairoMakie.colsize!(fig.layout, 1, CairoMakie.Fixed(AX_A_BOX_W))

        # Panels (b)/(c)/(d) share one colorbar with panel (a) (same :balance
        # colormap and (-1,1) colorrange throughout), so panel (a) does not
        # carry its own -- that space goes to enlarging the panels instead.
        # Nested in their own GridLayout (gl_bcd), so their headers'
        # protrusion is resolved entirely within it and can't create a height
        # mismatch against column 1.
        gl_bcd = fig[1, 2] = CairoMakie.GridLayout()

        function panel!(row, map_data, tag, ntext)
            gl = gl_bcd[row, 1] = CairoMakie.GridLayout()
            CairoMakie.Label(gl[1, 1], tag; halign = :left, tellwidth = false, fontsize = TAGSIZE)
            CairoMakie.Label(gl[1, 1], ntext; halign = :center, tellwidth = false, fontsize = ROM_TITLE_SIZE)
            ax = CairoMakie.Axis(gl[2, 1])
            hm = draw_panel!(ax, map_data; show_xlabel = false, show_yticks = false, show_xticks = false)
            CairoMakie.rowgap!(gl, HEADER_GAP)
            CairoMakie.rowsize!(gl, 1, CairoMakie.Fixed(HEADER_H))
            CairoMakie.rowsize!(gl, 2, CairoMakie.Fixed(PANEL_BCD_H))
            return ax, hm
        end

        ax_b, _    = panel!(1, map_b, "(b)", "ROM N = 8")
        ax_c, _    = panel!(2, map_c, "(c)", "ROM N = 4")
        ax_d, hm_d = panel!(3, map_d, "(d)", "ROM N = 2")

        CairoMakie.linkxaxes!(ax_b, ax_c, ax_d)
        CairoMakie.linkyaxes!(ax_b, ax_c, ax_d)
        CairoMakie.Colorbar(gl_bcd[1:3, 2], hm_d; label = L"\alpha",
            labelsize = TAGSIZE, ticklabelsize = TICKSIZE_BIG)
        CairoMakie.rowgap!(gl_bcd, INTERNAL_GAP)
        CairoMakie.colgap!(gl_bcd, INTERNAL_GAP)
        CairoMakie.colsize!(fig.layout, 2, CairoMakie.Fixed(PANEL_BCD_W))

        CairoMakie.colgap!(fig.layout, 1, COLGAP_OUTER)

        out = joinpath(fig_dir, "plot_fig8_modal_composite.pdf")
        CairoMakie.save(out, fig)
        println("Saved $out")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
