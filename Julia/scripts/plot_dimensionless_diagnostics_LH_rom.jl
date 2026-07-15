# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_dimensionless_diagnostics_LH_rom.jl

ROM-only counterpart of plot_dimensionless_diagnostics_LH.jl's build_LH_plot.

Instead of reading α_LH from the 30,000-row full-PDE sweep CSV, this script
fills the entire (log10_EI, xM_over_L) heatmap using only the reduced-order
model: solve_theoretical_modal_response (an 8x8 complex linear solve against
the cached per-mode Ψ basis) + theoretical_endpoint_diagnostics_LH (linear
superposition of the per-mode domain-end elevations) give η at both domain
ends directly, from which α_LH follows via beam_asymmetry — no cluster sweep
required.

The (log10_EI, xM_over_L) grid axes are read from the existing sweep CSV (for
visual comparability with plot_dimensionless_diagnostics_LH_cpl_theo.pdf) but
none of the CSV's η data is used — only the axis values, and only every
EI_STRIDE-th / XM_STRIDE-th of them, since each ROM evaluation is cheap but
30,000 is still more resolution than the figure needs.

Output: output/figures/plot_dimensionless_diagnostics_LH_rom.pdf
"""

using Surferbot, JLD2, Plots, LaTeXStrings, Printf, LinearAlgebra,
      CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "plot_dimensionless_diagnostics_LH.jl"))
# theoretical_modal_context_LH, solve_theoretical_modal_response,
# theoretical_endpoint_diagnostics_LH, get_roots_theoretical_LH,
# cluster_branches, CURVE_NAMES, CURVE_LABELS are all available after
# the include above.

const EI_STRIDE = 3
const XM_STRIDE = 2

# ── ROM-only heatmap: no CSV η columns, just the axis definition ─────────────
function build_LH_alpha_rom(theory_ctx, csv_path; xlim_min::Float64)
    params = theory_ctx.params
    shift  = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                   Float64(params.omega)^2)

    all_logEI  = sort(unique(CSV.read(csv_path, DataFrame; select=[:log10_EI]).log10_EI))
    all_xM     = sort(unique(CSV.read(csv_path, DataFrame; select=[:xM_over_L]).xM_over_L))
    logEI_axis = all_logEI[1:EI_STRIDE:end]
    logEI_axis = logEI_axis[logEI_axis .- shift .>= xlim_min]
    xM_axis    = all_xM[1:XM_STRIDE:end]
    max_logK   = maximum(logEI_axis) - shift

    alpha_LH = zeros(Float64, length(xM_axis), length(logEI_axis))
    for (j, logEI) in enumerate(logEI_axis), (i, xM) in enumerate(xM_axis)
        EI   = 10^logEI
        q    = solve_theoretical_modal_response(EI, xM, theory_ctx)
        diag = theoretical_endpoint_diagnostics_LH(q, theory_ctx)
        alpha_LH[i, j] = Surferbot.Analysis.beam_asymmetry(diag.eta_LH_1, diag.eta_LH_end)
    end

    if minimum(xM_axis) > -0.5
        xM_axis  = vcat([-0.5], xM_axis)
        alpha_LH = vcat(alpha_LH[1:1, :], alpha_LH)
    end

    return alpha_LH, xM_axis, logEI_axis, shift, max_logK
end

function build_LH_rom_plot(artifact, csv_path, output_dir; xlim_min::Float64)
    theory_ctx = theoretical_modal_context_LH(artifact.base_params; output_dir=output_dir)
    alpha_LH, xM_axis, logEI_axis, shift, max_logK =
        build_LH_alpha_rom(theory_ctx, csv_path; xlim_min=xlim_min)

    XLIMS = (xlim_min, max_logK)
    YLIMS = (-0.5, 0.0)

    scatter_logK = collect(range(xlim_min - 0.1, max_logK; length=57))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    results = Dict{String, NamedTuple}()
    for cname in CURVE_NAMES
        @info "Computing ROM LH roots: $cname"
        res = get_roots_theoretical_LH(artifact, cname;
                                        output_dir=output_dir,
                                        EI_list=EI_scatter)
        results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
    end

    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
    curve_styles = [:solid, :solid, :solid, :solid]

    plt_opts = (
        xlabel  = L"\log_{10}\,\kappa",
        ylabel  = L"x_M / L",
        colormap = :balance,
        clims   = (-1, 1),
        levels  = 51,
        interpolate = true,
        xlims   = XLIMS,
        ylims   = YLIMS,
        legend  = :bottomright,
        background_color_legend = RGBA(1, 1, 1, 0.85),
        foreground_color_legend = :black,
        legend_font_halign = :left,
        size    = (820, 640),
        margin  = 6Plots.mm,
        dpi     = 300,
        titlefontsize     = 14,
        guidefontsize     = 14,
        tickfontsize      = 12,
        legendfontsize    = 11,
        fontfamily        = "Computer Modern",
        framestyle        = :box,
        grid              = false,
        tick_direction    = :out,
        colorbar_title    = L"\alpha",
        colorbar_titlefontsize = 14,
        colorbar_tickfontsize  = 11,
    )

    p = heatmap(logEI_axis .- shift, xM_axis, alpha_LH; plt_opts...)

    for (i, cname) in enumerate(CURVE_NAMES)
        res  = results[cname]
        mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
               (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
        isempty(res.logK[mask]) && continue
        lk_path, xm_path, res_lks = cluster_branches(res.logK[mask], res.xM_norm[mask])
        isempty(lk_path) || plot!(p, lk_path, xm_path;
            label      = CURVE_LABELS[i],
            color      = curve_colors[i],
            linestyle  = curve_styles[i],
            linewidth  = 4.0)
        for rlk in res_lks
            (XLIMS[1] <= rlk <= XLIMS[2]) || continue
            vline!(p, [rlk]; color=curve_colors[i], linewidth=4.0, label=false)
        end
    end

    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    art_cpl = load_sweep(joinpath(output_dir, "jld2",
                                  "sweep_motor_position_EI_coupled_from_matlab.jld2"))

    t0 = time()
    p_rom = build_LH_rom_plot(art_cpl,
                               joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                               output_dir; xlim_min=-4.0)
    @info "ROM heatmap + overlay build time: $(round(time() - t0, digits=2)) s"

    out_rom = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_rom.pdf")
    savefig(p_rom, out_rom)
    println("Saved $out_rom")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
