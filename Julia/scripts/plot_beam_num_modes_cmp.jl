# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
"""
plot_beam_num_modes_cmp.jl

Beam-end (coupled + uncoupled) diagnostic plots with theoretical zero-crossing
curves overlaid for three modal truncation levels:

  N = 8 modes  →  solid   lines   (default)
  N = 6 modes  →  dashed  lines
  N = 4 modes  →  dotted  lines

The heatmap background (α from CSV sweep data) is identical to the one in
plot_dimensionless_diagnostics_LH.jl.  Only the theoretical overlay changes.

Output: output/figures/beam_num_modes_cmp_{cpl,ucpl}.{pdf,png}

Usage:
  julia --project=. --threads auto scripts/plot_beam_num_modes_cmp.jl
"""

using Surferbot, JLD2, Plots, LaTeXStrings, Printf, LinearAlgebra,
      CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "plot_dimensionless_diagnostics_LH.jl"))
# ModalPressureMap, coerce_flexible_params, theoretical_endpoint_diagnostics_beam,
# solve_theoretical_modal_response, get_roots_theoretical_beam, cluster_branches,
# CURVE_NAMES, BEAM_CURVE_LABELS, RESONANCE_N_PTS, RESONANCE_ALPHA_CUTOFF are all
# available after the include above.

# ── Parametrised modal context (not bound to NUM_MODES constant) ───────────────
function make_beam_ctx_nm(params; output_dir::AbstractString, num_modes::Int)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
                  fparams; output_dir=output_dir, num_modes_basis=num_modes)
    derived = Surferbot.derive_params(fparams)
    Psi     = payload.psi_basis.Psi
    return (
        params        = fparams,
        derived       = derived,
        payload       = payload,
        mode_numbers  = collect(Int.(payload.mode_labels)),
        Psi           = Matrix{Float64}(Psi),
        x_raft        = collect(Float64.(payload.x_raft)),
        weights       = collect(Float64.(payload.weights)),
        w_end         = Psi[end, :],
        w_start       = Psi[1,   :],
        beta          = collect(Float64.(payload.beta)),
        Z_psi         = ComplexF64.(payload.Z_psi),
        c_hydro       = derived.d * fparams.rho * fparams.g,
        F0            = fparams.motor_inertia * fparams.omega^2,
        forcing_width = fparams.forcing_width,
        C_sigma       = derived.d * fparams.sigma .*
                          (Psi[end, :] * transpose(ComplexF64.(payload.s_vec)) .+
                           Psi[1,   :] * transpose(ComplexF64.(payload.s_vec_left))),
    )
end

# ── Heatmap builder (identical logic to build_beam_end_plot, no overlays) ──────
function build_beam_alpha(artifact, csv_path; xlim_min::Float64)
    params     = artifact.base_params
    shift      = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                       Float64(params.omega)^2)
    df_heat    = CSV.read(csv_path, DataFrame)
    all_logEI  = sort(unique(df_heat.log10_EI))
    logEI_axis = all_logEI[all_logEI .- shift .>= xlim_min]
    xM_axis    = sort(unique(df_heat.xM_over_L))
    max_logK   = maximum(logEI_axis) - shift

    alpha = zeros(Float64, length(xM_axis), length(logEI_axis))
    lut = Dict{Tuple{Float64,Float64}, Float64}(
        (row.log10_EI, row.xM_over_L) =>
            Surferbot.Analysis.beam_asymmetry(
                complex(row.eta_1_beam_re,  row.eta_1_beam_im),
                complex(row.eta_end_beam_re, row.eta_end_beam_im))
        for row in eachrow(df_heat))
    for (j, le) in enumerate(logEI_axis), (i, xm) in enumerate(xM_axis)
        alpha[i, j] = lut[(le, xm)]
    end
    if maximum(xM_axis) < 0.5
        xM_axis = vcat(xM_axis, [0.5])
        alpha   = vcat(alpha, alpha[end:end, :])
    end
    return alpha, xM_axis, logEI_axis, shift, max_logK
end

# ── Main figure builder ────────────────────────────────────────────────────────
function make_nm_comparison(artifact, csv_path, output_dir;
                             xlim_min::Float64, outname::String)
    params = artifact.base_params
    alpha, xM_axis, logEI_axis, shift, max_logK =
        build_beam_alpha(artifact, csv_path; xlim_min=xlim_min)

    XLIMS        = (xlim_min, max_logK)
    YLIMS        = (0.0, 0.5)
    # 181 (not 57): the coarse grid missed narrow resonances entirely and
    # under-sampled branches approaching them into a jagged/spurious-looking
    # wiggle; see plot_dimensionless_diagnostics_LH.jl for the same fix.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK; length=181))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]

    plt_opts = (
        xlabel   = L"\kappa",
        xticks   = kappa_exp_xticks(XLIMS),
        ylabel   = L"x_M / L",
        colormap = :balance,
        clims    = (-1, 1),
        levels   = 51,
        interpolate  = true,
        xlims    = XLIMS,
        ylims    = YLIMS,
        legend   = :bottomright,
        background_color_legend = RGBA(1, 1, 1, 0.85),
        foreground_color_legend = :black,
        legend_font_halign = :left,
        size     = (820, 640),
        margin   = 6Plots.mm,
        dpi      = 300,
        titlefontsize      = 13,
        guidefontsize      = 14,
        tickfontsize       = 12,
        legendfontsize     = 10,
        fontfamily         = "Computer Modern",
        framestyle         = :box,
        grid               = false,
        tick_direction     = :out,
        colorbar_title     = L"\alpha",
        colorbar_titlefontsize = 14,
        colorbar_tickfontsize  = 11,
    )

    p = heatmap(logEI_axis .- shift, xM_axis, alpha; plt_opts...)

    # N = 8 (solid), 6 (dashed), 4 (dotted) — keep equal weight so no family looks secondary
    modes_cfg = [(8, :solid, 4.0), (6, :dash, 4.0), (4, :dot, 4.0)]

    for (nm, lstyle, lw) in modes_cfg
        @info "N=$nm: computing roots"
        ctx = make_beam_ctx_nm(params; output_dir=output_dir, num_modes=nm)
        for (i, cname) in enumerate(CURVE_NAMES)
            res = get_roots_theoretical_beam(EI_scatter, cname, ctx)
            logK_pts = res.logEI .- shift
            mask = (XLIMS[1] .<= logK_pts .<= XLIMS[2]) .&
                   (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
            isempty(logK_pts[mask]) && continue
            lk_path, xm_path, res_lks =
                cluster_branches(logK_pts[mask], res.xM_norm[mask])
            # Condition labels only on the N=8 pass; line-style key via dummy entries
            lbl = nm == 8 ? BEAM_CURVE_LABELS[i] : false
            isempty(lk_path) || plot!(p, lk_path, xm_path;
                color=curve_colors[i], linestyle=lstyle, linewidth=lw, label=lbl)
            for rlk in res_lks
                (XLIMS[1] <= rlk <= XLIMS[2]) || continue
                vline!(p, [rlk]; color=curve_colors[i], linestyle=lstyle,
                       linewidth=lw, label=false)
            end
        end
    end

    # Dummy entries to explain line-style encoding in the legend
    gray = RGB(0.4, 0.4, 0.4)
    plot!(p, [NaN], [NaN]; color=gray, linestyle=:solid, linewidth=4.0, label="N = 8")
    plot!(p, [NaN], [NaN]; color=gray, linestyle=:dash,  linewidth=4.0, label="N = 6")
    plot!(p, [NaN], [NaN]; color=gray, linestyle=:dot,   linewidth=4.0, label="N = 4")

    fig_dir = joinpath(output_dir, "figures")
    mkpath(fig_dir)
    for ext in (".pdf", ".png")
        out = joinpath(fig_dir, outname * ext)
        savefig(p, out)
        println("Saved $out")
    end
    return p
end

# ── Entry point ────────────────────────────────────────────────────────────────
function main()
    output_dir = joinpath(@__DIR__, "..", "output")

    art_cpl  = Surferbot.Sweep.load_sweep(joinpath(output_dir, "jld2",
                   "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    art_ucpl = Surferbot.Sweep.load_sweep(joinpath(output_dir, "jld2",
                   "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))

    make_nm_comparison(art_cpl,
        joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
        output_dir; xlim_min=-4.0, outname="plot_beam_num_modes_cmp_cpl")

    make_nm_comparison(art_ucpl,
        joinpath(output_dir, "csv", "sweeper_uncoupled_full_grid.csv"),
        output_dir; xlim_min=-4.0, outname="plot_beam_num_modes_cmp_ucpl")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
