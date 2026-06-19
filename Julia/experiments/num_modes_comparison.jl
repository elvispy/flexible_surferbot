"""
num_modes_comparison.jl

Experiment: overlay theoretical curves for two modal truncations (NUM_MODES=8 vs
NUM_MODES=5) on the coupled beam-end xM–κ heatmap to assess mode-count sensitivity.

Solid lines   = NUM_MODES_REF (default 8)
Dashed lines  = NUM_MODES_EXP (default 5)

Output: Julia/output/figures/experiments/beam_num_modes_cmp.pdf
"""

include(joinpath(@__DIR__, "..", "plot_dimensionless_diagnostics_LH.jl"))

const NUM_MODES_REF = 8
const NUM_MODES_EXP = 5

function run()
    output_dir = joinpath(@__DIR__, "..", "..", "output")
    exp_dir    = joinpath(output_dir, "figures", "experiments")
    mkpath(exp_dir)

    csv_path = joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv")
    art_cpl  = load_sweep(joinpath(output_dir, "jld2",
                   "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    params   = art_cpl.base_params

    shift    = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                     Float64(params.omega)^2)

    # ── Build heatmap base from the coupled beam plot ──────────────────────────
    # Re-use build_beam_end_plot for the heatmap + reference curves, then
    # overlay the low-mode curves on the returned plot object.
    p = build_beam_end_plot(art_cpl, csv_path, output_dir; xlim_min=-4.1)

    # ── Compute EI scatter grid (same as build_beam_end_plot) ─────────────────
    df_heat      = CSV.read(csv_path, DataFrame)
    all_logEI    = sort(unique(df_heat.log10_EI))
    logEI_axis   = all_logEI[all_logEI .- shift .>= -4.1]
    max_logK     = maximum(logEI_axis) - shift
    scatter_logK = collect(range(-4.1 - 0.1, max_logK; length=57))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    XLIMS = (-4.1, max_logK)
    YLIMS = (0.0, 0.5)

    # ── Overlay low-mode curves (dashed) ──────────────────────────────────────
    ctx_exp = theoretical_modal_context(params; output_dir=output_dir,
                                        num_modes=NUM_MODES_EXP)
    exp_colors = Dict("S" => :black, "A" => RGB(0.902, 0.624, 0.0),
                      "eta_1" => :steelblue, "eta_end" => :crimson)

    for cname in CURVE_NAMES
        res  = get_roots_theoretical_beam(EI_scatter, cname, ctx_exp)
        lk   = res.logEI .- shift
        xm   = res.xM_norm
        mask = (XLIMS[1] .<= lk .<= XLIMS[2]) .& (YLIMS[1] .<= xm .<= YLIMS[2])
        isempty(lk[mask]) && continue
        lk_line, xm_line, res_lks = split_xing_resonance(lk[mask], xm[mask])
        isempty(lk_line) || plot!(p, lk_line, xm_line;
              label=false, color=exp_colors[cname], linewidth=1.5, linestyle=:dash)
        for rlk in res_lks
            (XLIMS[1] <= rlk <= XLIMS[2]) || continue
            vline!(p, [rlk]; color=exp_colors[cname], linewidth=1.5, linestyle=:dash, label=false)
        end
    end

    # Legend patch to explain the line styles
    plot!(p, [NaN], [NaN]; color=:grey, linewidth=1.5, linestyle=:solid,
          label="$NUM_MODES_REF modes (ref)")
    plot!(p, [NaN], [NaN]; color=:grey, linewidth=1.5, linestyle=:dash,
          label="$NUM_MODES_EXP modes")

    out = joinpath(exp_dir, "beam_num_modes_cmp.pdf")
    savefig(p, out)
    println("Saved $out")
end

run()
