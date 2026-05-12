"""
plot_modal_decomp_eta_end_zeros.jl

Uses the full-solver sweep CSV (sweeper_coupled_full_grid.csv) — no a-priori law.

  - Selects rows with alpha > 0.99 (genuine |η_end|≈0, correct branch)
  - Filters to log₁₀(κ) > −2.8 and xM/L < 0.25, sorts by xM/L
  - Scatter-plots |q_n| vs xM/L for modes 0..4
  - Renders videos for 3 representative points (first, middle, last)
"""

using Surferbot, JLD2, Plots, LaTeXStrings, Printf, LinearAlgebra, CSV, DataFrames

const N_PLOT      = 5      # modes 0..4 to show in scatter
const ALPHA_MAX   = -0.99  # alpha < this to count as a genuine zero (α → −1 branch)

# ── Read CSV and select rows on the |η_end|=0 branch ─────────────────────────
function select_alpha_branch(csv_path; shift::Float64, logK_min, xM_max)
    df = CSV.read(csv_path, DataFrame)

    mask = (df.alpha .< ALPHA_MAX) .&
           ((df.log10_EI .- shift) .> logK_min) .&
           (df.xM_over_L .< xM_max)

    sub = df[mask, :]

    pts_logEI = Float64.(sub.log10_EI)
    pts_xM    = Float64.(sub.xM_over_L)
    pts_Q     = [[abs(complex(sub[i, Symbol("q_w$(n)_re")],
                              sub[i, Symbol("q_w$(n)_im")])) for n in 0:(N_PLOT-1)]
                 for i in 1:nrow(sub)]

    return (; logEI=pts_logEI, xM_norm=pts_xM, Q=pts_Q)
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    jld2_path  = joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2")
    csv_path   = joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv")

    artifact = Surferbot.Sweep.load_sweep(jld2_path)
    params   = artifact.base_params
    shift    = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 * Float64(params.omega)^2)

    @info "Selecting alpha < $(ALPHA_MAX) rows from CSV…"
    result = select_alpha_branch(csv_path; shift, logK_min=-2.8, xM_max=0.25)
    @info "Selected $(length(result.xM_norm)) points"

    # Sort by xM/L
    order  = sortperm(result.xM_norm)
    xM_sel = result.xM_norm[order]
    EI_sel = 10 .^ result.logEI[order]
    Q_sel  = result.Q[order]

    # Build Q_abs matrix [n_pts × N_PLOT]
    Q_abs = zeros(Float64, length(xM_sel), N_PLOT)
    for (i, q) in enumerate(Q_sel)
        Q_abs[i, :] = q
    end

    # ── Plot ─────────────────────────────────────────────────────────────────
    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
                    "#D55E00", "#CC79A7", "#000000"]
    mode_colors  = [okabe_ito[8], okabe_ito[1], okabe_ito[2], okabe_ito[3], okabe_ito[5]]
    mode_markers = [:circle, :rect, :diamond, :utriangle, :dtriangle]

    p = plot(
        xlabel      = L"x_M / L",
        ylabel      = L"|q_n|",
        title       = "Modal amplitudes on "*L"|\eta_{\mathrm{end}}|=0"*" curve\n"*
                      L"(\alpha < -0.99"*", coupled; "*L"\log_{10}\kappa > -2.8"*", "*
                      L"x_M/L < 0.25"*")",
        legend      = :topright,
        background_color_legend = RGBA(1,1,1,0.85),
        size        = (820, 520),
        margin      = 6Plots.mm,
        dpi         = 220,
        guidefontsize  = 13,
        tickfontsize   = 11,
        legendfontsize = 11,
        titlefontsize  = 12,
        fontfamily  = "Computer Modern",
        framestyle  = :box,
        grid        = true,
        gridalpha   = 0.25,
    )

    for n in 0:(N_PLOT-1)
        scatter!(p, xM_sel, Q_abs[:, n+1];
                 label             = latexstring("n = $n"),
                 color             = mode_colors[n+1],
                 marker            = mode_markers[n+1],
                 markersize        = 6,
                 markerstrokewidth = 0.6,
                 markerstrokecolor = :white,
                 markeralpha       = 0.92)
    end

    fig_dir = joinpath(output_dir, "figures")
    out_pdf = joinpath(fig_dir, "plot_modal_decomp_eta_end_zeros.pdf")
    out_png = joinpath(fig_dir, "plot_modal_decomp_eta_end_zeros.png")
    savefig(p, out_pdf)
    savefig(p, out_png)
    println("Saved $out_pdf")
    println("Saved $out_png")

    # ── Videos for 3 representative points (first, middle, last) ─────────────
    n_pts         = length(xM_sel)
    video_indices = [1, n_pts ÷ 2, n_pts]
    base_params   = params isa Surferbot.FlexibleParams ? params :
                    Surferbot.FlexibleParams(; (k => getproperty(params, k)
                        for k in fieldnames(Surferbot.FlexibleParams))...)

    for idx in video_indices
        xM        = xM_sel[idx]
        EI        = EI_sel[idx]
        motor_pos = xM * Float64(base_params.L_raft)
        run_params = Surferbot.Sweep.apply_parameter_overrides(
            base_params, (motor_position = motor_pos, EI = EI))
        @info @sprintf("Rendering video: xM/L=%.4f  EI=%.3e", xM, EI)
        res   = Surferbot.flexible_solver(run_params)
        bname = @sprintf("eta_end_zero_xM%.4f_EI%.2e", xM, EI)
        render_surferbot_run(res; outdir=fig_dir, basename=bname,
                             fps=30, duration_periods=8)
        println("Video saved: $bname.mp4")
    end
end

main()
