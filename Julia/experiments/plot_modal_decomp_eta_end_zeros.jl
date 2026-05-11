"""
plot_modal_decomp_eta_end_zeros.jl

For the coupled-raft, a-priori-theoretical case (cpl_theo):
  - Take the |η_end| = 0 scatter points with log₁₀(κ) > −2.8 and xM/L < 0.25
  - Sort them by xM/L (x-axis)
  - For each point, solve the a-priori modal law and extract |q_n|, n = 0..4
  - Scatter-plot each mode as a separate series
"""

using Surferbot, JLD2, Plots, LaTeXStrings, Printf, LinearAlgebra

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

const NUM_MODES = 5

# ── Helpers (mirrors plot_dimensionless_diagnostics.jl) ─────────────────────

function coerce_flexible_params(params)
    params isa Surferbot.FlexibleParams && return params
    pairs = Pair{Symbol, Any}[]
    for k in fieldnames(Surferbot.FlexibleParams)
        if hasproperty(params, k)
            push!(pairs, k => getproperty(params, k))
        end
    end
    return Surferbot.FlexibleParams(; pairs...)
end

function theoretical_modal_context(params; output_dir::AbstractString)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams; output_dir=output_dir, num_modes_basis=NUM_MODES)
    derived = Surferbot.derive_params(fparams)
    Psi = payload.psi_basis.Psi
    return (
        params        = fparams,
        derived       = derived,
        payload       = payload,
        mode_numbers  = collect(Int.(payload.mode_labels)),
        Psi           = Matrix{Float64}(Psi),
        x_raft        = collect(Float64.(payload.x_raft)),
        weights       = collect(Float64.(payload.weights)),
        w_end         = Psi[end, :],
        beta          = collect(Float64.(payload.beta)),
        Z_psi         = ComplexF64.(payload.Z_psi),
        c_hydro       = derived.d * fparams.rho * fparams.g,
        F0            = fparams.motor_inertia * fparams.omega^2,
        forcing_width = fparams.forcing_width,
    )
end

function solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
    p   = theory_ctx.params
    F_c = theory_ctx.derived.F_c
    L_c = theory_ctx.derived.L_c
    x_raft_adim = theory_ctx.x_raft ./ L_c
    loads_adim  = (theory_ctx.F0 / F_c) .*
                  Surferbot.gaussian_load(Float64(xM_norm), p.forcing_width, x_raft_adim)
    loads_dim   = loads_adim .* (F_c / L_c)
    F_psi       = theory_ctx.Psi' * (loads_dim .* theory_ctx.weights)
    D = ComplexF64.(EI .* theory_ctx.beta .^ 4
                    .- p.rho_raft * p.omega^2
                    .+ theory_ctx.c_hydro)
    A_sys = Diagonal(D) - theory_ctx.Z_psi
    return -(A_sys \ ComplexF64.(F_psi))
end

function find_filtered_minima(xgrid, values, ratio; ratio_cutoff::Float64)
    roots = Float64[]
    for i in 2:(length(xgrid) - 1)
        if values[i] <= values[i-1] && values[i] <= values[i+1] && ratio[i] < ratio_cutoff
            push!(roots, Float64(xgrid[i]))
        end
    end
    return roots
end

function theoretical_eta_end_zeros(artifact; output_dir::AbstractString)
    params     = artifact.base_params
    EI_list    = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    xM_grid    = collect(range(0.0, 0.49, length=401))
    theory_ctx = theoretical_modal_context(params; output_dir=output_dir)

    RATIO_CUTOFF = 0.5
    pts_logEI = Float64[]
    pts_xM    = Float64[]

    for (iei, EI) in enumerate(EI_list)
        abs_eta_end = Float64[]; abs_eta_1 = Float64[]
        for xM_norm in xM_grid
            q = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
            w = theory_ctx.w_end
            mn = theory_ctx.mode_numbers
            S = sum(iseven(mn[j]) ? q[j]*w[j] : zero(ComplexF64) for j in eachindex(mn))
            A = sum(isodd(mn[j])  ? q[j]*w[j] : zero(ComplexF64) for j in eachindex(mn))
            push!(abs_eta_end, abs(S + A))
            push!(abs_eta_1,   abs(S - A))
        end
        denom = abs_eta_end .+ abs_eta_1 .+ eps()
        ratio = abs_eta_end ./ denom
        roots = find_filtered_minima(xM_grid, abs_eta_end, ratio; ratio_cutoff=RATIO_CUTOFF)
        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM, theory_ctx)
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    jld2_path  = joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2")
    artifact   = Surferbot.Sweep.load_sweep(jld2_path)
    params     = artifact.base_params

    shift = log10(params.rho_raft * params.L_raft^4 * params.omega^2)

    @info "Computing |η_end|=0 zero curves (theoretical, coupled)…"
    result     = theoretical_eta_end_zeros(artifact; output_dir=output_dir)
    theory_ctx = result.theory_ctx

    logK    = result.logEI .- shift
    xM_norm = result.xM_norm

    # Filter: log₁₀(κ) > −2.8 and xM/L < 0.25
    mask     = (logK .> -2.8) .& (xM_norm .< 0.25)
    xM_sel   = xM_norm[mask]
    EI_sel   = 10 .^ (result.logEI[mask])
    @info "Selected $(sum(mask)) points"

    # Sort by xM/L
    order  = sortperm(xM_sel)
    xM_sel = xM_sel[order]
    EI_sel = EI_sel[order]

    # Modal amplitudes |q_n|, n = 0..4
    Q_abs = zeros(Float64, length(xM_sel), NUM_MODES)
    for (i, (EI, xM)) in enumerate(zip(EI_sel, xM_sel))
        q = solve_theoretical_modal_response(EI, xM, theory_ctx)
        for n in 0:(NUM_MODES-1)
            Q_abs[i, n+1] = abs(q[n+1])
        end
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
                      "(coupled, a-priori; "*L"\log_{10}\kappa > -2.8"*", "*
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

    for n in 0:(NUM_MODES-1)
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
    n_pts = length(xM_sel)
    video_indices = [1, n_pts ÷ 2, n_pts]
    base_params = coerce_flexible_params(artifact.base_params)

    for idx in video_indices
        xM  = xM_sel[idx]
        EI  = EI_sel[idx]
        motor_pos = xM * base_params.L_raft   # xM_over_L * L_raft
        run_params = Surferbot.Sweep.apply_parameter_overrides(
            base_params, (motor_position = motor_pos, EI = EI))
        @info @sprintf("Rendering video: xM/L=%.4f  EI=%.3e", xM, EI)
        res = Surferbot.flexible_solver(run_params)
        bname = @sprintf("eta_end_zero_xM%.4f_EI%.2e", xM, EI)
        render_surferbot_run(res; outdir=fig_dir, basename=bname,
                             fps=30, duration_periods=8)
        println("Video saved: $bname.mp4")
    end
end

main()
