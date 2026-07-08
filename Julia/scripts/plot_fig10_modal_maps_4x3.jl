using CairoMakie
using CSV
using DataFrames
using JLD2
using LinearAlgebra
using Printf
using Surferbot

const JULIA_DIR = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_DIR = joinpath(JULIA_DIR, "output")
include(joinpath(JULIA_DIR, "scripts", "plot_dimensionless_diagnostics_LH.jl"))

function make_ctx_nm(params; num_modes::Int)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams; output_dir=OUTPUT_DIR, num_modes_basis=num_modes)
    derived = Surferbot.derive_params(fparams)
    Psi = payload.psi_basis.Psi
    return (
        params = fparams,
        derived = derived,
        payload = payload,
        mode_numbers = collect(Int.(payload.mode_labels)),
        Psi = Matrix{Float64}(Psi),
        x_raft = collect(Float64.(payload.x_raft)),
        weights = collect(Float64.(payload.weights)),
        w_end = Psi[end, :],
        w_start = Psi[1, :],
        beta = collect(Float64.(payload.beta)),
        Z_psi = ComplexF64.(payload.Z_psi),
        c_hydro = derived.d * fparams.rho * fparams.g,
        F0 = fparams.motor_inertia * fparams.omega^2,
        forcing_width = fparams.forcing_width,
        C_sigma = derived.d * fparams.sigma .*
            (Psi[end, :] * transpose(ComplexF64.(payload.s_vec)) .+
             Psi[1, :] * transpose(ComplexF64.(payload.s_vec_left))),
        a_vec = ComplexF64.(payload.a_vec),
        a_vec_left = ComplexF64.(payload.a_vec_left),
    )
end

function csv_alpha_map(csv_path, artifact; kind::Symbol, xlim_min::Float64)
    params = artifact.base_params
    shift = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                  Float64(params.omega)^2)
    df = CSV.read(csv_path, DataFrame)
    logEI_all = sort(unique(df.log10_EI))
    logEI = logEI_all[logEI_all .- shift .>= xlim_min]
    xM = sort(unique(df.xM_over_L))
    alpha = zeros(Float64, length(xM), length(logEI))

    left_cols = kind == :farfield ? (:eta_1_domain_re, :eta_1_domain_im) :
                                    (:eta_1_beam_re, :eta_1_beam_im)
    right_cols = kind == :farfield ? (:eta_end_domain_re, :eta_end_domain_im) :
                                     (:eta_end_beam_re, :eta_end_beam_im)

    lut = Dict{Tuple{Float64, Float64}, Float64}()
    for row in eachrow(df)
        eta_left = complex(row[left_cols[1]], row[left_cols[2]])
        eta_right = complex(row[right_cols[1]], row[right_cols[2]])
        lut[(row.log10_EI, row.xM_over_L)] =
            Surferbot.Analysis.beam_asymmetry(eta_left, eta_right)
    end
    for (j, le) in enumerate(logEI), (i, xm) in enumerate(xM)
        alpha[i, j] = lut[(le, xm)]
    end
    if maximum(xM) < 0.5
        xM = vcat(xM, [0.5])
        alpha = vcat(alpha, alpha[end:end, :])
    end
    return (; kappa = 10.0 .^ (logEI .- shift), xM, alpha,
            logK = logEI .- shift, xlim = (xlim_min, maximum(logEI) - shift))
end

function rom_alpha_map(artifact, axes_ref; kind::Symbol, num_modes::Int)
    ctx = make_ctx_nm(artifact.base_params; num_modes)
    alpha = zeros(Float64, length(axes_ref.xM), length(axes_ref.kappa))
    logEI = log10.(axes_ref.kappa) .+
        log10(Float64(ctx.params.rho_raft) * Float64(ctx.params.L_raft)^4 *
              Float64(ctx.params.omega)^2)
    for (j, le) in enumerate(logEI), (i, xm) in enumerate(axes_ref.xM)
        q = solve_theoretical_modal_response(10.0^le, xm, ctx)
        diag = kind == :farfield ?
            theoretical_endpoint_diagnostics_LH(q, ctx) :
            theoretical_endpoint_diagnostics_beam(q, ctx)
        if kind == :farfield
            alpha[i, j] = Surferbot.Analysis.beam_asymmetry(
                diag.eta_LH_1, diag.eta_LH_end)
        else
            alpha[i, j] = Surferbot.Analysis.beam_asymmetry(
                diag.eta_beam_1, diag.eta_beam_end)
        end
    end
    return (; kappa = axes_ref.kappa, xM = axes_ref.xM, alpha,
            logK = axes_ref.logK, xlim = axes_ref.xlim)
end

function exp_ticks(log_xlim)
    lo = ceil(Int, log_xlim[1])
    hi = floor(Int, log_xlim[2])
    vals = 10.0 .^ collect(lo:hi)
    labels = [rich("10", superscript(string(i))) for i in lo:hi]
    return vals, labels
end

function draw_panel!(ax, data; show_xlabel::Bool, show_yticks::Bool, show_xticks::Bool)
    hm = CairoMakie.heatmap!(ax, data.kappa, data.xM, data.alpha';
        colormap = :balance, colorrange = (-1, 1))
    CairoMakie.xlims!(ax, 10.0^data.xlim[1], 10.0^data.xlim[2])
    CairoMakie.ylims!(ax, 0, 0.5)
    ax.xscale = log10
    ticks, labels = exp_ticks(data.xlim)
    ax.xticks = (ticks, labels)
    ax.yticks = 0:0.25:0.5
    ax.xlabel = show_xlabel ? L"\kappa" : ""
    if !show_xticks
        CairoMakie.hidexdecorations!(ax; grid = false)
    end
    if !show_yticks
        CairoMakie.hideydecorations!(ax; grid = false)
    end
    return hm
end

function main()
    art_cpl = Surferbot.Sweep.load_sweep(joinpath(
        OUTPUT_DIR, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    art_ucpl = Surferbot.Sweep.load_sweep(joinpath(
        OUTPUT_DIR, "jld2", "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))

    csv_cpl = joinpath(OUTPUT_DIR, "csv", "sweeper_coupled_full_grid.csv")
    csv_ucpl = joinpath(OUTPUT_DIR, "csv", "sweeper_uncoupled_full_grid.csv")

    xlim_min = -4.0
    refs = [
        csv_alpha_map(csv_cpl, art_cpl; kind = :farfield, xlim_min),
        csv_alpha_map(csv_cpl, art_cpl; kind = :beam, xlim_min),
        csv_alpha_map(csv_ucpl, art_ucpl; kind = :beam, xlim_min),
    ]

    maps = Matrix{Any}(undef, 4, 3)
    maps[1, :] .= refs
    for (r, nm) in zip(2:4, (8, 6, 4))
        @info "Building ROM row" nm
        maps[r, 1] = rom_alpha_map(art_cpl, refs[1]; kind = :farfield, num_modes = nm)
        maps[r, 2] = rom_alpha_map(art_cpl, refs[2]; kind = :beam, num_modes = nm)
        maps[r, 3] = rom_alpha_map(art_ucpl, refs[3]; kind = :beam, num_modes = nm)
    end

    CairoMakie.with_theme(CairoMakie.theme_latexfonts()) do
        fig = CairoMakie.Figure(size = (780, 900), backgroundcolor = :white,
            fontsize = 9, figure_padding = (6, 10, 4, 6))
        col_titles = [L"\eta(\pm\ell),\ \mathrm{coupled}",
                      L"\eta(\pm 1/2),\ \mathrm{coupled}",
                      L"\eta(\pm 1/2),\ \mathrm{uncoupled}"]
        row_titles = ["CSV", "N = 8", "N = 6", "N = 4"]
        last_hm = nothing
        axes = Matrix{CairoMakie.Axis}(undef, 4, 3)
        for c in 1:3
            CairoMakie.Label(fig[1, c + 2], col_titles[c], fontsize = 10, tellwidth = false)
        end
        CairoMakie.Label(fig[2:5, 1], L"x_M/L", rotation = pi / 2,
            fontsize = 10, tellheight = false)
        for r in 1:4
            CairoMakie.Label(fig[r + 1, 2], row_titles[r], rotation = pi / 2,
                fontsize = 10, tellheight = false)
            for c in 1:3
                ax = CairoMakie.Axis(fig[r + 1, c + 2],
                    xticklabelsize = 8, yticklabelsize = 8,
                    xlabelsize = 10, ylabelsize = 10,
                    xminorticksvisible = true, xminorgridvisible = false,
                    yminorgridvisible = false)
                axes[r, c] = ax
                last_hm = draw_panel!(ax, maps[r, c];
                    show_xlabel = r == 4,
                    show_yticks = c == 1,
                    show_xticks = r == 4)
            end
        end
        for r in 1:4, c in 1:3
            c > 1 && CairoMakie.linkyaxes!(axes[r, 1], axes[r, c])
        end
        for c in 1:3, r in 1:4
            r > 1 && CairoMakie.linkxaxes!(axes[1, c], axes[r, c])
        end
        CairoMakie.Colorbar(fig[2:5, 6], last_hm, label = L"\alpha",
            labelsize = 10, ticklabelsize = 8, width = 12)
        CairoMakie.colgap!(fig.layout, 5)
        CairoMakie.rowgap!(fig.layout, 7)
        CairoMakie.colsize!(fig.layout, 1, CairoMakie.Relative(0.025))
        CairoMakie.colsize!(fig.layout, 2, CairoMakie.Relative(0.035))
        CairoMakie.colsize!(fig.layout, 6, CairoMakie.Relative(0.045))
        CairoMakie.rowsize!(fig.layout, 1, CairoMakie.Relative(0.035))

        out = joinpath(OUTPUT_DIR, "figures", "plot_fig10_modal_maps_4x3.pdf")
        mkpath(dirname(out))
        CairoMakie.save(out, fig)
        println("Saved $out")
    end
end

main()
