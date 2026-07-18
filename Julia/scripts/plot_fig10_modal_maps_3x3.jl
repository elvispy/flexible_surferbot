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

const NEWCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
const NEWCM_FONT = joinpath(NEWCM_DIR, "NewCM10-Regular.otf")
const NEWCM_MATH = joinpath(NEWCM_DIR, "NewCMMath-Regular.otf")

function setup_newcm_mathfonts()
    MTE_ID = Base.PkgId(Base.UUID("0a4f8689-d25c-4efe-a92b-7142dfc1aa53"), "MathTeXEngine")
    MTE = get(Base.loaded_modules, MTE_ID, nothing)
    MTE === nothing && return
    isfile(NEWCM_MATH) || return
    try
        MTE.set_texfont_family!(
            regular    = joinpath(NEWCM_DIR, "NewCM10-Regular.otf"),
            italic     = joinpath(NEWCM_DIR, "NewCM10-Italic.otf"),
            bold       = joinpath(NEWCM_DIR, "NewCM10-Bold.otf"),
            bolditalic = joinpath(NEWCM_DIR, "NewCM10-BoldItalic.otf"),
            math       = NEWCM_MATH,
        )
    catch
    end
end

struct RowSpec
    title
    artifact::Symbol
    csv_file::String
    kind::Symbol
end

struct ColSpec
    title::String
    num_modes::Union{Nothing, Int}
end

const ROW_SPECS = [
    RowSpec(L"\bar{\eta}(\pm\bar{\ell}),\ \Lambda\ne0",
            :coupled, "sweeper_coupled_full_grid.csv", :farfield),
    RowSpec(L"\bar{\eta}(\pm 1/2),\ \Lambda\ne0",
            :coupled, "sweeper_coupled_full_grid.csv", :beam),
    RowSpec(L"\bar{\eta}(\pm 1/2),\ \Lambda=0",
            :uncoupled, "sweeper_uncoupled_full_grid.csv", :beam),
]

const COL_SPECS = [
    ColSpec("Full Sweep", nothing),
    ColSpec("N = 8", 8),
    ColSpec("N = 4", 4),
]

function csv_columns(kind::Symbol)
    if kind == :farfield
        return ((:eta_1_domain_re, :eta_1_domain_im),
                (:eta_end_domain_re, :eta_end_domain_im))
    elseif kind == :beam
        return ((:eta_1_beam_re, :eta_1_beam_im),
                (:eta_end_beam_re, :eta_end_beam_im))
    else
        error("Unknown map kind: $kind")
    end
end

function diagnostic_alpha(q, ctx, kind::Symbol)
    if kind == :farfield
        diag = theoretical_endpoint_diagnostics_LH(q, ctx)
        @assert hasproperty(diag, :eta_LH_1)
        @assert hasproperty(diag, :eta_LH_end)
        return Surferbot.Analysis.beam_asymmetry(diag.eta_LH_1, diag.eta_LH_end)
    elseif kind == :beam
        diag = theoretical_endpoint_diagnostics_beam(q, ctx)
        @assert hasproperty(diag, :eta_beam_1)
        @assert hasproperty(diag, :eta_beam_end)
        return Surferbot.Analysis.beam_asymmetry(diag.eta_beam_1, diag.eta_beam_end)
    else
        error("Unknown map kind: $kind")
    end
end

function make_ctx_nm(params; num_modes::Int)
    fparams = coerce_flexible_params(params)
    derived = Surferbot.derive_params(fparams)
    key = ModalPressureMap.operator_cache_key(fparams; num_modes_basis = num_modes)
    cache_file = ModalPressureMap.cache_path(
        OUTPUT_DIR, ModalPressureMap.DEFAULT_OPERATOR_CACHE_FILE)
    cached = ModalPressureMap.load_cached_result(cache_file, key)
    if isnothing(cached)
        @assert iszero(derived.d) "Missing cached modal pressure map for N=$num_modes at $cache_file"
        cached = ModalPressureMap.zero_modal_pressure_map(fparams; num_modes_basis = num_modes)
    end
    payload = cached
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

function csv_alpha_map(csv_path, artifact, spec::RowSpec; xlim_min::Float64)
    params = artifact.base_params
    shift = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                  Float64(params.omega)^2)
    df = CSV.read(csv_path, DataFrame)
    left_cols, right_cols = csv_columns(spec.kind)
    names_df = Set(Symbol.(names(df)))
    for col in (left_cols..., right_cols...)
        @assert col in names_df "Missing CSV column $col for $(spec.title)"
    end

    logEI_all = sort(unique(df.log10_EI))
    logEI = logEI_all[logEI_all .- shift .>= xlim_min]
    xM = sort(unique(df.xM_over_L))
    alpha = zeros(Float64, length(xM), length(logEI))

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
    if minimum(xM) > -0.5
        xM = vcat([-0.5], xM)
        alpha = vcat(alpha[1:1, :], alpha)
    end
    return (; kappa = 10.0 .^ (logEI .- shift), xM, alpha,
            logK = logEI .- shift, xlim = (xlim_min, maximum(logEI) - shift))
end

function axes_match(a, b)
    return length(a.kappa) == length(b.kappa) &&
           length(a.xM) == length(b.xM) &&
           all(isapprox.(a.kappa, b.kappa; rtol = 1e-12, atol = 0.0)) &&
           all(isapprox.(a.xM, b.xM; rtol = 1e-12, atol = 0.0))
end

function rom_alpha_map(artifact, axes_ref, spec::RowSpec; num_modes::Int)
    ctx = make_ctx_nm(artifact.base_params; num_modes)
    alpha = zeros(Float64, length(axes_ref.xM), length(axes_ref.kappa))
    logEI = log10.(axes_ref.kappa) .+
        log10(Float64(ctx.params.rho_raft) * Float64(ctx.params.L_raft)^4 *
              Float64(ctx.params.omega)^2)
    for (j, le) in enumerate(logEI), (i, xm) in enumerate(axes_ref.xM)
        q = solve_theoretical_modal_response(10.0^le, xm, ctx)
        alpha[i, j] = diagnostic_alpha(q, ctx, spec.kind)
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

function draw_panel!(ax, data; show_xlabel::Bool, show_yticks::Bool, show_xticks::Bool,
                     xaxisposition::Symbol=:bottom, xticks=-0.5:0.25:0)
    hm = CairoMakie.heatmap!(ax, data.xM, data.kappa, data.alpha;
        colormap = :balance, colorrange = (-1, 1))
    CairoMakie.xlims!(ax, -0.5, 0)
    CairoMakie.ylims!(ax, 10.0^data.xlim[1], 10.0^data.xlim[2])
    ax.yscale = log10
    ticks, labels = exp_ticks(data.xlim)
    ax.yticks = (ticks, labels)
    ax.xticks = xticks
    ax.xaxisposition = xaxisposition
    ax.xlabel = show_xlabel ? L"x_M/L" : ""
    ax.ylabel = show_yticks ? L"\kappa" : ""
    if !show_xticks
        CairoMakie.hidexdecorations!(ax; label = !show_xlabel, grid = false)
    end
    if !show_yticks
        CairoMakie.hideydecorations!(ax; grid = false)
    end
    return hm
end

function main()
    setup_newcm_mathfonts()

    artifacts = Dict(
        :coupled => Surferbot.Sweep.load_sweep(joinpath(
            OUTPUT_DIR, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2")),
        :uncoupled => Surferbot.Sweep.load_sweep(joinpath(
            OUTPUT_DIR, "jld2", "sweep_motor_position_EI_uncoupled_from_matlab.jld2")),
    )

    xlim_min = -4.0
    refs = [
        csv_alpha_map(joinpath(OUTPUT_DIR, "csv", spec.csv_file),
                      artifacts[spec.artifact], spec; xlim_min)
        for spec in ROW_SPECS
    ]
    for ref in refs[2:end]
        @assert axes_match(refs[1], ref) "All panels must share the same κ and xM/L axes"
    end

    maps = Matrix{Any}(undef, length(ROW_SPECS), length(COL_SPECS))
    for (r, spec) in enumerate(ROW_SPECS), (c, col) in enumerate(COL_SPECS)
        if isnothing(col.num_modes)
            maps[r, c] = refs[r]
        else
            @info "Building ROM panel" row = r num_modes = col.num_modes
            maps[r, c] = rom_alpha_map(artifacts[spec.artifact], refs[r], spec;
                                        num_modes = col.num_modes)
        end
    end

    CairoMakie.with_theme(CairoMakie.theme_latexfonts(); fonts = (; regular = NEWCM_FONT)) do
        fig = CairoMakie.Figure(size = (940, 820), backgroundcolor = :white,
            fontsize = 20, figure_padding = (2, 1, 40, 2))
        last_hm = nothing
        panel_axes = Matrix{CairoMakie.Axis}(undef, length(ROW_SPECS), length(COL_SPECS))
        for c in eachindex(COL_SPECS)
            CairoMakie.Label(fig[1, c + 2], COL_SPECS[c].title,
                fontsize = 22, tellwidth = false)
        end
        for r in eachindex(ROW_SPECS)
            CairoMakie.Label(fig[r + 1, 1], ROW_SPECS[r].title, rotation = pi / 2,
                fontsize = 20, tellheight = false)
            for c in eachindex(COL_SPECS)
                show_top_ticks = r == 1 && c == 2
                ax = CairoMakie.Axis(fig[r + 1, c + 2],
                    xticklabelsize = 18, yticklabelsize = 18,
                    xlabelsize = 22, ylabelsize = 22,
                    xminorticksvisible = false, xminorgridvisible = false,
                    yminorgridvisible = false)
                panel_axes[r, c] = ax
                last_hm = draw_panel!(ax, maps[r, c];
                    show_xlabel = r == length(ROW_SPECS),
                    show_yticks = c == 1,
                    show_xticks = show_top_ticks,
                    xaxisposition = show_top_ticks ? :top : :bottom)
            end
        end
        for r in 1:size(panel_axes, 1), c in 1:size(panel_axes, 2)
            c > 1 && CairoMakie.linkyaxes!(panel_axes[r, 1], panel_axes[r, c])
        end
        for c in 1:size(panel_axes, 2), r in 1:size(panel_axes, 1)
            r > 1 && CairoMakie.linkxaxes!(panel_axes[1, c], panel_axes[r, c])
        end
        CairoMakie.Colorbar(fig[2:4, 6], last_hm, label = L"\alpha",
            labelsize = 22, ticklabelsize = 18, width = 18)
        CairoMakie.colgap!(fig.layout, 18)
        CairoMakie.rowgap!(fig.layout, 32)
        CairoMakie.colsize!(fig.layout, 1, CairoMakie.Fixed(40))
        CairoMakie.colsize!(fig.layout, 2, CairoMakie.Fixed(5))
        CairoMakie.colsize!(fig.layout, 6, CairoMakie.Fixed(42))
        CairoMakie.rowsize!(fig.layout, 1, CairoMakie.Fixed(34))

        out = joinpath(OUTPUT_DIR, "figures", "plot_fig10_modal_maps_3x3.pdf")
        mkpath(dirname(out))
        CairoMakie.save(out, fig)
        println("Saved $out")
    end
end

main()
