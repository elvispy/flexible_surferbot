"""
plot_dimensionless_diagnostics_LH.jl

Longuet-Higgins (domain-end) counterpart of plot_dimensionless_diagnostics.jl.

Instead of evaluating η at the beam ends (x = ±L/2), this script evaluates η at
the ends of the computational domain (x = ±ℓ), which is the physically relevant
quantity for the radiation-stress / thrust diagnostic.

Theory:
  The same modal system (D − Z_ψ) q = −f gives the Ψ-basis modal amplitudes q.
  Domain-end elevations follow by linear superposition:

      η_LH_end = a_vec       · q      (right domain end, x = +ℓ)
      η_LH_1   = a_vec_left  · q      (left  domain end, x = −ℓ)

  where a_vec[n] = η(+ℓ) and a_vec_left[n] = η(−ℓ) from the radiation solve
  with Ψ_n prescribed on the raft (stored in the modal_pressure_maps.jld2 cache).

  Symmetry check: for even modes expect a_vec_left ≈ +a_vec,
                  for odd  modes expect a_vec_left ≈ −a_vec.

α_LH = −(|η_LH_1|² − |η_LH_end|²) / (|η_LH_1|² + |η_LH_end|²)

Heatmap: α_LH read directly from CSV domain-end columns (no new sweep needed).
Scatter: theoretical prediction via the a-priori modal law.

Output: output/figures/plot_dimensionless_diagnostics_cpl_theo_LH.pdf
"""

using Surferbot, JLD2, Plots, LaTeXStrings, Printf, LinearAlgebra, CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "experiments", "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

const NUM_MODES   = 6
const RATIO_CUTOFF = 0.5

const RESONANCE_ALPHA_CUTOFF  = 0.04  # 10th-percentile of |α_LH| across xM < this → resonance column
const RESONANCE_N_PTS         = 20   # number of evenly-spaced xM points to emit per resonance column

const CURVE_NAMES  = ["S", "A", "eta_1", "eta_end"]
const CURVE_LABELS = [L"|S| = 0", L"|A| = 0",
                      L"|\hat{\eta}(-\ell)| = 0",
                      L"|\hat{\eta}(\ell)| = 0"]

# ─── Helpers (parallel to plot_dimensionless_diagnostics.jl) ─────────────────

function coerce_flexible_params(params)
    params isa Surferbot.FlexibleParams && return params
    pairs = Pair{Symbol,Any}[]
    for k in fieldnames(Surferbot.FlexibleParams)
        hasproperty(params, k) && push!(pairs, k => getproperty(params, k))
    end
    return Surferbot.FlexibleParams(; pairs...)
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

function roots_for_condition(condition_name, xgrid, absS, absA, abs_eta_1, abs_eta_end)
    if condition_name == "S"
        ratio = absS ./ max.(absA, eps())
        return find_filtered_minima(xgrid, absS, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "A"
        ratio = absA ./ max.(absS, eps())
        return find_filtered_minima(xgrid, absA, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "eta_1"
        denom = abs_eta_1 .+ abs_eta_end .+ eps()
        return find_filtered_minima(xgrid, abs_eta_1, abs_eta_1 ./ denom; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "eta_end"
        denom = abs_eta_1 .+ abs_eta_end .+ eps()
        return find_filtered_minima(xgrid, abs_eta_end, abs_eta_end ./ denom; ratio_cutoff=RATIO_CUTOFF)
    end
    return Float64[]
end

# ─── Modal context (beam-end version kept for comparison / backwards compat) ──

function theoretical_modal_context(params; output_dir::AbstractString)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams; output_dir=output_dir, num_modes_basis=NUM_MODES)
    derived = Surferbot.derive_params(fparams)
    Psi = payload.psi_basis.Psi
    return (
        params       = fparams,
        derived      = derived,
        payload      = payload,
        mode_numbers = collect(Int.(payload.mode_labels)),
        Psi          = Matrix{Float64}(Psi),
        x_raft       = collect(Float64.(payload.x_raft)),
        weights      = collect(Float64.(payload.weights)),
        w_end        = Psi[end, :],
        w_start      = Psi[1,   :],
        beta         = collect(Float64.(payload.beta)),
        Z_psi        = ComplexF64.(payload.Z_psi),
        c_hydro      = derived.d * fparams.rho * fparams.g,
        F0           = fparams.motor_inertia * fparams.omega^2,
        forcing_width = fparams.forcing_width,
    )
end

# LH context: same as beam-end but adds a_vec (right) and a_vec_left (left)
function theoretical_modal_context_LH(params; output_dir::AbstractString)
    ctx = theoretical_modal_context(params; output_dir=output_dir)
    return merge(ctx, (
        a_vec      = ComplexF64.(ctx.payload.a_vec),
        a_vec_left = ComplexF64.(ctx.payload.a_vec_left),
    ))
end

# ─── Modal solve (unchanged from beam-end version) ────────────────────────────

function solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
    p   = theory_ctx.params
    F_c = theory_ctx.derived.F_c
    L_c = theory_ctx.derived.L_c

    x_raft_adim = theory_ctx.x_raft ./ L_c
    loads_adim  = (theory_ctx.F0 / F_c) .*
                  Surferbot.gaussian_load(Float64(xM_norm), p.forcing_width, x_raft_adim)
    loads_dim   = loads_adim .* (F_c / L_c)
    F_psi       = theory_ctx.Psi' * (loads_dim .* theory_ctx.weights)

    D     = ComplexF64.(EI .* theory_ctx.beta .^ 4
                        .- p.rho_raft * p.omega^2
                        .+ theory_ctx.c_hydro)
    A_sys = Diagonal(D) - theory_ctx.Z_psi
    return -(A_sys \ ComplexF64.(F_psi))
end

# ─── LH endpoint diagnostics ─────────────────────────────────────────────────

# Compute S_LH, A_LH, η_LH_end, η_LH_1 from Ψ-basis modal amplitudes q.
#
# S_LH and A_LH use a_vec (right-end radiation amplitudes) with the even/odd
# split — this retains the same α = 0 sub-condition structure as the beam-end
# version and lets us test the symmetry hypothesis:
#   even modes → a_vec_left ≈ +a_vec  (symmetric radiation)
#   odd  modes → a_vec_left ≈ −a_vec  (antisymmetric radiation)
#
# η_LH_end and η_LH_1 are computed DIRECTLY from a_vec and a_vec_left so we
# do not assume the symmetry — we measure it.
function theoretical_endpoint_diagnostics_LH(q, theory_ctx)
    S = zero(ComplexF64)
    A = zero(ComplexF64)
    for j in eachindex(theory_ctx.mode_numbers)
        if iseven(theory_ctx.mode_numbers[j])
            S += q[j] * theory_ctx.a_vec[j]
        else
            A += q[j] * theory_ctx.a_vec[j]
        end
    end
    # Direct evaluation — no symmetry assumption
    eta_LH_end = sum(q[j] * theory_ctx.a_vec[j]      for j in eachindex(q))
    eta_LH_1   = sum(q[j] * theory_ctx.a_vec_left[j] for j in eachindex(q))
    return (; S, A, eta_LH_1, eta_LH_end)
end

# ─── Root extraction (LH version) ────────────────────────────────────────────

function get_roots_theoretical_LH(artifact, condition_name; output_dir::AbstractString,
                                   EI_list::Union{Nothing,AbstractVector{Float64}}=nothing)
    params     = artifact.base_params
    EI_list    = EI_list === nothing ? collect(Float64.(artifact.parameter_axes.EI)) : EI_list
    logEI_axis = log10.(EI_list)
    xM_grid    = collect(range(0.0, 0.49; length=401))
    theory_ctx = theoretical_modal_context_LH(params; output_dir=output_dir)

    pts_logEI = Float64[]
    pts_xM    = Float64[]

    for (iei, EI) in enumerate(EI_list)
        absS = Float64[]; absA = Float64[]
        abs_eta_1 = Float64[]; abs_eta_end = Float64[]

        for xM_norm in xM_grid
            q    = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
            diag = theoretical_endpoint_diagnostics_LH(q, theory_ctx)
            push!(absS,      abs(diag.S))
            push!(absA,      abs(diag.A))
            push!(abs_eta_1, abs(diag.eta_LH_1))
            push!(abs_eta_end, abs(diag.eta_LH_end))
        end

        roots = roots_for_condition(condition_name, xM_grid,
                                    absS, absA, abs_eta_1, abs_eta_end)
        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM,    r)
        end

        # Resonance pass: α_LH ≈ 0 for ALL xM → whole column is a resonance.
        # Check directly via the domain-end amplitudes (already computed above).
        # At a resonance, only even OR odd modes are driven, so S ≈ 0 (odd resonance)
        # or A ≈ 0 (even resonance). Emit the full vertical stripe on the correct series.
        if condition_name in ("S", "A")
            alpha_col = @. -(abs_eta_1^2 - abs_eta_end^2) /
                            (abs_eta_1^2 + abs_eta_end^2 + eps())
            if quantile(abs.(alpha_col), 0.1) < RESONANCE_ALPHA_CUTOFF
                # Determine parity: odd resonance → S ≈ 0; even resonance → A ≈ 0
                is_odd_resonance = mean(absS) < mean(absA)
                if (condition_name == "S" && is_odd_resonance) ||
                   (condition_name == "A" && !is_odd_resonance)
                    res_xM = collect(range(xM_grid[1], xM_grid[end]; length=RESONANCE_N_PTS))
                    append!(pts_logEI, fill(logEI_axis[iei], RESONANCE_N_PTS))
                    append!(pts_xM,    res_xM)
                end
            end
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

# ─── Symmetry diagnostic ─────────────────────────────────────────────────────

function print_symmetry_check(theory_ctx)
    println("\nSymmetry check: a_vec_left vs ±a_vec (Ψ basis)")
    println("mode  parity  |a_right|     |a_left|   |a_left - (+)a_right|  |a_left - (-)a_right|")
    for (j, n) in enumerate(theory_ctx.mode_numbers)
        ar = theory_ctx.a_vec[j]
        al = theory_ctx.a_vec_left[j]
        parity = iseven(n) ? "even" : "odd"
        @printf("  %2d   %-4s  %.4e   %.4e       %.4e            %.4e\n",
                n, parity, abs(ar), abs(al), abs(al - ar), abs(al + ar))
    end
    println()
end

# ─── Shared plot builder ──────────────────────────────────────────────────────
#
# scatter_EI_list: explicit EI values for the theoretical scatter.
#   Pass nothing  → use artifact.parameter_axes.EI (original coupled behaviour).
#   Pass a vector → use that instead (needed for uncoupled where the artifact
#                   EI range is narrower than the desired x axis).

function build_LH_plot(artifact, csv_path, output_dir;
                       xlim_min::Float64, fig_title::LaTeXString,
                       scatter_EI_list=nothing)
    params = artifact.base_params
    shift  = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                   Float64(params.omega)^2)

    # Heatmap from CSV — keep only columns whose logκ falls within [xlim_min, ∞)
    # so that no data sits outside the left xlim (Plots.jl misrenders otherwise).
    df_heat       = CSV.read(csv_path, DataFrame)
    all_logEI     = sort(unique(df_heat.log10_EI))
    logEI_axis    = all_logEI[all_logEI .- shift .>= xlim_min]
    xM_axis       = sort(unique(df_heat.xM_over_L))
    max_logK_data = maximum(logEI_axis) - shift

    alpha_LH = zeros(Float64, length(xM_axis), length(logEI_axis))
    let lookup = Dict{Tuple{Float64,Float64}, Float64}(
            (row.log10_EI, row.xM_over_L) =>
                Surferbot.Analysis.beam_asymmetry(
                    complex(row.eta_1_domain_re,  row.eta_1_domain_im),
                    complex(row.eta_end_domain_re, row.eta_end_domain_im))
            for row in eachrow(df_heat))
        for (j, le) in enumerate(logEI_axis), (i, xm) in enumerate(xM_axis)
            alpha_LH[i, j] = lookup[(le, xm)]
        end
    end

    XLIMS = (xlim_min, max_logK_data)
    YLIMS = (0.0, 0.5)

    # Theoretical scatter EI grid
    EI_scatter = scatter_EI_list === nothing ?
                     collect(Float64.(artifact.parameter_axes.EI)) :
                     scatter_EI_list

    results = Dict{String, NamedTuple}()
    for cname in CURVE_NAMES
        @info "Computing LH roots: $cname"
        res = get_roots_theoretical_LH(artifact, cname;
                                        output_dir=output_dir,
                                        EI_list=EI_scatter)
        results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
    end

    # Build plot
    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
    markers      = [:circle, :rect, :diamond, :utriangle]

    plt_opts = (
        title   = fig_title,
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
        dpi     = 220,
        titlefontsize     = 14,
        guidefontsize     = 14,
        tickfontsize      = 12,
        legendfontsize    = 11,
        fontfamily        = "Computer Modern",
        framestyle        = :box,
        grid              = false,
        colorbar_title    = L"\alpha_{\mathrm{LH}}",
        colorbar_titlefontsize = 14,
        colorbar_tickfontsize  = 11,
    )

    p = heatmap(logEI_axis .- shift, xM_axis, alpha_LH; plt_opts...)

    for (i, cname) in enumerate(CURVE_NAMES)
        res  = results[cname]
        mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
               (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
        isempty(res.logK[mask]) && continue
        scatter!(p, res.logK[mask], res.xM_norm[mask];
                 label             = CURVE_LABELS[i],
                 color             = curve_colors[i],
                 marker            = markers[i],
                 markersize        = 5,
                 markerstrokewidth = 0.6,
                 markerstrokecolor = :white,
                 markeralpha       = 0.95)
    end

    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    # ── Coupled figure (unchanged behaviour) ─────────────────────────────────
    art_cpl = load_sweep(joinpath(output_dir, "jld2",
                                  "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    theory_ctx_cpl = theoretical_modal_context_LH(art_cpl.base_params; output_dir=output_dir)
    print_symmetry_check(theory_ctx_cpl)

    Lambda_val = @sprintf("%.2f", Float64(art_cpl.base_params.d) /
                                  Float64(art_cpl.base_params.L_raft))
    title_cpl  = LaTeXString("Coupled raft, \$\\Lambda = $Lambda_val\$ — LH domain-end prediction")

    p_cpl = build_LH_plot(art_cpl,
                           joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                           output_dir;
                           xlim_min = -4.0,
                           fig_title = title_cpl)

    out_cpl = joinpath(fig_dir, "plot_dimensionless_diagnostics_cpl_theo_LH.pdf")
    savefig(p_cpl, out_cpl)
    println("Saved $out_cpl")

    # ── Uncoupled figure (x axis to −5, scatter covers full range) ───────────
    art_ucpl = load_sweep(joinpath(output_dir, "jld2",
                                   "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))
    title_ucpl = LaTeXString("Uncoupled raft (\$\\Lambda = 0\$) — LH domain-end prediction")

    # Derive EI scatter list from CSV so theory scatter covers the full x axis [-5, max_logK]
    df_ucpl = CSV.read(joinpath(output_dir, "csv", "sweeper_uncoupled_full_grid.csv"), DataFrame)
    ucpl_shift      = log10(Float64(art_ucpl.base_params.rho_raft) *
                            Float64(art_ucpl.base_params.L_raft)^4 *
                            Float64(art_ucpl.base_params.omega)^2)
    ucpl_all_logEI  = sort(unique(df_ucpl.log10_EI))
    ucpl_EI_scatter = 10 .^ ucpl_all_logEI[ucpl_all_logEI .- ucpl_shift .>= -5.0]

    p_ucpl = build_LH_plot(art_ucpl,
                            joinpath(output_dir, "csv", "sweeper_uncoupled_full_grid.csv"),
                            output_dir;
                            xlim_min        = -5.0,
                            fig_title       = title_ucpl,
                            scatter_EI_list = ucpl_EI_scatter)

    out_ucpl = joinpath(fig_dir, "plot_dimensionless_diagnostics_uncpl_theo_LH.pdf")
    savefig(p_ucpl, out_ucpl)
    println("Saved $out_ucpl")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
