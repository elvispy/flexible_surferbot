# THIS SCRIPT GENERATES FIGURES FOR THE PAPER
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

const NUM_MODES   = 8
const RATIO_CUTOFF = 0.5

const RESONANCE_ALPHA_CUTOFF  = 0.04  # 10th-percentile of |α_LH| across xM < this → resonance column
const RESONANCE_N_PTS         = 20   # number of evenly-spaced xM points to emit per resonance column
const N_CURVE_LOGK            = 150  # logK resolution for theoretical contour grid
const N_CURVE_XM              = 150  # xM resolution for theoretical contour grid

const CURVE_NAMES  = ["S", "A", "eta_1", "eta_end"]

# Separate resonance stripe logK values (≥ RESONANCE_N_PTS points at same logK)
# from zero-crossing points, then trace connected branches via nearest-neighbour
# matching between consecutive logK columns.  Returns (lk_path, xm_path, res_lks)
# where lk_path/xm_path contain NaN separators between branches.
function split_xing_resonance(logK, xM)
    # --- detect resonance columns ---
    counts = Dict{Float64,Int}()
    for lk in logK; counts[lk] = get(counts, lk, 0) + 1; end
    res_lk_set = Set(lk for (lk, c) in counts if c >= RESONANCE_N_PTS)

    # --- collect zero-crossing points grouped by logK ---
    cols = Dict{Float64, Vector{Float64}}()
    for (lk, xm) in zip(logK, xM)
        lk ∈ res_lk_set && continue
        push!(get!(cols, lk, Float64[]), xm)
    end
    unique_lks = sort(collect(keys(cols)))
    for lk in unique_lks; sort!(cols[lk]); end

    if isempty(unique_lks)
        return Float64[], Float64[], sort(collect(res_lk_set))
    end

    # --- nearest-neighbour branch tracing ---
    # Each branch is a Vector{Tuple{Float64,Float64}} of (logK, xM) pairs.
    branches = [[(unique_lks[1], xm)] for xm in cols[unique_lks[1]]]

    for i in 2:length(unique_lks)
        lk      = unique_lks[i]
        new_xms = copy(cols[lk])
        matched = fill(false, length(new_xms))

        for branch in branches
            isempty(branch) && continue
            last_xm = last(branch)[2]
            isnan(last_xm) && continue        # branch already terminated
            best_j, best_d = 0, Inf
            for (j, xm) in enumerate(new_xms)
                matched[j] && continue
                d = abs(xm - last_xm)
                if d < best_d; best_d = d; best_j = j; end
            end
            if best_j > 0 && best_d < 0.15   # connect if close enough
                push!(branch, (lk, new_xms[best_j]))
                matched[best_j] = true
            else
                push!(branch, (NaN, NaN))     # branch ends / gap
            end
        end
        # start new branches for unmatched roots
        for (j, xm) in enumerate(new_xms)
            matched[j] || push!(branches, [(lk, xm)])
        end
    end

    # --- flatten branches with NaN separators ---
    lk_out, xm_out = Float64[], Float64[]
    for branch in branches
        append!(lk_out, first.(branch))
        append!(xm_out, last.(branch))
        push!(lk_out, NaN); push!(xm_out, NaN)
    end
    return lk_out, xm_out, sort(collect(res_lk_set))
end
const CURVE_LABELS = [L"|S| = 0", L"|A| = 0",
                      L"|\overline{\eta}(-\bar{\ell})| = 0",
                      L"|\overline{\eta}(\bar{\ell})| = 0"]

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

function theoretical_modal_context(params; output_dir::AbstractString, num_modes::Int=NUM_MODES)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams; output_dir=output_dir, num_modes_basis=num_modes)
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
function theoretical_modal_context_LH(params; output_dir::AbstractString, num_modes::Int=NUM_MODES)
    ctx = theoretical_modal_context(params; output_dir=output_dir, num_modes=num_modes)
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
            if quantile(abs.(alpha_col), 0.15) < RESONANCE_ALPHA_CUTOFF
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

# Compute |S|, |A|, |η₁|, |η_end| on a (N_CURVE_XM × N_CURVE_LOGK) grid.
# eta1_field / etae_field: symbol for the η₁ and η_end fields in the diagnostics NamedTuple.
function compute_diagnostic_grid(EI_grid, xM_grid, theory_ctx, endpoint_fn,
                                  eta1_field::Symbol, etae_field::Symbol)
    nEI = length(EI_grid);  nxM = length(xM_grid)
    S_mat  = zeros(nxM, nEI);  A_mat  = zeros(nxM, nEI)
    η1_mat = zeros(nxM, nEI);  ηe_mat = zeros(nxM, nEI)
    Threads.@threads for j in 1:nEI
        for i in 1:nxM
            q = solve_theoretical_modal_response(EI_grid[j], xM_grid[i], theory_ctx)
            d = endpoint_fn(q, theory_ctx)
            S_mat[i,j]  = abs(d.S);  A_mat[i,j]  = abs(d.A)
            η1_mat[i,j] = abs(getfield(d, eta1_field))
            ηe_mat[i,j] = abs(getfield(d, etae_field))
        end
    end
    return S_mat, A_mat, η1_mat, ηe_mat
end

# Contour level: matches RATIO_CUTOFF threshold used in find_filtered_minima.
# f = |S|/(|S|+|A|+ε) ∈ [0,1]; |S|/|A|=RATIO_CUTOFF ↔ f = RATIO_CUTOFF/(1+RATIO_CUTOFF)
const CONTOUR_LEVEL = RATIO_CUTOFF / (1.0 + RATIO_CUTOFF)

function add_diagnostic_contours!(p, lk_grid, xM_grid, S_mat, A_mat, η1_mat, ηe_mat,
                                   curve_labels, curve_colors, XLIMS, YLIMS)
    ε = eps()
    fS  = @. S_mat  / (S_mat  + A_mat  + ε)
    fA  = @. A_mat  / (A_mat  + S_mat  + ε)
    fη1 = @. η1_mat / (η1_mat + ηe_mat + ε)
    fηe = @. ηe_mat / (ηe_mat + η1_mat + ε)
    for (g, col) in zip([fS, fA, fη1, fηe], curve_colors)
        contour!(p, lk_grid, xM_grid, g;
                 levels   = [CONTOUR_LEVEL],
                 linecolor = col,
                 linewidth = 1.5,
                 colorbar  = false,
                 label     = false)
    end
    # Dummy series for legend (contour! label= is ignored by GR backend)
    for (lbl, col) in zip(curve_labels, curve_colors)
        plot!(p, [NaN], [NaN]; color=col, linewidth=1.5, label=lbl)
    end
end

function build_LH_plot(artifact, csv_path, output_dir; xlim_min::Float64)
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

    # Pad to xM/L = 0.5 by repeating the last row (CSV data ends at 0.48)
    if maximum(xM_axis) < 0.5
        xM_axis  = vcat(xM_axis, [0.5])
        alpha_LH = vcat(alpha_LH, alpha_LH[end:end, :])
    end

    XLIMS = (xlim_min, max_logK_data)
    YLIMS = (0.0, 0.5)

    # Coarse grid (57 pts): resonance detection only.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK_data; length=57))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    # Resonance-stripe detection via beam-end quantile (same logic as before,
    # now just collects logK values for vline! rather than injecting into results).
    beam_sync_ctx   = theoretical_modal_context(params; output_dir=output_dir)
    xM_sync         = collect(range(0.0, 0.49; length=401))
    sync_candidates = Dict{String, Vector{Tuple{Int,Float64}}}("S"=>[], "A"=>[])
    for (iei, EI) in enumerate(EI_scatter)
        scatter_logK[iei] < xlim_min && continue
        absS=Float64[]; absA=Float64[]; abs1=Float64[]; abse=Float64[]
        for xM in xM_sync
            q = solve_theoretical_modal_response(EI, xM, beam_sync_ctx)
            d = theoretical_endpoint_diagnostics_beam(q, beam_sync_ctx)
            push!(absS, abs(d.S)); push!(absA, abs(d.A))
            push!(abs1, abs(d.eta_beam_1)); push!(abse, abs(d.eta_beam_end))
        end
        alpha_col   = @. -(abs1^2 - abse^2) / (abs1^2 + abse^2 + eps())
        is_odd      = mean(absS) < mean(absA)
        resonance_q = is_odd ? 0.20 : 0.15
        q_val       = quantile(abs.(alpha_col), resonance_q)
        q_val < RESONANCE_ALPHA_CUTOFF || continue
        cond = is_odd ? "S" : "A"
        push!(sync_candidates[cond], (iei, q_val))
    end
    resonance_lk_LH = Dict{String, Vector{Float64}}("S"=>[], "A"=>[])
    for cond in ("S", "A")
        cands = sort(sync_candidates[cond]; by = x -> x[1])
        isempty(cands) && continue
        groups = Vector{Vector{Tuple{Int,Float64}}}()
        for cand in cands
            if isempty(groups) || cand[1] - groups[end][end][1] > 1
                push!(groups, [cand])
            else
                push!(groups[end], cand)
            end
        end
        for group in groups
            _, best_pos = findmin(x -> x[2], group)
            push!(resonance_lk_LH[cond], scatter_logK[group[best_pos][1]])
        end
    end

    # Contour grid for theoretical curves.
    @info "Computing LH diagnostic contour grid ($(N_CURVE_LOGK)×$(N_CURVE_XM))…"
    ctx_LH      = theoretical_modal_context_LH(params; output_dir=output_dir)
    curve_logK  = collect(range(xlim_min, max_logK_data; length=N_CURVE_LOGK))
    curve_xM    = collect(range(YLIMS[1], YLIMS[2]; length=N_CURVE_XM))
    EI_curve    = 10 .^ (curve_logK .+ shift)
    S_mat, A_mat, η1_mat, ηe_mat = compute_diagnostic_grid(
        EI_curve, curve_xM, ctx_LH, theoretical_endpoint_diagnostics_LH,
        :eta_LH_1, :eta_LH_end)

    # Build plot
    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
    markers      = [:circle, :rect, :diamond, :utriangle]

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
        dpi     = 220,
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

    add_diagnostic_contours!(p, curve_logK, curve_xM, S_mat, A_mat, η1_mat, ηe_mat,
                              CURVE_LABELS, curve_colors, XLIMS, YLIMS)

    # Resonance stripes from beam-end sync
    for (ci, cname) in enumerate(("S", "A"))
        col = curve_colors[ci]
        for rlk in resonance_lk_LH[cname]
            (XLIMS[1] <= rlk <= XLIMS[2]) || continue
            vline!(p, [rlk]; color=col, linewidth=1.5, linestyle=:solid, label=false)
        end
    end

    logκ_surferbot = log10(Float64(params.EI)) - shift
    vline!(p, [logκ_surferbot];
           color     = RGB(0.95, 0.75, 0.05), linewidth = 2.0,
           linestyle = :dash, label = "Surferbot")

    return p
end

# ─── Beam-end theoretical diagnostics ────────────────────────────────────────
#
# Mirrors theoretical_endpoint_diagnostics_LH but evaluates at the beam ends
# (x = ±L/2) using w_end/w_start from the Ψ basis rather than a_vec/a_vec_left.
# Works for d=0 (uncoupled) because zero_modal_pressure_map still computes Ψ.

function theoretical_endpoint_diagnostics_beam(q, theory_ctx)
    S = zero(ComplexF64)
    A = zero(ComplexF64)
    for j in eachindex(theory_ctx.mode_numbers)
        if iseven(theory_ctx.mode_numbers[j])
            S += q[j] * theory_ctx.w_end[j]
        else
            A += q[j] * theory_ctx.w_end[j]
        end
    end
    eta_beam_end = sum(q[j] * theory_ctx.w_end[j]   for j in eachindex(q))
    eta_beam_1   = sum(q[j] * theory_ctx.w_start[j] for j in eachindex(q))
    return (; S, A, eta_beam_1, eta_beam_end)
end

function get_roots_theoretical_beam(EI_list::AbstractVector{Float64}, condition_name,
                                     theory_ctx)
    logEI_axis = log10.(EI_list)
    xM_grid    = collect(range(0.0, 0.49; length=401))

    pts_logEI      = Float64[]
    pts_xM         = Float64[]
    res_logEI      = Float64[]  # resonance stripe logEI values (not zero-crossing pts)
    res_candidates = Tuple{Int, Float64}[]  # (iei, q_val) for resonance stripe candidates

    for (iei, EI) in enumerate(EI_list)
        absS = Float64[]; absA = Float64[]
        abs_eta_1 = Float64[]; abs_eta_end = Float64[]

        for xM_norm in xM_grid
            q    = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
            diag = theoretical_endpoint_diagnostics_beam(q, theory_ctx)
            push!(absS,       abs(diag.S))
            push!(absA,       abs(diag.A))
            push!(abs_eta_1,  abs(diag.eta_beam_1))
            push!(abs_eta_end, abs(diag.eta_beam_end))
        end

        roots = roots_for_condition(condition_name, xM_grid,
                                    absS, absA, abs_eta_1, abs_eta_end)
        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM,    r)
        end

        # Resonance candidate collection (deduplication happens after the loop).
        # S-type resonances (odd, A-dominates) use q=0.20; A-type use q=0.15 (coupled)
        # or q=0.10 (uncoupled, where resonances are sharper with no radiation damping).
        if condition_name in ("S", "A")
            alpha_col = @. -(abs_eta_1^2 - abs_eta_end^2) /
                            (abs_eta_1^2 + abs_eta_end^2 + eps())
            is_odd_resonance = mean(absS) < mean(absA)
            is_coupled       = Float64(theory_ctx.params.d) > 0.0
            resonance_q = is_odd_resonance ? 0.20 : (is_coupled ? 0.15 : 0.10)
            q_val = quantile(abs.(alpha_col), resonance_q)
            if q_val < RESONANCE_ALPHA_CUTOFF
                if (condition_name == "S" && is_odd_resonance) ||
                   (condition_name == "A" && !is_odd_resonance)
                    push!(res_candidates, (iei, q_val))
                end
            end
        end
    end

    # Deduplicate resonance candidates: group runs of consecutive scatter-grid indices
    # and emit exactly one stripe per run (at the minimum-quantile index in the run).
    # This prevents adjacent scatter points that both pass the threshold for the same
    # physical resonance from generating two stacked vertical stripes.
    if !isempty(res_candidates)
        sort!(res_candidates; by = x -> x[1])
        groups = Vector{Vector{Tuple{Int,Float64}}}()
        for cand in res_candidates
            if isempty(groups) || cand[1] - groups[end][end][1] > 1
                push!(groups, [cand])
            else
                push!(groups[end], cand)
            end
        end
        res_xM_template = collect(range(xM_grid[1], xM_grid[end]; length=RESONANCE_N_PTS))
        for group in groups
            _, best_pos = findmin(x -> x[2], group)
            best_iei = group[best_pos][1]
            best_le  = logEI_axis[best_iei]
            append!(pts_logEI, fill(best_le, RESONANCE_N_PTS))
            append!(pts_xM,    res_xM_template)
            push!(res_logEI, best_le)
        end
    end

    return (; logEI=pts_logEI, xM_norm=pts_xM, resonance_logEI=res_logEI)
end

const BEAM_CURVE_LABELS = [L"|S| = 0", L"|A| = 0",
                            L"|\overline{\eta}(-1/2)| = 0",
                            L"|\overline{\eta}(+1/2)| = 0"]

function build_beam_end_plot(artifact, csv_path, output_dir; xlim_min::Float64)
    params = artifact.base_params
    shift  = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 *
                   Float64(params.omega)^2)

    df_heat       = CSV.read(csv_path, DataFrame)
    all_logEI     = sort(unique(df_heat.log10_EI))
    logEI_axis    = all_logEI[all_logEI .- shift .>= xlim_min]
    xM_axis       = sort(unique(df_heat.xM_over_L))
    max_logK_data = maximum(logEI_axis) - shift

    alpha_beam = zeros(Float64, length(xM_axis), length(logEI_axis))
    let lookup = Dict{Tuple{Float64,Float64}, Float64}(
            (row.log10_EI, row.xM_over_L) =>
                Surferbot.Analysis.beam_asymmetry(
                    complex(row.eta_1_beam_re,  row.eta_1_beam_im),
                    complex(row.eta_end_beam_re, row.eta_end_beam_im))
            for row in eachrow(df_heat))
        for (j, le) in enumerate(logEI_axis), (i, xm) in enumerate(xM_axis)
            alpha_beam[i, j] = lookup[(le, xm)]
        end
    end

    # Pad to xM/L = 0.5 by repeating the last row (CSV data ends at 0.48)
    if maximum(xM_axis) < 0.5
        xM_axis   = vcat(xM_axis, [0.5])
        alpha_beam = vcat(alpha_beam, alpha_beam[end:end, :])
    end

    XLIMS = (xlim_min, max_logK_data)
    YLIMS = (0.0, 0.5)

    # Coarse grid (57 pts): resonance detection only.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK_data; length=57))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    theory_ctx = theoretical_modal_context(params; output_dir=output_dir)

    resonance_logK = Dict{String, Vector{Float64}}(cname => Float64[] for cname in CURVE_NAMES)
    for cname in CURVE_NAMES
        @info "Computing beam-end resonance detection: $cname"
        res_coarse = get_roots_theoretical_beam(EI_scatter, cname, theory_ctx)
        resonance_logK[cname] = res_coarse.resonance_logEI .- shift
    end

    # CSV-based resonance seeding for uncoupled (d=0) only.
    if Float64(theory_ctx.params.d) == 0.0
        firing_logK = Float64[]
        for (j, le) in enumerate(logEI_axis)
            quantile(abs.(alpha_beam[:, j]), 0.10) < RESONANCE_ALPHA_CUTOFF || continue
            push!(firing_logK, le - shift)
        end
        if !isempty(firing_logK)
            clusters = Vector{Vector{Float64}}()
            for lk in sort(firing_logK)
                if isempty(clusters) || lk - clusters[end][end] > 0.05
                    push!(clusters, [lk])
                else
                    push!(clusters[end], lk)
                end
            end
            xM_mid = Float64(xM_axis[max(1, length(xM_axis) ÷ 2)])
            for cluster in clusters
                logK_val = sum(cluster) / length(cluster)
                j_near = argmin(abs.(logEI_axis .- (logK_val + shift)))
                alpha_col = abs.(alpha_beam[:, j_near])
                q_chk = solve_theoretical_modal_response(10^(logK_val + shift), xM_mid, theory_ctx)
                d_chk = theoretical_endpoint_diagnostics_beam(q_chk, theory_ctx)
                is_odd_csv = abs(d_chk.S) < abs(d_chk.A)
                cond = is_odd_csv ? "S" : "A"
                resonance_q_csv = is_odd_csv ? 0.20 : 0.10
                quantile(alpha_col, resonance_q_csv) < RESONANCE_ALPHA_CUTOFF || continue
                any(abs.(resonance_logK[cond] .- logK_val) .< 0.05) && continue
                push!(resonance_logK[cond], logK_val)
                @info "CSV-seeded resonance stripe: $cond at logK=$(round(logK_val; digits=3))"
            end
        end
    end

    # Contour grid for theoretical curves.
    @info "Computing beam-end diagnostic contour grid ($(N_CURVE_LOGK)×$(N_CURVE_XM))…"
    curve_logK = collect(range(xlim_min, max_logK_data; length=N_CURVE_LOGK))
    curve_xM   = collect(range(YLIMS[1], YLIMS[2]; length=N_CURVE_XM))
    EI_curve   = 10 .^ (curve_logK .+ shift)
    S_mat, A_mat, η1_mat, ηe_mat = compute_diagnostic_grid(
        EI_curve, curve_xM, theory_ctx, theoretical_endpoint_diagnostics_beam,
        :eta_beam_1, :eta_beam_end)

    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
    markers      = [:circle, :rect, :diamond, :utriangle]

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
        dpi     = 220,
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

    p = heatmap(logEI_axis .- shift, xM_axis, alpha_beam; plt_opts...)

    add_diagnostic_contours!(p, curve_logK, curve_xM, S_mat, A_mat, η1_mat, ηe_mat,
                              BEAM_CURVE_LABELS, curve_colors, XLIMS, YLIMS)

    for (ci, cname) in enumerate(CURVE_NAMES[1:2])  # S and A only have resonance stripes
        col = curve_colors[ci]
        for rlk in resonance_logK[cname]
            (XLIMS[1] <= rlk <= XLIMS[2]) || continue
            vline!(p, [rlk]; color=col, linewidth=1.5, linestyle=:solid, label=false)
        end
    end

    logκ_surferbot = log10(Float64(theory_ctx.params.EI)) - shift
    vline!(p, [logκ_surferbot];
           color     = RGB(0.95, 0.75, 0.05), linewidth = 2.0,
           linestyle = :dash, label = "Surferbot")

    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    # ── 1. LH (domain-end) coupled plot ──────────────────────────────────────
    art_cpl = load_sweep(joinpath(output_dir, "jld2",
                                  "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    theory_ctx_cpl = theoretical_modal_context_LH(art_cpl.base_params; output_dir=output_dir)
    print_symmetry_check(theory_ctx_cpl)

    p_cpl_LH = build_LH_plot(art_cpl,
                               joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                               output_dir; xlim_min=-4.0)
    out_cpl_LH = joinpath(fig_dir, "plot_dimensionless_diagnostics_cpl_theo_LH.pdf")
    savefig(p_cpl_LH, out_cpl_LH)
    println("Saved $out_cpl_LH")

    # ── 2. Beam-end coupled plot ──────────────────────────────────────────────
    p_cpl_beam = build_beam_end_plot(art_cpl,
                                      joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                                      output_dir; xlim_min=-4.0)
    out_cpl_beam = joinpath(fig_dir, "plot_dimensionless_diagnostics_cpl_beam.pdf")
    savefig(p_cpl_beam, out_cpl_beam)
    println("Saved $out_cpl_beam")

    # ── 3. Beam-end uncoupled plot (x axis −5 to max) ────────────────────────
    art_ucpl = load_sweep(joinpath(output_dir, "jld2",
                                   "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))

    p_ucpl_beam = build_beam_end_plot(art_ucpl,
                                       joinpath(output_dir, "csv", "sweeper_uncoupled_full_grid.csv"),
                                       output_dir; xlim_min=-5.0)
    out_ucpl_beam = joinpath(fig_dir, "plot_dimensionless_diagnostics_ucpl_beam.pdf")
    savefig(p_ucpl_beam, out_ucpl_beam)
    println("Saved $out_ucpl_beam")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
