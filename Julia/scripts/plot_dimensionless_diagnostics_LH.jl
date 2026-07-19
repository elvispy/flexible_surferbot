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

using Surferbot, JLD2, CairoMakie, LaTeXStrings, Printf, LinearAlgebra, CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "paper_plot_theme.jl"))
using .PaperPlotTheme

# κ axis: integer log10(κ) grid, ticks shown as 10^n (exponentials, not log₁₀κ)
function kappa_exp_xticks(xlims)
    lo, hi = ceil(Int, xlims[1]), floor(Int, xlims[2])
    ticks  = collect(lo:hi)
    return (Float64.(ticks), [latexstring(@sprintf("10^{%d}", t)) for t in ticks])
end

include(joinpath(@__DIR__, "..", "experiments", "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

const NUM_MODES    = 8
const RATIO_CUTOFF = 0.5
# The cached modal map has approximately 1e-7 opposite-parity leakage at
# centered forcing, so phase is not numerically meaningful below this scale.
const PHASE_AMPLITUDE_RELATIVE_FLOOR = 1e-6

# Branch-terminus detection: when a tracked high-xM root disappears between
# consecutive EI steps and the function is still decreasing at xM=0.5, we add
# one terminal scatter point at xM=0.5 for that EI.
const TERM_MIN_XM = -0.35  # only track roots below this xM as upward (raft-end) branches
const TERM_TOL    = 0.08   # max xM gap to consider a root "continued"

const RESONANCE_ALPHA_CUTOFF  = 0.04  # 10th-percentile of |α_LH| across xM < this → resonance column
const RESONANCE_N_PTS         = 20   # number of evenly-spaced xM points to emit per resonance column
# A low alpha-quantile alone is not sufficient: in the smooth high-κ asymptotic
# regime it can dip below RESONANCE_ALPHA_CUTOFF without any actual near-singular
# amplification (verified: |q| there is *below* the typical baseline, not above
# it). A genuine resonance amplifies the dominant-parity response; require the
# column's mean(max(|S|,|A|)) to exceed this multiple of the across-grid median
# to be accepted as a real resonance. Confirmed resonances measure 6-32x this
# baseline; the false-positive case measured 0.28x — wide margin either way.
const RESONANCE_AMP_FACTOR    = 2.0

const CURVE_NAMES  = ["S", "A", "eta_1", "eta_end"]

# ─── Branch clustering for line plots ────────────────────────────────────────
#
# Zero-crossing points at each logK column form non-intersecting branches in
# (logK, xM) space.  This function:
#   1. Detects resonance stripe logK values (≥ RESONANCE_N_PTS points at same logK)
#      and returns them separately for vline! rendering.
#   2. Groups the remaining zero-crossing points into connected branches via
#      nearest-neighbour matching between consecutive logK columns.
#   3. Returns each branch as a (lk, xm) pair of sorted vectors, NaN-separated
#      into a single flat path for a single plot! call.
# cluster_branches is provided by Surferbot (Julia/src/modal.jl).
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

# find_filtered_minima is provided by Surferbot (Julia/src/modal.jl) — interpolates
# a sub-grid root location instead of snapping to the xgrid point, avoiding a
# staircase artifact wherever the branch-tracing loop below samples logK densely.

function phase_orthogonality_residual(S::Number, A::Number;
                                       amplitude_floor_S::Real=0.0,
                                       amplitude_floor_A::Real=0.0)
    absS = abs(S)
    absA = abs(A)
    (absS <= amplitude_floor_S || absA <= amplitude_floor_A) && return NaN
    return clamp(real(S * conj(A)) / (absS * absA), -1.0, 1.0)
end

function phase_orthogonality_field(S::AbstractMatrix, A::AbstractMatrix;
                                    relative_floor::Real=PHASE_AMPLITUDE_RELATIVE_FLOOR)
    size(S) == size(A) || throw(DimensionMismatch("S and A fields must have the same size"))
    residual = fill(NaN, size(S))
    cross_term = S .* conj.(A)
    for j in axes(S, 2)
        floorS = relative_floor * maximum(abs, view(S, :, j))
        floorA = relative_floor * maximum(abs, view(A, :, j))
        for i in axes(S, 1)
            residual[i, j] = phase_orthogonality_residual(
                S[i, j], A[i, j];
                amplitude_floor_S=floorS,
                amplitude_floor_A=floorA,
            )
        end
    end

    # A sampled passage through S=0 or A=0 flips the phase by pi. If the zero
    # lies between grid points, contour interpolation otherwise mistakes that
    # discontinuity for cos(arg(S)-arg(A)) = 0. At such a passage both real and
    # imaginary parts of S*conj(A) reverse sign; at true quadrature only its real
    # part crosses zero while the imaginary part remains nonzero with one sign.
    invalid = falses(size(residual))
    opposite_sign(a, b) = a != 0 && b != 0 && signbit(a) != signbit(b)
    if size(cross_term, 1) > 1
        for j in axes(cross_term, 2), i in firstindex(cross_term, 1):(lastindex(cross_term, 1) - 1)
            z1, z2 = cross_term[i, j], cross_term[i + 1, j]
            if opposite_sign(real(z1), real(z2)) && opposite_sign(imag(z1), imag(z2))
                invalid[i, j] = true
                invalid[i + 1, j] = true
            end
        end
    end
    if size(cross_term, 2) > 1
        for j in firstindex(cross_term, 2):(lastindex(cross_term, 2) - 1), i in axes(cross_term, 1)
            z1, z2 = cross_term[i, j], cross_term[i, j + 1]
            if opposite_sign(real(z1), real(z2)) && opposite_sign(imag(z1), imag(z2))
                invalid[i, j] = true
                invalid[i, j + 1] = true
            end
        end
    end
    residual[invalid] .= NaN
    return residual
end

function theoretical_phase_orthogonality_field(EI_list, xM_grid, theory_ctx)
    S_field = Matrix{ComplexF64}(undef, length(xM_grid), length(EI_list))
    A_field = similar(S_field)
    for (j, EI) in enumerate(EI_list), (i, xM_norm) in enumerate(xM_grid)
        q = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
        diag = theoretical_endpoint_diagnostics_LH(q, theory_ctx)
        S_field[i, j] = diag.S
        A_field[i, j] = diag.A
    end
    return phase_orthogonality_field(S_field, A_field)
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

# Return the condition-specific value and ratio at grid index i.
function _cond_val_ratio(cname, i, absS, absA, abs_eta_1, abs_eta_end)
    val = cname == "S"     ? absS[i] :
          cname == "A"     ? absA[i] :
          cname == "eta_1" ? abs_eta_1[i] : abs_eta_end[i]
    den = cname == "S"     ? max(absA[i], eps()) :
          cname == "A"     ? max(absS[i], eps()) :
          (abs_eta_1[i] + abs_eta_end[i] + eps())
    return val, val / den
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
        # Capillary endpoint map (Λ/We absorbed into the explicit d·σ prefactor,
        # matching how c_hydro absorbs ΛΓ/Fr²) -- see Surferbot.capillary_endpoint_map
        # for the sign-convention proof (a past `+` here silently broke the
        # A(xM=0)=0 symmetry a physically symmetric problem must have).
        C_sigma      = derived.d * fparams.sigma .*
                        capillary_endpoint_map(Psi[end, :], Psi[1, :],
                            ComplexF64.(payload.s_vec), ComplexF64.(payload.s_vec_left)),
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
    A_sys = Diagonal(D) .- theory_ctx.Z_psi .+ theory_ctx.C_sigma
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
                                   EI_list::Union{Nothing,AbstractVector{Float64}}=nothing,
                                   n_EI::Int=301,
                                   include_resonance_stripes::Bool=true)
    params    = artifact.base_params
    EI_artifact = collect(Float64.(artifact.parameter_axes.EI))
    # When an explicit EI_list is provided (e.g. EI_scatter from build_LH_plot),
    # use it as-is for both root-finding and resonance — the scatter x-axis is
    # calibrated to it.  When none is given, densify the artifact EI grid so branch
    # termini are resolved, but keep the artifact grid for resonance detection.
    if EI_list !== nothing
        EI_coarse = collect(EI_list)   # resonance uses the same grid as root-finding
    else
        EI_coarse = EI_artifact        # resonance: original sparse sweep grid
        EI_list   = 10 .^ collect(range(log10(EI_artifact[1]),
                                        log10(EI_artifact[end]); length=n_EI))
    end
    logEI_axis = log10.(EI_list)
    xM_grid    = collect(range(-0.5, 0.0; length=601))
    theory_ctx = theoretical_modal_context_LH(params; output_dir=output_dir)

    pts_logEI      = Float64[]
    pts_xM         = Float64[]
    prev_hi_roots  = Float64[]
    prev_hi_lEI    = NaN
    term_intervals = Tuple{Float64,Float64}[]   # (lEI_last_root, lEI_first_miss)
    n              = length(xM_grid)

    # ── Fine-grid pass: zero-crossing root finding only (no resonance) ───────────
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

        # Record terminus interval (no synthetic point — just note the logEI gap).
        if !isempty(prev_hi_roots)
            for pxM in prev_hi_roots
                if !any(r -> abs(r - pxM) < TERM_TOL, roots)
                    push!(term_intervals, (prev_hi_lEI, logEI_axis[iei]))
                    break
                end
            end
        end

        curr_hi = filter(r -> r < TERM_MIN_XM, roots)
        if !isempty(curr_hi)
            prev_hi_roots = curr_hi
            prev_hi_lEI   = logEI_axis[iei]
        else
            prev_hi_roots = Float64[]
        end

        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM,    r)
        end
    end

    # ── Targeted terminus refinement: dense re-scan of each branch-exit interval ─
    # Evaluates N_TERM_FINE additional κ steps inside the narrow logEI gap where a
    # high-xM branch disappeared, finding real roots that get closer to xM = -0.5.
    # Purely theoretical: same root-finder, no synthetic points.
    N_TERM_FINE = 30
    for (lEI_lo, lEI_hi) in unique(term_intervals)
        fine = collect(range(lEI_lo, lEI_hi; length = N_TERM_FINE + 2))[2:end-1]
        for lEI in fine
            EI = 10^lEI
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
            for r in roots_for_condition(condition_name, xM_grid,
                                         absS, absA, abs_eta_1, abs_eta_end)
                push!(pts_logEI, lEI)
                push!(pts_xM,    r)
            end
        end
    end

    # ── Coarse-grid pass: resonance detection only (uses original EI spacing) ───
    # Resonance thresholds were calibrated on the coarse grid; re-running on the
    # fine grid introduces spurious stripes at new κ values near true resonances.
    res_candidates = Tuple{Int,Float64}[]
    col_amp = fill(NaN, length(EI_coarse))
    if include_resonance_stripes && condition_name in ("S", "A")
        logEI_coarse = log10.(EI_coarse)
        for (iei, EI) in enumerate(EI_coarse)
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
            col_amp[iei] = max(mean(absS), mean(absA))
            # This still uses the pre-fix right-minus-left sign convention
            # (opposite Surferbot.Analysis.beam_asymmetry's corrected formula)
            # -- harmless here because only abs(alpha_col) is used below for a
            # magnitude-only resonance threshold, which is sign-invariant.
            alpha_col = @. -(abs_eta_1^2 - abs_eta_end^2) /
                            (abs_eta_1^2 + abs_eta_end^2 + eps())
            q_val = quantile(abs.(alpha_col), 0.15)
            q_val < RESONANCE_ALPHA_CUTOFF || continue
            is_odd_resonance = mean(absS) < mean(absA)
            if (condition_name == "S" && is_odd_resonance) ||
               (condition_name == "A" && !is_odd_resonance)
                push!(res_candidates, (iei, q_val))
            end
        end
    end

    # Amplitude guard: a low alpha-quantile alone can occur in the smooth
    # high-κ regime without any real near-singular amplification (see
    # RESONANCE_AMP_FACTOR). Require genuine amplification relative to the
    # across-grid baseline before accepting a candidate as a real resonance.
    if !isempty(res_candidates)
        baseline = median(filter(isfinite, col_amp))
        filter!(c -> col_amp[c[1]] > RESONANCE_AMP_FACTOR * baseline, res_candidates)
    end

    # Deduplicate resonance candidates via the shared Surferbot helper: without
    # this, a dense scatter_logK grid samples several adjacent columns inside one
    # physical resonance's width, each independently passing the threshold,
    # producing multiple near-duplicate vertical stripes for a single resonance.
    if !isempty(res_candidates)
        logEI_coarse = log10.(EI_coarse)
        res_xM = collect(range(-0.5, 0.0; length=RESONANCE_N_PTS))
        for (best_iei, _) in dedup_resonance_runs(res_candidates)
            append!(pts_logEI, fill(logEI_coarse[best_iei], RESONANCE_N_PTS))
            append!(pts_xM,    res_xM)
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

function build_LH_plot(artifact, csv_path, output_dir; xlim_min::Float64,
                       include_resonance_stripes::Bool=false)
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

    # Pad to xM/L = -0.5 by repeating the first row (CSV data starts at -0.48)
    if minimum(xM_axis) > -0.5
        xM_axis  = vcat([-0.5], xM_axis)
        alpha_LH = vcat(alpha_LH[1:1, :], alpha_LH)
    end

    XLIMS = (xlim_min, max_logK_data)
    YLIMS = (-0.5, 0.0)

    # 181 (not 57): the coarse grid missed narrow resonances entirely (no
    # scatter_logK column landed inside their width) and under-sampled branches
    # approaching a resonance into a jagged, spurious-looking wiggle. Safe now
    # that find_filtered_minima interpolates and resonance stripes are deduped
    # by run (Surferbot.dedup_resonance_runs) -- neither problem re-appears at
    # higher density.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK_data; length=181))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    results = Dict{String, NamedTuple}()
    for cname in CURVE_NAMES
        @info "Computing LH roots: $cname"
        res = get_roots_theoretical_LH(artifact, cname;
                                        output_dir=output_dir,
                                        EI_list=EI_scatter,
                                        include_resonance_stripes=include_resonance_stripes)
        results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
    end

    orth_xM = collect(range(YLIMS[1], YLIMS[2]; length=301))
    orth_ctx = theoretical_modal_context_LH(params; output_dir=output_dir)
    orthogonality = theoretical_phase_orthogonality_field(EI_scatter, orth_xM, orth_ctx)

    # Resonance-stripe sync: beam-end is more sensitive to certain resonances that the
    # LH far-field quantile misses.  Collect beam-detected resonance candidates (index,
    # q_val), group consecutive scatter-grid indices into runs, keep the minimum-q
    # index per run, then inject any stripe not already present in the LH results.
    if include_resonance_stripes
        beam_sync_ctx   = theoretical_modal_context(params; output_dir=output_dir)
        xM_sync         = collect(range(-0.49, 0.0; length=401))
        sync_candidates = Dict{String, Vector{Tuple{Int,Float64}}}("S"=>[], "A"=>[])
        sync_amp        = fill(NaN, length(EI_scatter))
        for (iei, EI) in enumerate(EI_scatter)
            scatter_logK[iei] < xlim_min && continue
            absS=Float64[]; absA=Float64[]; abs1=Float64[]; abse=Float64[]
            for xM in xM_sync
                q = solve_theoretical_modal_response(EI, xM, beam_sync_ctx)
                d = theoretical_endpoint_diagnostics_beam(q, beam_sync_ctx)
                push!(absS, abs(d.S)); push!(absA, abs(d.A))
                push!(abs1, abs(d.eta_beam_1)); push!(abse, abs(d.eta_beam_end))
            end
            sync_amp[iei] = max(mean(absS), mean(absA))
            alpha_col   = @. -(abs1^2 - abse^2) / (abs1^2 + abse^2 + eps())
            is_odd      = mean(absS) < mean(absA)
            resonance_q = is_odd ? 0.20 : 0.15
            q_val       = quantile(abs.(alpha_col), resonance_q)
            q_val < RESONANCE_ALPHA_CUTOFF || continue
            cond = is_odd ? "S" : "A"
            push!(sync_candidates[cond], (iei, q_val))
        end
        let baseline = median(filter(isfinite, sync_amp))
            for cond in ("S", "A")
                filter!(c -> sync_amp[c[1]] > RESONANCE_AMP_FACTOR * baseline,
                        sync_candidates[cond])
            end
        end
        res_xM_sync = collect(range(xM_sync[1], xM_sync[end]; length=RESONANCE_N_PTS))
        for cond in ("S", "A")
            for (best_iei, _) in dedup_resonance_runs(sync_candidates[cond])
                logK_val = scatter_logK[best_iei]
                already_present = any(abs.(results[cond].logK .- logK_val) .< 0.01)
                already_present && continue
                prev = results[cond]
                results[cond] = (
                    logK=vcat(prev.logK, fill(logK_val, RESONANCE_N_PTS)),
                    xM_norm=vcat(prev.xM_norm, res_xM_sync),
                )
            end
        end
    end

    # Build plot
    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    # Pair the two domain-end elevation conditions by color; line styles retain
    # the distinction in greyscale and separate the black S/A/orthogonality laws.
    # The S, A, and orthogonality families remain distinct after grayscale printing.
    curve_colors = [RGBf(0.00, 0.00, 0.00), RGBf(0.20, 0.20, 0.20),
                    okabe_ito[7], okabe_ito[7]]
    curve_styles = [:solid, :dashdot, :dash, :solid]
    orthogonality_color = RGBf(0.40, 0.40, 0.40)

    return PaperPlotTheme.with_theme() do
        fig = Figure(size = (820, 640), backgroundcolor = :white,
            figure_padding = (12, 8, 12, 12))
        ax = Axis(fig[1, 1]; xlabel = L"x_M / L", ylabel = L"\kappa",
            xlabelsize = 16, ylabelsize = 16, xticklabelsize = 14, yticklabelsize = 14,
            yticks = kappa_exp_xticks(XLIMS), xgridvisible = false, ygridvisible = false)
        xlims!(ax, YLIMS...)
        ylims!(ax, XLIMS...)
        hm = heatmap!(ax, xM_axis, logEI_axis .- shift, alpha_LH;
            colormap = :balance, colorrange = (-1, 1))
        for (i, cname) in enumerate(CURVE_NAMES)
            res = results[cname]
            mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
                   (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
            isempty(res.logK[mask]) && continue
            lk_path, xm_path, res_lks = cluster_branches(res.logK[mask], res.xM_norm[mask])
            isempty(lk_path) || lines!(ax, xm_path, lk_path;
                color = curve_colors[i], linestyle = curve_styles[i], linewidth = 4.0)
            for rlk in res_lks
                XLIMS[1] <= rlk <= XLIMS[2] || continue
                hlines!(ax, [rlk]; color = curve_colors[i], linewidth = 4.0)
            end
        end
        contour!(ax, orth_xM, scatter_logK, orthogonality; levels = [0.0],
            color = orthogonality_color, linewidth = 3.0, linestyle = :dash)
        legend_entries = [
            LineElement(color = :black, linestyle = :solid, linewidth = 4.0),
            LineElement(color = RGBf(0.20, 0.20, 0.20), linestyle = :dashdot, linewidth = 4.0),
            LineElement(color = orthogonality_color, linestyle = :dash, linewidth = 3.0),
            LineElement(color = okabe_ito[7], linestyle = :dash, linewidth = 4.0),
            LineElement(color = okabe_ito[7], linestyle = :solid, linewidth = 4.0),
        ]
        axislegend(ax, legend_entries, [CURVE_LABELS[1], CURVE_LABELS[2], L"S \perp A",
            CURVE_LABELS[3], CURVE_LABELS[4]]; position = :lt, labelsize = 14,
            patchsize = (84, 20), framecolor = :black, backgroundcolor = (:white, 0.85))
        Colorbar(fig[1, 2], hm; label = L"\alpha", labelsize = 16, ticklabelsize = 14)
        return fig
    end
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
                                     theory_ctx; n_EI::Int=301)
    EI_coarse   = collect(EI_list)  # keep original for resonance detection
    # Densify the EI grid so branch termini are resolved without post-hoc bisection.
    logEI_dense = collect(range(log10(EI_coarse[1]), log10(EI_coarse[end]); length=n_EI))
    EI_list     = 10 .^ logEI_dense
    logEI_axis  = logEI_dense
    xM_grid     = collect(range(-0.5, 0.0; length=601))

    pts_logEI      = Float64[]
    pts_xM         = Float64[]
    res_logEI      = Float64[]
    res_candidates = Tuple{Int, Float64}[]
    prev_hi_roots  = Float64[]
    prev_hi_lEI    = NaN
    term_intervals = Tuple{Float64,Float64}[]
    n              = length(xM_grid)

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

        if !isempty(prev_hi_roots)
            for pxM in prev_hi_roots
                if !any(r -> abs(r - pxM) < TERM_TOL, roots)
                    push!(term_intervals, (prev_hi_lEI, logEI_axis[iei]))
                    break
                end
            end
        end
        curr_hi = filter(r -> r < TERM_MIN_XM, roots)
        if !isempty(curr_hi)
            prev_hi_roots = curr_hi; prev_hi_lEI = logEI_axis[iei]
        else
            prev_hi_roots = Float64[]
        end

        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM,    r)
        end
    end

    # ── Targeted terminus refinement ─────────────────────────────────────────────
    N_TERM_FINE = 30
    for (lEI_lo, lEI_hi) in unique(term_intervals)
        fine = collect(range(lEI_lo, lEI_hi; length = N_TERM_FINE + 2))[2:end-1]
        for lEI in fine
            EI = 10^lEI
            absS = Float64[]; absA = Float64[]
            abs_eta_1 = Float64[]; abs_eta_end = Float64[]
            for xM_norm in xM_grid
                q    = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
                diag = theoretical_endpoint_diagnostics_beam(q, theory_ctx)
                push!(absS, abs(diag.S)); push!(absA, abs(diag.A))
                push!(abs_eta_1, abs(diag.eta_beam_1))
                push!(abs_eta_end, abs(diag.eta_beam_end))
            end
            for r in roots_for_condition(condition_name, xM_grid,
                                         absS, absA, abs_eta_1, abs_eta_end)
                push!(pts_logEI, lEI); push!(pts_xM, r)
            end
        end
    end

    # ── Coarse-grid resonance detection (original EI spacing, no spurious hits) ─
    col_amp = fill(NaN, length(EI_coarse))
    if condition_name in ("S", "A")
        logEI_coarse = log10.(EI_coarse)
        is_coupled   = Float64(theory_ctx.params.d) > 0.0
        n_res        = searchsortedfirst(xM_grid, -0.49)
        for (iei, EI) in enumerate(EI_coarse)
            absS = Float64[]; absA = Float64[]
            abs_eta_1 = Float64[]; abs_eta_end = Float64[]
            for xM_norm in xM_grid
                q    = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
                diag = theoretical_endpoint_diagnostics_beam(q, theory_ctx)
                push!(absS, abs(diag.S)); push!(absA, abs(diag.A))
                push!(abs_eta_1, abs(diag.eta_beam_1))
                push!(abs_eta_end, abs(diag.eta_beam_end))
            end
            col_amp[iei] = max(mean(absS), mean(absA))
            # Pre-fix right-minus-left convention -- harmless, see the identical
            # note near the first alpha_col above (only abs(alpha_col) is used).
            alpha_col = @. -(abs_eta_1^2 - abs_eta_end^2) /
                            (abs_eta_1^2 + abs_eta_end^2 + eps())
            is_odd_resonance = mean(absS) < mean(absA)
            resonance_q = is_odd_resonance ? 0.20 : (is_coupled ? 0.15 : 0.10)
            q_val = quantile(abs.(alpha_col[n_res:end]), resonance_q)
            if q_val < RESONANCE_ALPHA_CUTOFF
                if (condition_name == "S" && is_odd_resonance) ||
                   (condition_name == "A" && !is_odd_resonance)
                    push!(res_candidates, (iei, q_val))
                end
            end
        end
    end

    # Amplitude guard (see RESONANCE_AMP_FACTOR): reject candidates that dip
    # below the alpha-quantile cutoff without genuine amplification.
    if !isempty(res_candidates)
        baseline = median(filter(isfinite, col_amp))
        filter!(c -> col_amp[c[1]] > RESONANCE_AMP_FACTOR * baseline, res_candidates)
    end

    # Deduplicate resonance candidates via the shared Surferbot helper. This
    # prevents adjacent scatter points that both pass the threshold for the same
    # physical resonance from generating two stacked vertical stripes.
    if !isempty(res_candidates)
        logEI_coarse    = log10.(EI_coarse)
        res_xM_template = collect(range(-0.5, 0.0; length=RESONANCE_N_PTS))
        for (best_iei, _) in dedup_resonance_runs(res_candidates)
            best_le = logEI_coarse[best_iei]
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

    # Pad to xM/L = -0.5 by repeating the first row (CSV data starts at -0.48)
    if minimum(xM_axis) > -0.5
        xM_axis    = vcat([-0.5], xM_axis)
        alpha_beam = vcat(alpha_beam[1:1, :], alpha_beam)
    end

    XLIMS = (xlim_min, max_logK_data)
    YLIMS = (-0.5, 0.0)

    # 181 (not 57): see build_LH_plot for why.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK_data; length=181))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    theory_ctx = theoretical_modal_context(params; output_dir=output_dir)

    results       = Dict{String, NamedTuple}()
    resonance_logK = Dict{String, Vector{Float64}}()
    for cname in CURVE_NAMES
        @info "Computing beam-end roots: $cname"
        res = get_roots_theoretical_beam(EI_scatter, cname, theory_ctx)
        results[cname]        = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
        resonance_logK[cname] = res.resonance_logEI .- shift
    end

    # CSV-based resonance seeding: for uncoupled (d=0) only.  Uncoupled resonances
    # have half-width ~0.025 logK (no radiation damping); the scatter grid can miss
    # the sharpest ones.  For coupled, the scatter grid already catches all resonances
    # (wider peaks), so seeding is skipped to avoid borderline stripes inconsistent
    # with the LH plot.
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
                res_xM = collect(range(Float64(xM_axis[1]), Float64(xM_axis[end]); length=RESONANCE_N_PTS))
                prev = results[cond]
                results[cond]        = (logK    = vcat(prev.logK,    fill(logK_val, RESONANCE_N_PTS)),
                                        xM_norm = vcat(prev.xM_norm, res_xM))
                push!(resonance_logK[cond], logK_val)
                @info "CSV-seeded resonance stripe: $cond at logK=$(round(logK_val; digits=3))"
            end
        end
    end

    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
    curve_styles = [:solid, :solid, :solid, :solid]

    return PaperPlotTheme.with_theme() do
        fig = Figure(size = (820, 640), backgroundcolor = :white,
            figure_padding = (12, 8, 12, 12))
        ax = Axis(fig[1, 1]; xlabel = L"x_M / L", ylabel = L"\kappa",
            xlabelsize = 16, ylabelsize = 16, xticklabelsize = 14, yticklabelsize = 14,
            yticks = kappa_exp_xticks(XLIMS), xgridvisible = false, ygridvisible = false)
        xlims!(ax, YLIMS...)
        ylims!(ax, XLIMS...)
        hm = heatmap!(ax, xM_axis, logEI_axis .- shift, alpha_beam;
            colormap = :balance, colorrange = (-1, 1))
        for (i, cname) in enumerate(CURVE_NAMES)
            res = results[cname]
            mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
                   (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
            isempty(res.logK[mask]) && continue
            lk_path, xm_path, res_lks = cluster_branches(res.logK[mask], res.xM_norm[mask])
            isempty(lk_path) || lines!(ax, xm_path, lk_path;
                color = curve_colors[i], linestyle = curve_styles[i], linewidth = 4.0)
            for rlk in res_lks
                XLIMS[1] <= rlk <= XLIMS[2] || continue
                hlines!(ax, [rlk]; color = curve_colors[i], linewidth = 4.0)
            end
        end
        legend_entries = [LineElement(color = curve_colors[i], linestyle = curve_styles[i], linewidth = 4.0)
                          for i in eachindex(CURVE_NAMES)]
        axislegend(ax, legend_entries, BEAM_CURVE_LABELS; position = :lt, labelsize = 14,
            framecolor = :black, backgroundcolor = (:white, 0.85))
        Colorbar(fig[1, 2], hm; label = L"\alpha", labelsize = 16, ticklabelsize = 14)
        return fig
    end
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main(args=ARGS)
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
    out_cpl_LH = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_cpl_theo.pdf")
    save(out_cpl_LH, p_cpl_LH)
    println("Saved $out_cpl_LH")

    "--lh-coupled-only" in args && return

    # ── 2. Beam-end coupled plot ──────────────────────────────────────────────
    p_cpl_beam = build_beam_end_plot(art_cpl,
                                      joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                                      output_dir; xlim_min=-4.0)
    out_cpl_beam = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_cpl_beam.pdf")
    save(out_cpl_beam, p_cpl_beam)
    println("Saved $out_cpl_beam")

    # ── 3. Beam-end uncoupled plot (x axis −5 to max) ────────────────────────
    art_ucpl = load_sweep(joinpath(output_dir, "jld2",
                                   "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))

    p_ucpl_beam = build_beam_end_plot(art_ucpl,
                                       joinpath(output_dir, "csv", "sweeper_uncoupled_full_grid.csv"),
                                       output_dir; xlim_min=-4.0)
    out_ucpl_beam = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_ucpl_beam.pdf")
    save(out_ucpl_beam, p_ucpl_beam)
    println("Saved $out_ucpl_beam")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
