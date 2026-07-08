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

# Branch-terminus detection: when a tracked high-xM root disappears between
# consecutive EI steps and the function is still decreasing at xM=0.5, we add
# one terminal scatter point at xM=0.5 for that EI.
const TERM_MIN_XM = 0.35   # only track roots above this xM as upward branches
const TERM_TOL    = 0.08   # max xM gap to consider a root "continued"

const RESONANCE_ALPHA_CUTOFF  = 0.04  # 10th-percentile of |α_LH| across xM < this → resonance column
const RESONANCE_N_PTS         = 20   # number of evenly-spaced xM points to emit per resonance column

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
function cluster_branches(logK_pts, xM_pts)
    # ── 1. Separate resonance stripes ────────────────────────────────────────
    counts = Dict{Float64,Int}()
    for lk in logK_pts; counts[lk] = get(counts, lk, 0) + 1; end
    res_lk_set = Set(lk for (lk,c) in counts if c >= RESONANCE_N_PTS)

    # zero-crossing points only
    xing_lk = Float64[]; xing_xm = Float64[]
    for (lk, xm) in zip(logK_pts, xM_pts)
        lk ∈ res_lk_set && continue
        push!(xing_lk, lk); push!(xing_xm, xm)
    end

    resonance_lks = sort(collect(res_lk_set))
    isempty(xing_lk) && return Float64[], Float64[], resonance_lks

    # ── 2. Group by logK column ───────────────────────────────────────────────
    cols = Dict{Float64, Vector{Float64}}()
    for (lk, xm) in zip(xing_lk, xing_xm)
        push!(get!(cols, lk, Float64[]), xm)
    end
    unique_lks = sort(collect(keys(cols)))
    for lk in unique_lks; sort!(cols[lk]); end

    # ── 3. Nearest-neighbour branch assignment ────────────────────────────────
    # Each branch is a list of (logK, xM) pairs in logK order.
    branches = [[(unique_lks[1], xm)] for xm in cols[unique_lks[1]]]

    for i in 2:length(unique_lks)
        lk       = unique_lks[i]
        new_xms  = copy(cols[lk])
        matched  = fill(false, length(new_xms))

        for branch in branches
            isempty(branch) && continue
            last_xm = last(branch)[2]
            isnan(last_xm) && continue
            # find closest unmatched root in the new column
            best_j, best_d = 0, Inf
            for (j, xm) in enumerate(new_xms)
                matched[j] && continue
                d = abs(xm - last_xm)
                if d < best_d; best_d = d; best_j = j; end
            end
            # connect only if close enough (< half the xM range)
            if best_j > 0 && best_d < 0.25
                push!(branch, (lk, new_xms[best_j]))
                matched[best_j] = true
            else
                push!(branch, (NaN, NaN))   # branch gap
            end
        end
        # start a new branch for any unmatched root
        for (j, xm) in enumerate(new_xms)
            matched[j] || push!(branches, [(lk, xm)])
        end
    end

    # ── 4. Flatten branches with NaN separators ───────────────────────────────
    lk_out = Float64[]; xm_out = Float64[]
    for branch in branches
        length(branch) < 2 && continue    # skip isolated single points
        append!(lk_out, first.(branch))
        append!(xm_out, last.(branch))
        push!(lk_out, NaN); push!(xm_out, NaN)
    end
    return lk_out, xm_out, resonance_lks
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

# find_filtered_minima is provided by Surferbot (Julia/src/modal.jl) — interpolates
# a sub-grid root location instead of snapping to the xgrid point, avoiding a
# staircase artifact wherever the branch-tracing loop below samples logK densely.

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
        # Capillary endpoint map C^σ_mn = d·σ·(W_m(L/2) s_n^+ + W_m(-L/2) s_n^-),
        # dimensional analogue of paper eq:Ksigma_linear_map_app (Λ/We absorbed
        # into the explicit d·σ prefactor, matching how c_hydro absorbs ΛΓ/Fr²).
        C_sigma      = derived.d * fparams.sigma .*
                        (Psi[end, :] * transpose(ComplexF64.(payload.s_vec)) .+
                         Psi[1,   :] * transpose(ComplexF64.(payload.s_vec_left))),
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
                                   n_EI::Int=301)
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
    xM_grid    = collect(range(0.0, 0.5; length=601))
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

        curr_hi = filter(r -> r > TERM_MIN_XM, roots)
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
    # high-xM branch disappeared, finding real roots that get closer to xM = 0.5.
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
    if condition_name in ("S", "A")
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

    # Deduplicate resonance candidates via the shared Surferbot helper: without
    # this, a dense scatter_logK grid samples several adjacent columns inside one
    # physical resonance's width, each independently passing the threshold,
    # producing multiple near-duplicate vertical stripes for a single resonance.
    if !isempty(res_candidates)
        logEI_coarse = log10.(EI_coarse)
        res_xM = collect(range(0.0, 0.5; length=RESONANCE_N_PTS))
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

    # 57 evenly-spaced logK values, extended 0.1 units left of xlim_min so resonances
    # near the left edge are not missed.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK_data; length=57))
    EI_scatter   = 10 .^ (scatter_logK .+ shift)

    results = Dict{String, NamedTuple}()
    for cname in CURVE_NAMES
        @info "Computing LH roots: $cname"
        res = get_roots_theoretical_LH(artifact, cname;
                                        output_dir=output_dir,
                                        EI_list=EI_scatter)
        results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
    end

    # Resonance-stripe sync: beam-end is more sensitive to certain resonances that the
    # LH far-field quantile misses.  Collect beam-detected resonance candidates (index,
    # q_val), group consecutive scatter-grid indices into runs, keep the minimum-q
    # index per run, then inject any stripe not already present in the LH results.
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
    res_xM_sync = collect(range(xM_sync[1], xM_sync[end]; length=RESONANCE_N_PTS))
    for cond in ("S", "A")
        for (best_iei, _) in dedup_resonance_runs(sync_candidates[cond])
            logK_val    = scatter_logK[best_iei]
            already_present = any(abs.(results[cond].logK .- logK_val) .< 0.01)
            already_present && continue
            prev = results[cond]
            results[cond] = (logK    = vcat(prev.logK,    fill(logK_val, RESONANCE_N_PTS)),
                             xM_norm = vcat(prev.xM_norm, res_xM_sync))
        end
    end

    # Build plot
    okabe_ito    = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
    # Redundant line-style encoding so curves are distinguishable in greyscale
    curve_styles = [:solid, :solid, :solid, :solid]

    plt_opts = (
        xlabel  = L"\kappa",
        xticks  = kappa_exp_xticks(XLIMS),
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
            # Resonance stripes: same color, thinner, slightly transparent
            vline!(p, [rlk]; color=curve_colors[i], linewidth=4.0, label=false)
        end
    end

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
                                     theory_ctx; n_EI::Int=301)
    EI_coarse   = collect(EI_list)  # keep original for resonance detection
    # Densify the EI grid so branch termini are resolved without post-hoc bisection.
    logEI_dense = collect(range(log10(EI_coarse[1]), log10(EI_coarse[end]); length=n_EI))
    EI_list     = 10 .^ logEI_dense
    logEI_axis  = logEI_dense
    xM_grid     = collect(range(0.0, 0.5; length=601))

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
        curr_hi = filter(r -> r > TERM_MIN_XM, roots)
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
    if condition_name in ("S", "A")
        logEI_coarse = log10.(EI_coarse)
        is_coupled   = Float64(theory_ctx.params.d) > 0.0
        n_res        = searchsortedlast(xM_grid, 0.49)
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
            alpha_col = @. -(abs_eta_1^2 - abs_eta_end^2) /
                            (abs_eta_1^2 + abs_eta_end^2 + eps())
            is_odd_resonance = mean(absS) < mean(absA)
            resonance_q = is_odd_resonance ? 0.20 : (is_coupled ? 0.15 : 0.10)
            q_val = quantile(abs.(alpha_col[1:n_res]), resonance_q)
            if q_val < RESONANCE_ALPHA_CUTOFF
                if (condition_name == "S" && is_odd_resonance) ||
                   (condition_name == "A" && !is_odd_resonance)
                    push!(res_candidates, (iei, q_val))
                end
            end
        end
    end

    # Deduplicate resonance candidates via the shared Surferbot helper. This
    # prevents adjacent scatter points that both pass the threshold for the same
    # physical resonance from generating two stacked vertical stripes.
    if !isempty(res_candidates)
        logEI_coarse    = log10.(EI_coarse)
        res_xM_template = collect(range(0.0, 0.5; length=RESONANCE_N_PTS))
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

    # Pad to xM/L = 0.5 by repeating the last row (CSV data ends at 0.48)
    if maximum(xM_axis) < 0.5
        xM_axis   = vcat(xM_axis, [0.5])
        alpha_beam = vcat(alpha_beam, alpha_beam[end:end, :])
    end

    XLIMS = (xlim_min, max_logK_data)
    YLIMS = (0.0, 0.5)

    # 57 evenly-spaced logK values across the full visible range — extend 0.1 units
    # left of xlim_min so resonances sitting just inside the left edge are not missed.
    scatter_logK = collect(range(xlim_min - 0.1, max_logK_data; length=57))
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

    plt_opts = (
        xlabel  = L"\kappa",
        xticks  = kappa_exp_xticks(XLIMS),
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

    p = heatmap(logEI_axis .- shift, xM_axis, alpha_beam; plt_opts...)

    for (i, cname) in enumerate(CURVE_NAMES)
        res  = results[cname]
        mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
               (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
        isempty(res.logK[mask]) && continue
        lk_path, xm_path, res_lks = cluster_branches(res.logK[mask], res.xM_norm[mask])
        isempty(lk_path) || plot!(p, lk_path, xm_path;
            label      = BEAM_CURVE_LABELS[i],
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

    # ── 1. LH (domain-end) coupled plot ──────────────────────────────────────
    art_cpl = load_sweep(joinpath(output_dir, "jld2",
                                  "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    theory_ctx_cpl = theoretical_modal_context_LH(art_cpl.base_params; output_dir=output_dir)
    print_symmetry_check(theory_ctx_cpl)

    p_cpl_LH = build_LH_plot(art_cpl,
                               joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                               output_dir; xlim_min=-4.0)
    out_cpl_LH = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_cpl_theo.pdf")
    savefig(p_cpl_LH, out_cpl_LH)
    println("Saved $out_cpl_LH")

    # ── 2. Beam-end coupled plot ──────────────────────────────────────────────
    p_cpl_beam = build_beam_end_plot(art_cpl,
                                      joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"),
                                      output_dir; xlim_min=-4.0)
    out_cpl_beam = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_cpl_beam.pdf")
    savefig(p_cpl_beam, out_cpl_beam)
    println("Saved $out_cpl_beam")

    # ── 3. Beam-end uncoupled plot (x axis −5 to max) ────────────────────────
    art_ucpl = load_sweep(joinpath(output_dir, "jld2",
                                   "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))

    p_ucpl_beam = build_beam_end_plot(art_ucpl,
                                       joinpath(output_dir, "csv", "sweeper_uncoupled_full_grid.csv"),
                                       output_dir; xlim_min=-4.0)
    out_ucpl_beam = joinpath(fig_dir, "plot_dimensionless_diagnostics_LH_ucpl_beam.pdf")
    savefig(p_ucpl_beam, out_ucpl_beam)
    println("Saved $out_ucpl_beam")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
