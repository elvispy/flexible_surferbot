using Surferbot
using JLD2
using Plots
using LaTeXStrings
using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Statistics
import SpecialFunctions: erf

"""
Julia/experiments/plot_dimensionless_diagnostics.jl

# Physics overview

The raft is an Euler–Bernoulli beam on a 2-D potential-flow fluid. Motor vibration
(frequency ω, Gaussian force envelope of width σ, position xM) excites steady-state
complex amplitudes. The principal diagnostic is the **asymmetry index**

    α = (|η₁|² − |η_end|²) / (|η₁|² + |η_end|²)        α ∈ [-1, 1]

where η₁ and η_end are the surface-elevation Fourier amplitudes at the left and
right beam endpoints. α = 0 is the thrust-neutral condition; its sign determines
the drift direction.

# Endpoint decomposition into S and A

Expand the beam response in the L²-orthonormal free-free modes {Ψ_n}.  Even
modes are symmetric about the beam centre (same value at both ends); odd modes
are antisymmetric (opposite sign).  Define

    S ≡ Σ_{n even}  q_n · Ψ_n(x_end)    (symmetric endpoint amplitude)
    A ≡ Σ_{n odd}   q_n · Ψ_n(x_end)    (antisymmetric endpoint amplitude)

where Ψ_n(x_end) denotes the n-th mode evaluated at the right endpoint.  Then

    η_end = S + A    (right endpoint: both contributions add)
    η₁    = S − A    (left  endpoint: odd modes flip sign)

so  α ∝ Re(S A*), and α = 0 when |S| = 0, |A| = 0, or ∠S − ∠A = ±π/2.
The scatter overlays mark where each sub-condition vanishes in the (κ, xM) plane.

# A-priori modal law

The modal coefficient vector q ∈ ℂ^N satisfies

    (D − Z_psi) q = −f                             (a-priori law)

where:
  D[m]   = EI β_m⁴ − ρ_R ω² + d ρ g    (structural stiffness − inertia + hydrostatic;
                                           diagonal in the Ψ basis)
  β_m    = m-th free-free eigenvalue / L  (from `freefree_betaL_roots`)
  Z_psi  = N×N radiation impedance matrix in the Ψ basis — a purely fluid property
            (constant across all EI and xM; computed once via N prescribed-mode
            fluid solves by `prescribed_wn_diagonal_impedance.jl`)
  f[m]   = ⟨Ψ_m, F_motor⟩  (L²-projection of the Gaussian motor load onto mode m)

`solve_theoretical_modal_response` assembles and solves this system analytically.
`get_roots_theoretical` sweeps (EI, xM) using the law and finds the zero curves.
`get_roots_integral` does the same from the pre-computed CSV sweep data.

# Dimensionless stiffness κ

Both plot axes use

    κ = EI / (ρ_R L⁴ ω²)

which is the ratio of beam bending stiffness to inertial loading. In `main()`, the
EI axis of the raw CSV is shifted by  shift = log₁₀(ρ_R L⁴ ω²)  to obtain log₁₀(κ).
Rigid-limit: κ ≳ 1.  Highly flexible: κ ≪ 1.

# Basis bookkeeping

Two mode sets appear:
- **W basis**  — raw analytical free-free modes, L∞-normalised.
  `prescribed_wn_diagonal_impedance.jl` works in this basis to extract Z_raw.
- **Ψ basis**  — L²-orthonormal version of W obtained by weighted Gram–Schmidt.
  All modal coefficients in this file use the Ψ basis.
  The CSV columns `q_w0_re`, `q_w0_im`, … store Ψ-basis coefficients despite
  the "w" in their name (historical artefact from earlier sweeper versions).

Z_psi = T_{Ψ←W} Z_raw T_{W←Ψ} is cached alongside Z_raw; this file never
re-enters the W basis.

All amplitudes are complex Fourier envelopes under the e^{iωt} convention.
"""

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

# ─── Root-finding helpers ────────────────────────────────────────────────────
# `all_positive_roots`: sign-change crossings of a real 1-D curve (linear interp).
# `complex_roots`: crossings of Re(η)=0 gated by |Im(η)| < threshold (used for
#   older single-condition plots; superseded by find_filtered_minima in this file).

function all_positive_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]
        b = vals[i + 1]
        if a == 0
            xs[i] > 1e-6 && push!(roots, Float64(xs[i]))
        elseif a * b < 0
            # Linear interpolation for high-precision crossing
            t = a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            root > 1e-6 && push!(roots, Float64(root))
        end
    end
    return unique!(roots)
end

function complex_roots(xs::AbstractVector{<:Real}, re_vals::AbstractVector{<:Real}, im_vals::AbstractVector{<:Real}; threshold=1e-3)
    # Find crossings of Re(eta) and filter by Im(eta) near 0
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = re_vals[i]
        b = re_vals[i + 1]
        if a * b <= 0
            t = a == b ? 0.0 : a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            
            # Interpolate Imaginary part at the root
            im_root = im_vals[i] + t * (im_vals[i+1] - im_vals[i])
            if abs(im_root) < threshold
                push!(roots, Float64(root))
            end
        end
    end
    return unique!(roots)
end

# ─── Modal framework helpers ─────────────────────────────────────────────────

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

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    Phi = zeros(Float64, length(xi_motor), max_mode + 1)
    
    xi_L = [0.0]
    xi_R = [L]
    Phi_L = zeros(Float64, 1, max_mode + 1)
    Phi_R = zeros(Float64, 1, max_mode + 1)

    Phi[:, 1] .= 1.0
    Phi_L[1, 1] = 1.0
    Phi_R[1, 1] = 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
        Phi_L[1, 2] = -L / 2
        Phi_R[1, 2] = L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_L[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_L, L, betaL_el[n - 1])[1]
            Phi_R[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_R, L, betaL_el[n - 1])[1]
        end
    end
    return (; xM_norm=collect(Float64.(xM_norm)), Phi, Phi_L=vec(Phi_L), Phi_R=vec(Phi_R))
end

# ─── Root extraction ─────────────────────────────────────────────────────────

const NUM_MODES = 6

# Ratio cutoff for the filtered-minimum root-finder (see roots_for_condition).
# A local minimum of |S| at xM only counts as a root if |S|/|A| < RATIO_CUTOFF,
# i.e., the symmetric amplitude is genuinely small relative to the antisymmetric
# one — not merely a shallow dip. This suppresses spurious detections in regions
# where |S| and |A| are both large and the minimum is not physically meaningful.
const RATIO_CUTOFF = 0.5

# Run one fluid solve (any EI suffices; Ψ is EI-independent) to obtain the
# orthonormal mode matrix Ψ on the raft grid.  Phi (= W) is returned for
# reference but this file only uses Psi.
function get_basis_for_plotting(params)
    fparams = coerce_flexible_params(params)
    res = Surferbot.flexible_solver(fparams)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(res; num_modes=NUM_MODES, verbose=false)
    return (Phi=modal.Phi, Psi=modal.Psi, x=modal.x_raft)
end

# Read the sweep CSV, group by EI slice, compute (S, A, η_end, η₁) for each
# (EI, xM) row, then find the xM values where the requested condition vanishes.
#
# S and A are assembled by projecting the Ψ-basis modal coefficients q_n onto
# the right-endpoint weights w_end = Ψ(x_end):
#   S = Σ_{n even} q_n · w_end[n]    A = Σ_{n odd} q_n · w_end[n]
# The CSV stores these coefficients in columns q_w0_{re,im}, q_w1_{re,im}, …
# (historical name; they are Ψ-basis, not W-basis, coefficients).
function get_roots_integral(csv_path, condition_name; modes=0:(NUM_MODES-1))
    df = CSV.read(csv_path, DataFrame)
    L_raft = first(df.L_raft)
    basis = get_basis_for_plotting((L_raft=L_raft,))
    # w_end[n] = Ψ_n evaluated at the right beam endpoint.
    # w_start[n] = Ψ_n at left endpoint (kept for reference; currently unused).
    w_end = basis.Psi[end, :]
    w_start = basis.Psi[1, :]

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for group in groupby(df, :log10_EI)
        logEI = first(group.log10_EI)
        sorted = sort(group, :xM_over_L)
        xM_slice = collect(Float64.(sorted.xM_over_L))

        absS = Float64[]; absA = Float64[]
        abs_eta_1 = Float64[]; abs_eta_end = Float64[]
        for row in eachrow(sorted)
            S = 0.0 + 0.0im; A = 0.0 + 0.0im
            for n in 0:(NUM_MODES - 1)
                qn = complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")])
                if iseven(n)
                    S += qn * w_end[n + 1]   # even mode: same sign at both ends
                else
                    A += qn * w_end[n + 1]   # odd  mode: flips sign end-to-end
                end
            end
            eta_end = S + A    # right endpoint: η = S + A
            eta_1   = S - A    # left  endpoint: odd modes contribute with opposite sign
            push!(absS, abs(S))
            push!(absA, abs(A))
            push!(abs_eta_1, abs(eta_1))
            push!(abs_eta_end, abs(eta_end))
        end

        roots = roots_for_condition(condition_name, xM_slice,
                                    absS, absA, abs_eta_1, abs_eta_end)

        for r in roots
            push!(pts_logEI, logEI)
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

# For each condition, find xM values where that amplitude has a local minimum
# that passes the ratio test (see RATIO_CUTOFF).
#
# The ratio is defined so that it measures how dominant the vanishing quantity
# is relative to its complement:
#   |S| = 0 condition:    ratio = |S| / |A|   (must be < 0.5 to avoid spurious hits)
#   |A| = 0 condition:    ratio = |A| / |S|
#   |η₁| = 0 condition:   ratio = |η₁| / (|η₁| + |η_end|)
#   |η_end| = 0 condition: ratio = |η_end| / (|η₁| + |η_end|)
function roots_for_condition(condition_name, xgrid, absS, absA, abs_eta_1, abs_eta_end)
    if condition_name == "S"
        ratio = absS ./ max.(absA, eps())
        return find_filtered_minima(xgrid, absS, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "A"
        ratio = absA ./ max.(absS, eps())
        return find_filtered_minima(xgrid, absA, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "eta_1"
        denom = abs_eta_1 .+ abs_eta_end .+ eps()
        ratio = abs_eta_1 ./ denom
        return find_filtered_minima(xgrid, abs_eta_1, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "eta_end"
        denom = abs_eta_1 .+ abs_eta_end .+ eps()
        ratio = abs_eta_end ./ denom
        return find_filtered_minima(xgrid, abs_eta_end, ratio; ratio_cutoff=RATIO_CUTOFF)
    end
    return Float64[]
end

# Load (or compute and cache) the radiation impedance Z_psi and all fixed modal
# quantities needed to evaluate the a-priori law for arbitrary (EI, xM).
# Z_psi is computed once via N prescribed-displacement fluid solves; it captures
# the full fluid-structure radiation coupling in the Ψ basis.
# beta[m] = β_m L (dimensionless wavenumber) from the free-free characteristic eq.
function theoretical_modal_context(params; output_dir::AbstractString)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams;
        output_dir=output_dir,
        num_modes_basis=NUM_MODES,
    )
    derived = Surferbot.derive_params(fparams)
    Psi = payload.psi_basis.Psi
    return (
        params  = fparams,
        derived = derived,
        payload = payload,
        mode_numbers = collect(Int.(payload.mode_labels)),
        Psi     = Matrix{Float64}(Psi),
        x_raft  = collect(Float64.(payload.x_raft)),
        weights = collect(Float64.(payload.weights)),  # quadrature weights for ⟨·,·⟩
        w_start = Psi[1, :],    # Ψ_n(x_left)  — left-endpoint evaluation vector
        w_end   = Psi[end, :],  # Ψ_n(x_right) — right-endpoint evaluation vector
        beta    = collect(Float64.(payload.beta)),  # β_m L, free-free wavenumbers
        Z_psi   = ComplexF64.(payload.Z_psi),       # N×N radiation impedance, Ψ basis
        c_hydro = derived.d * fparams.rho * fparams.g,  # d ρ g (hydrostatic stiffness)
        F0      = fparams.motor_inertia * fparams.omega^2,
        forcing_width = fparams.forcing_width,
    )
end

# Solve the a-priori modal law  (D − Z_psi) q = −f  for given EI and xM.
#
# Step 1 — project the Gaussian motor load onto the Ψ basis:
#   f[m] = ⟨Ψ_m, F_motor⟩ = Σ_k Ψ_m(x_k) F(x_k) w_k    (quadrature)
#
# Step 2 — assemble the diagonal structural operator:
#   D[m] = EI (β_m/L)⁴  −  ρ_R ω²  +  d ρ g
#          ^^^^^^^^^^^^     ^^^^^^^     ^^^^^
#          bending          inertia     hydrostatic restoring
#   Note: payload.beta stores β_m * L, so (β_m/L)⁴ = (beta[m]/L)⁴ = beta[m]⁴ / L⁴.
#   The factor 1/L⁴ is absorbed into EI * beta^4 because beta was extracted at L=1
#   in the solver convention — verify against the modal.jl normalisation if in doubt.
#
# Step 3 — solve the linear system A_sys q = -f,  A_sys = Diagonal(D) - Z_psi.
function solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
    p = theory_ctx.params
    F_c = theory_ctx.derived.F_c
    L_c = theory_ctx.derived.L_c

    x_raft_adim = theory_ctx.x_raft ./ L_c
    loads_adim  = (theory_ctx.F0 / F_c) .*
                  Surferbot.gaussian_load(Float64(xM_norm), p.forcing_width, x_raft_adim)
    loads_dim   = loads_adim .* (F_c / L_c)           # dimensional load, N/m
    F_psi       = theory_ctx.Psi' * (loads_dim .* theory_ctx.weights)  # modal force f[m]

    # D[m] = EI β_m⁴ − ρ_R ω² + d ρ g   (diagonal structural-inertia-hydrostatic operator)
    D = ComplexF64.(EI .* theory_ctx.beta .^ 4
                    .- p.rho_raft * p.omega^2
                    .+ theory_ctx.c_hydro)
    A_sys = Diagonal(D) - theory_ctx.Z_psi   # full system matrix (D − Z_psi)
    return -(A_sys \ ComplexF64.(F_psi))      # q = −(D − Z_psi)⁻¹ f
end

# Compute S, A, η_end, η₁ from a Ψ-basis modal coefficient vector q.
# Mirrors get_roots_integral exactly so both paths are consistent.
function theoretical_endpoint_diagnostics(q, theory_ctx)
    S = zero(ComplexF64)
    A = zero(ComplexF64)
    for j in eachindex(theory_ctx.mode_numbers)
        if iseven(theory_ctx.mode_numbers[j])
            S += q[j] * theory_ctx.w_end[j]   # even mode: symmetric contribution
        else
            A += q[j] * theory_ctx.w_end[j]   # odd  mode: antisymmetric contribution
        end
    end
    eta_end = S + A    # η(x_right) = S + A
    eta_1   = S - A    # η(x_left)  = S − A  (odd modes flip sign across beam centre)
    return (; S, A, eta_1, eta_end)
end

# Discrete local-minimum detector with a ratio gate.
# A grid point i is accepted as a root only when:
#   (a) values[i] is a local minimum (≤ both neighbours), AND
#   (b) ratio[i] < ratio_cutoff  — the amplitude is genuinely small relative
#       to its complement (avoids reporting shallow troughs where both |S| and
#       |A| are large and the minimum is physically irrelevant).
function find_filtered_minima(xgrid, values, ratio; ratio_cutoff::Float64)
    roots = Float64[]
    for i in 2:(length(xgrid) - 1)
        if values[i] <= values[i - 1] && values[i] <= values[i + 1] && ratio[i] < ratio_cutoff
            push!(roots, Float64(xgrid[i]))
        end
    end
    return roots
end

const CURVE_NAMES  = ["S", "A", "eta_1", "eta_end"]
const CURVE_LABELS = [L"|S| = 0", L"|A| = 0", L"|\eta_1| = 0", L"|\eta_{\mathrm{end}}| = 0"]

function get_roots_theoretical(artifact, condition_name; output_dir::AbstractString)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    xM_grid = collect(range(0.0, 0.49, length=401))
    theory_ctx = theoretical_modal_context(params; output_dir=output_dir)

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for (iei, EI) in enumerate(EI_list)
        absS = Float64[]; absA = Float64[]
        abs_eta_1 = Float64[]; abs_eta_end = Float64[]

        for xM_norm in xM_grid
            q = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
            diag = theoretical_endpoint_diagnostics(q, theory_ctx)
            push!(absS, abs(diag.S))
            push!(absA, abs(diag.A))
            push!(abs_eta_1, abs(diag.eta_1))
            push!(abs_eta_end, abs(diag.eta_end))
        end

        roots = roots_for_condition(condition_name, xM_grid,
                                    absS, absA, abs_eta_1, abs_eta_end)

        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

# ─── Optional GPR background smoothing (disabled by default) ─────────────────
# When USE_GPR=true in main(), the raw α heatmap is replaced by a GP posterior
# mean on a denser grid. The kernel is squared-exponential with length scales
# chosen heuristically from the grid spacing.  Not used for publication figures.

function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector)
    n = length(values)
    mean_value = mean(values)
    centered = collect(Float64.(values .- mean_value))

    dx = diff(sort(unique(Float64.(x))))
    dy = diff(sort(unique(Float64.(y))))
    dx = dx[dx .> 0]
    dy = dy[dy .> 0]
    # Length scales: heuristic based on grid spacing
    lx = isempty(dx) ? 0.05 : max(0.02, 3 * median(dx))
    ly = isempty(dy) ? 0.15 : max(0.05, 3 * median(dy))
    sigma_f = max(std(values), 1e-3)
    noise = max(1e-4, 2e-2 * sigma_f)

    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        r2 = ((x[i] - x[j]) / lx)^2 + ((y[i] - y[j]) / ly)^2
        kij = sigma_f^2 * exp(-0.5 * r2)
        K[i, j] = kij
        K[j, i] = kij
    end
    for i in 1:n
        K[i, i] += noise^2 + 1e-10
    end

    F = cholesky(Symmetric(K))
    weights = F \ centered
    return (x = Float64.(x), y = Float64.(y), weights = weights, mean = mean_value, lx = lx, ly = ly, sigma_f2 = sigma_f^2, noise=noise)
end

function predict_gp2d(model, xq::Real, yq::Real)
    acc = 0.0
    for i in eachindex(model.weights)
        r2 = ((xq - model.x[i]) / model.lx)^2 + ((yq - model.y[i]) / model.ly)^2
        acc += model.weights[i] * (model.sigma_f2 * exp(-0.5 * r2))
    end
    return model.mean + acc
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    configs = [
        (name="unc_theo", coupled=false, source=:theoretical),
        (name="unc_int",  coupled=false, source=:integral),
        (name="cpl_theo", coupled=true,  source=:theoretical),
        (name="cpl_int",  coupled=true,  source=:integral)
    ]

    # Modular Toggle for GPR Smoothing
    USE_GPR = false 

    for cfg in configs
        println("Processing $(cfg.name)...")

        # Artifact still supplies base_params (for the κ shift) and the EI grid
        # over which the theoretical curve is computed. The α heatmap and the
        # integral overlay both come from the higher-resolution CSV so that
        # the heatmap field is self-consistent with the integral scatter.
        jld2_file = cfg.coupled ?
            "sweep_motor_position_EI_coupled_from_matlab.jld2" :
            "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
        artifact = load_sweep(joinpath(output_dir, "jld2", jld2_file))

        params = artifact.base_params
        # κ = EI / (ρ_R L⁴ ω²)  →  log₁₀(κ) = log₁₀(EI) − shift
        shift = log10(params.rho_raft * params.L_raft^4 * params.omega^2)

        # α heatmap from CSV (300 EI × 100 xM grid).
        # Using the CSV (not the JLD2 artifact) keeps the heatmap self-consistent
        # with the integral-scatter overlay — both come from the same solver runs.
        csv_file = cfg.coupled ?
            "sweeper_coupled_full_grid.csv" :
            "sweeper_uncoupled_full_grid.csv"
        csv_path = joinpath(output_dir, "csv", csv_file)
        df_heat = CSV.read(csv_path, DataFrame)
        logEI_axis = sort(unique(df_heat.log10_EI))
        xM_axis    = sort(unique(df_heat.xM_over_L))
        # alpha[i, j] = α(xM_axis[i], logEI_axis[j])
        alpha = zeros(Float64, length(xM_axis), length(logEI_axis))
        let row_lookup = Dict{Tuple{Float64,Float64}, Float64}(
                (row.log10_EI, row.xM_over_L) => row.alpha for row in eachrow(df_heat))
            for (j, le) in enumerate(logEI_axis), (i, xm) in enumerate(xM_axis)
                alpha[i, j] = row_lookup[(le, xm)]
            end
        end
        
        # --- Background GPR Smoothing ---
        local p_heatmap
        if USE_GPR
            println("  Fitting GPR background...")
            xtrain, ytrain, vtrain = Float64[], Float64[], Float64[]
            # Note: training feature space (x=logEI, y=xM)
            for (ie, lei) in enumerate(logEI_axis), (im, xm) in enumerate(xM_axis)
                push!(xtrain, lei)
                push!(ytrain, xm)
                push!(vtrain, alpha[im, ie])
            end
            model = fit_gp2d(xtrain, ytrain, vtrain)
            
            # Predict on a very dense grid for smoothness
            logEI_dense = collect(range(minimum(logEI_axis), maximum(logEI_axis), length=200))
            xM_dense = collect(range(minimum(xM_axis), maximum(xM_axis), length=200))
            alpha_dense = [predict_gp2d(model, lei, xm) for xm in xM_dense, lei in logEI_dense]
            
            # APPLY SHIFT TO HEATMAP AXIS
            p_heatmap = (logEI_dense .- shift, xM_dense, alpha_dense)
        else
            # APPLY SHIFT TO HEATMAP AXIS
            p_heatmap = (logEI_axis .- shift, xM_axis, alpha)
        end

        # Same 4 conditions in both paths so theoretical and integral plots line up directly.
        curve_names = CURVE_NAMES
        results = Dict()

        if cfg.source == :integral
            csv_file = cfg.coupled ? "sweeper_coupled_full_grid.csv" : "sweeper_uncoupled_full_grid.csv"
            csv_path = joinpath(output_dir, "csv", csv_file)
            for cname in curve_names
                res = get_roots_integral(csv_path, cname)
                # APPLY SHIFT TO SCATTER RESULTS
                results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
            end
        else
            for cname in curve_names
                res = get_roots_theoretical(artifact, cname; output_dir=output_dir)
                # APPLY SHIFT TO SCATTER RESULTS
                results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
            end
        end
        
        # ── Publication styling ───────────────────────────────────────────
        # Okabe–Ito colorblind-safe palette; pick four mutually distinct hues
        # that sit far from the red/blue heatmap extremes for visibility.
        okabe_ito  = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                      "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
                        # black,         orange,       green,        pink
        markers      = [:circle, :rect, :diamond, :utriangle]
        labels       = CURVE_LABELS

        # κ-window of physical interest; clip to CSV data extent so the panel
        # has no empty strip past the largest sampled κ.
        max_logK_data = maximum(logEI_axis) - shift
        XLIMS = (-4.0, max_logK_data)
        YLIMS = (0.0, 0.5)

        # Plot options shared by both heatmap and scatter calls.
        plt_opts = (
            xlabel = L"\log_{10}\,\kappa",
            ylabel = L"x_M / L",
            colormap = :balance,
            clims = (-1, 1),
            levels = 51,
            interpolate = true,
            xlims = XLIMS,
            ylims = YLIMS,
            legend = :topright,
            background_color_legend = RGBA(1, 1, 1, 0.85),
            foreground_color_legend = :black,
            legend_font_halign = :left,
            size = (820, 640),
            margin = 6Plots.mm,
            dpi = 220,
            titlefontsize = 14,
            guidefontsize = 14,
            tickfontsize = 12,
            legendfontsize = 11,
            fontfamily = "Computer Modern",
            framestyle = :box,
            grid = false,
            colorbar_title = L"\alpha",
            colorbar_titlefontsize = 14,
            colorbar_tickfontsize = 11,
        )

        # Title: keep prose outside math mode (GR's math renderer eats spaces
        # inside \textrm{...}); only Λ goes through math mode.
        adj         = cfg.coupled ? "Coupled" : "Uncoupled"
        method_str  = cfg.source == :theoretical ? "a-priori prediction" : "direct integration"
        Lambda_val  = cfg.coupled ? @sprintf("%.2f", params.d / params.L_raft) : "0"
        fig_title   = LaTeXString(
            "$adj raft, \$\\Lambda = $Lambda_val\$ — $method_str")

        p = heatmap(p_heatmap[1], p_heatmap[2], p_heatmap[3];
                    title = fig_title, plt_opts...)

        for (i, cname) in enumerate(curve_names)
            res = results[cname]
            # Restrict scatter to the publication κ-window.
            mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
                   (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
            isempty(res.logK[mask]) && continue
            scatter!(p, res.logK[mask], res.xM_norm[mask];
                     label = labels[i],
                     color = curve_colors[i],
                     marker = markers[i],
                     markersize = 5,
                     markerstrokewidth = 0.6,
                     markerstrokecolor = :white,
                     markeralpha = 0.95)
        end
        
        fig_dir = joinpath(output_dir, "figures")
        out_pdf = joinpath(fig_dir, "plot_dimensionless_diagnostics_$(cfg.name).pdf")
        out_png = joinpath(fig_dir, "plot_dimensionless_diagnostics_$(cfg.name).png")
        savefig(p, out_pdf)
        savefig(p, out_png)
        println("Saved $out_pdf")
        println("Saved $out_png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
