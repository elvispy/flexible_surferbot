"""
    PrescribedWnDiagonalImpedance

Experiment for identifying the empirical modal pressure map on the wetted strip.

Paper terminology:
- `\\hat{q}_n`: modal displacement coefficients
- `\\hat{p}_m`: modal pressure coefficients obtained by projecting `d * \\hat{p}_{dyn}`
  onto the beam basis
- `Z_{mn}(\\omega)`: modal map from displacement coefficients to modal pressure
  coefficients in this experiment

This file uses the codebase's sampled raw free-free basis built by
`build_raw_freefree_basis`. The paper draft writes the fluid operator in
velocity-impedance form, but this experiment composes that with the harmonic
kinematics and identifies the displacement-to-pressure map directly:

    p_modal = Z * q_modal

for a fixed `Surferbot.FlexibleParams`.

This is not a Korobkin comparison script. Its purpose is:
1. Prescribe one raw basis displacement shape at a time.
2. Solve the reduced fluid problem.
3. Project the resulting dynamic pressure back onto the same basis.
4. Assemble the empirical modal map `Z`.
5. Validate that map against the original forced problem for the same parameters.

Pressure convention note:
`flexible_solver` returns `result.pressure` using the postprocess convention
`p = p_dyn - rho * g * eta` (real hydrostatic, matching the paper Bernoulli
under the e^{i ω t} time convention). This experiment identifies the map
for `p_dyn`, so the forced-solve validation removes the hydrostatic term in
that convention.
"""
module PrescribedWnDiagonalImpedance

using JLD2
using LinearAlgebra
using Printf
using SHA
using SparseArrays
using Surferbot

const DEFAULT_OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "output"))
const DEFAULT_SWEEP_FILE = "sweep_motor_position_EI_coupled_from_matlab.jld2"
const DEFAULT_CACHE_FILE = "prescribed_wn_diagonal_impedance.jld2"
const DEFAULT_OPERATOR_CACHE_FILE = "modal_pressure_maps.jld2"
const DEFAULT_NUM_MODES_BASIS = 8

function sweep_path(output_dir::AbstractString, sweep_file::AbstractString)
    return joinpath(output_dir, "jld2", sweep_file)
end

function cache_path(output_dir::AbstractString, cache_file::AbstractString)
    return joinpath(output_dir, "jld2", cache_file)
end

"""
Bumped from 1 → 2 when the postprocess Bernoulli sign convention was corrected
(p_dyn = -iωρφ + 2ρν φ_xx, real hydrostatic). Old v1 cache entries are
automatically bypassed because their SHA-1 key no longer matches.
Bumped from 2 → 3 to add domain-end radiation amplitudes `a_vec` (η at x=+ℓ
per prescribed mode, Ψ-basis) to the slim payload. Needed for LH asymmetry.
"""
const PRESSURE_CONVENTION_VERSION = 3

function operator_signature(params::Surferbot.FlexibleParams; num_modes_basis::Int=DEFAULT_NUM_MODES_BASIS)
    derived = derive_params(params)
    return (
        omega = params.omega,
        L_raft = params.L_raft,
        d = derived.d,
        rho = params.rho,
        g = params.g,
        nu = params.nu,
        sigma = params.sigma,
        rho_raft = params.rho_raft,
        domain_depth = derived.domain_depth,
        L_domain = derived.L_domain,
        n = derived.n,
        M = derived.M,
        ooa = params.ooa,
        bc = params.bc,
        num_modes_basis = num_modes_basis,
        convention_version = PRESSURE_CONVENTION_VERSION,
    )
end

function operator_cache_key(params::Surferbot.FlexibleParams; num_modes_basis::Int=DEFAULT_NUM_MODES_BASIS)
    return "operator_" * bytes2hex(sha1(repr(operator_signature(params; num_modes_basis=num_modes_basis))))
end

function params_snapshot(params::Surferbot.FlexibleParams)
    names = fieldnames(typeof(params))
    values = Tuple(getfield(params, name) for name in names)
    return NamedTuple{names}(values)
end

function cache_key(params::Surferbot.FlexibleParams, mode_labels, num_modes_basis::Int)
    signature = (
        params = params_snapshot(params),
        mode_labels = collect(Int.(mode_labels)),
        num_modes_basis = Int(num_modes_basis),
    )
    return "case_" * bytes2hex(sha1(repr(signature)))
end

function load_cached_result(path::AbstractString, key::AbstractString)
    isfile(path) || return nothing
    return jldopen(path, "r") do io
        haskey(io, key) ? read(io, key) : nothing
    end
end

function save_cached_result(path::AbstractString, key::AbstractString, payload)
    mkpath(dirname(path))
    jldopen(path, "a+") do io
        io[key] = payload
    end
end

function build_surface_args(params::Surferbot.FlexibleParams, derived)
    return (
        sigma = params.sigma,
        rho = params.rho,
        omega = params.omega,
        nu = params.nu,
        g = params.g,
        L_raft = params.L_raft,
        d = derived.d,
        EI = params.EI,
        rho_raft = params.rho_raft,
        nd_groups = derived.nd_groups,
        x_contact = derived.x_contact,
        x = derived.x .* derived.L_c,
        loads = derived.loads .* derived.F_c ./ derived.L_c,
        motor_position = params.motor_position,
        N = derived.N,
        M = derived.M,
        dx = derived.dx .* derived.L_c,
        dz = derived.dz .* derived.L_c,
        t_c = derived.t_c,
        L_c = derived.L_c,
        m_c = derived.m_c,
        k = derived.k,
        ooa = params.ooa,
    )
end

function raw_basis_context(params::Surferbot.FlexibleParams, derived; num_modes_basis::Int)
    x_raft = collect(Float64.(derived.x[derived.x_contact] .* derived.L_c))
    basis = build_raw_freefree_basis(x_raft, params.L_raft; num_modes=num_modes_basis)
    gram = Matrix{Float64}(basis.Phi' * (basis.Phi .* basis.w))
    return (
        basis = basis,
        x_raft = x_raft,
        weights = collect(Float64.(basis.w)),
        gram = gram,
        gram_cond = Float64(basis.gram_cond),
        gram_error_inf = Float64(norm(gram - I, Inf)),
    )
end

function psi_basis_context(basis_ctx)
    Psi, keep = weighted_mgs(basis_ctx.basis.Phi, basis_ctx.weights)
    all(keep) || error("Psi conversion dropped raw basis columns; expected matching spans.")
    gram = Matrix{Float64}(Psi' * (Psi .* basis_ctx.weights))
    return (
        Psi = Matrix{Float64}(Psi),
        gram = gram,
        gram_error_inf = Float64(norm(gram - I, Inf)),
    )
end

function basis_transforms(basis_ctx, psi_ctx)
    Phi = basis_ctx.basis.Phi
    w = basis_ctx.weights
    raw_from_psi = basis_ctx.gram \ (Phi' * (psi_ctx.Psi .* w))
    psi_from_raw = psi_ctx.gram \ (psi_ctx.Psi' * (Phi .* w))
    return (
        raw_from_psi = Matrix{Float64}(raw_from_psi),
        psi_from_raw = Matrix{Float64}(psi_from_raw),
    )
end

function resolve_mode_labels(basis_ctx, mode_labels)
    available = collect(Int.(basis_ctx.basis.n))
    labels = isnothing(mode_labels) ? available : collect(Int.(mode_labels))
    all(in(available), labels) || error("Requested mode labels $(labels) are not all present in the retained raw basis $(available).")
    indices = [findfirst(==(label), available) for label in labels]
    return labels, collect(Int.(indices))
end

function project_modal_displacement(basis_ctx, eta_contact::AbstractVector{<:Complex})
    return ComplexF64.(basis_ctx.gram \ (basis_ctx.basis.Phi' * (ComplexF64.(eta_contact) .* basis_ctx.weights)))
end

function project_modal_pressure(basis_ctx, d::Real, p_dyn_contact::AbstractVector{<:Complex})
    rhs = basis_ctx.basis.Phi' * ((d .* ComplexF64.(p_dyn_contact)) .* basis_ctx.weights)
    return ComplexF64.(basis_ctx.gram \ rhs)
end

function prescribed_target(basis_ctx, params::Surferbot.FlexibleParams, derived, mode_label::Int)
    labels = basis_ctx.basis.n
    col = findfirst(==(mode_label), labels)
    col === nothing && error("Requested mode label $mode_label is not present in the raw basis.")

    eta_target = ComplexF64.(basis_ctx.basis.Phi[:, col])
    eta_adim = eta_target ./ derived.L_c
    phi_z_target = ComplexF64.(im * params.omega * derived.t_c .* eta_adim)

    q_target = project_modal_displacement(basis_ctx, eta_target)
    e_col = zeros(ComplexF64, size(basis_ctx.basis.Phi, 2))
    e_col[col] = 1 + 0im

    return (
        mode_label = mode_label,
        column_index = col,
        beta_n = Float64(basis_ctx.basis.beta[col]),
        eta_target = eta_target,
        phi_z_target = phi_z_target,
        q_target = q_target,
        q_target_error = Float64(norm(q_target - e_col)),
    )
end

function build_reduced_system(assembled, phi_z_target::AbstractVector{<:Complex})
    NP = assembled.derived.N * assembled.derived.M
    idx_contact = assembled.indices.idxContact
    length(idx_contact) == length(phi_z_target) || error("Contact target length mismatch.")

    kept_rows = vcat(setdiff(1:NP, idx_contact), (NP + 1):(2 * NP))
    prescribed_cols = NP .+ idx_contact
    kept_cols = setdiff(1:(2 * NP), prescribed_cols)

    A_full = assembled.A[1:(2 * NP), 1:(2 * NP)]
    b_full = assembled.b[1:(2 * NP)]
    rhs_shift = A_full[kept_rows, prescribed_cols] * ComplexF64.(phi_z_target)

    A = copy(A_full[kept_rows, kept_cols])
    b = copy(b_full[kept_rows] - rhs_shift)
    dropzeros!(A)

    return (
        A = A,
        b = b,
        kept_rows = kept_rows,
        kept_cols = kept_cols,
        prescribed_cols = prescribed_cols,
        diagnostics = (
            reduced_size = size(A),
            row_count = size(A, 1),
            column_count = size(A, 2),
            fluid_unknown_count = 2 * NP - length(idx_contact),
            prescribed_count = length(idx_contact),
            is_square = size(A, 1) == size(A, 2) == (2 * NP - length(idx_contact)),
            contact_constraint_block = :phi_z,
        ),
    )
end

function solve_prescribed_mode(assembled, reduced, phi_z_target::AbstractVector{<:Complex})
    NP = assembled.derived.N * assembled.derived.M
    solution = solve_tensor_system(reduced.A, reduced.b)
    full_solution = zeros(ComplexF64, 2 * NP)
    full_solution[reduced.kept_cols] = solution
    full_solution[reduced.prescribed_cols] = ComplexF64.(phi_z_target)

    phi = reshape(full_solution[1:NP], assembled.derived.M, assembled.derived.N)
    phi_z = reshape(full_solution[(NP + 1):(2 * NP)], assembled.derived.M, assembled.derived.N)
    return phi, phi_z
end

function reconstruct_dynamic_fields(params::Surferbot.FlexibleParams, derived, phi, phi_z)
    args = build_surface_args(params, derived)
    contact = collect(Bool.(derived.x_contact))
    Nr = count(contact)
    dx_adim = args.dx / args.L_c
    D2r = Matrix(getNonCompactFDmatrix(Nr, 1.0, 2, args.ooa)) / dx_adim^2

    eta_adim = (1 / (im * args.omega * args.t_c)) .* ComplexF64.(vec(phi_z[end, :]))
    eta = eta_adim .* args.L_c

    phi_raft = ComplexF64.(vec(phi[end, :])[contact])
    eta_contact = ComplexF64.(eta[contact])
    p_dyn_adim = -(im * args.nd_groups.Gamma) .* phi_raft .+
                 (2 * args.nd_groups.Gamma / args.nd_groups.Re) .* (D2r * phi_raft)
    p_dyn = p_dyn_adim .* derived.F_c ./ derived.L_c^2

    return (
        eta = collect(ComplexF64.(eta)),
        eta_contact = eta_contact,
        p_dyn = collect(ComplexF64.(p_dyn)),
    )
end

function prescribed_column_payload(params::Surferbot.FlexibleParams, assembled, basis_ctx, mode_label::Int)
    target = prescribed_target(basis_ctx, params, assembled.derived, mode_label)
    reduced = build_reduced_system(assembled, target.phi_z_target)
    phi, phi_z = solve_prescribed_mode(assembled, reduced, target.phi_z_target)
    fields = reconstruct_dynamic_fields(params, assembled.derived, phi, phi_z)
    p_modal = project_modal_pressure(basis_ctx, assembled.derived.d, fields.p_dyn)

    other_idx = setdiff(1:length(p_modal), [target.column_index])
    offdiag_ratio = norm(p_modal[other_idx]) / max(abs(p_modal[target.column_index]), eps())
    eta_contact_relerr = norm(fields.eta_contact - target.eta_target) / max(norm(target.eta_target), eps())

    return (
        mode_label = mode_label,
        column_index = target.column_index,
        beta_n = target.beta_n,
        prescribed_eta = collect(ComplexF64.(target.eta_target)),
        q_target = collect(ComplexF64.(target.q_target)),
        q_target_error = target.q_target_error,
        phi = Matrix{ComplexF64}(phi),
        phi_z = Matrix{ComplexF64}(phi_z),
        eta = collect(ComplexF64.(fields.eta)),
        p_dyn = collect(ComplexF64.(fields.p_dyn)),
        p_modal = collect(ComplexF64.(p_modal)),
        p_diag = ComplexF64(p_modal[target.column_index]),
        offdiag_ratio = Float64(offdiag_ratio),
        eta_contact_relerr = Float64(eta_contact_relerr),
        system_diagnostics = reduced.diagnostics,
        a_n       = ComplexF64(fields.eta[end]),
        a_n_left  = ComplexF64(fields.eta[1]),
    )
end

function dynamic_pressure_from_forced_result(result)
    args = result.metadata.args
    contact = collect(Bool.(args.x_contact))
    eta_contact = ComplexF64.(result.eta[contact])
    p_total_contact = ComplexF64.(result.pressure)
    # postprocess.jl now uses the corrected convention p = p_dyn - rho*g*eta
    # (real hydrostatic). Recover p_dyn by adding back the hydrostatic term.
    p_dyn_contact = p_total_contact .+ args.rho .* args.g .* eta_contact
    return eta_contact, p_dyn_contact
end

function validate_pressure_map(params::Surferbot.FlexibleParams, pressure_map_result)
    result = flexible_solver(params)
    basis_ctx = pressure_map_result.basis
    mode_labels = pressure_map_result.mode_labels
    mode_indices = pressure_map_result.mode_indices

    eta_contact, p_dyn_contact = dynamic_pressure_from_forced_result(result)
    q_modal_full = project_modal_displacement(basis_ctx, eta_contact)
    p_modal_full = project_modal_pressure(basis_ctx, params.d, p_dyn_contact)

    q_modal = q_modal_full[mode_indices]
    p_modal = p_modal_full[mode_indices]
    p_modal_pred = pressure_map_result.Z * q_modal
    residual = p_modal - p_modal_pred

    q_tail_ratio = norm(deleteat!(collect(q_modal_full), mode_indices)) / max(norm(q_modal_full), eps())
    p_tail_ratio = norm(deleteat!(collect(p_modal_full), mode_indices)) / max(norm(p_modal_full), eps())

    return (
        q_modal = collect(ComplexF64.(q_modal)),
        p_modal = collect(ComplexF64.(p_modal)),
        p_modal_pred = collect(ComplexF64.(p_modal_pred)),
        residual = collect(ComplexF64.(residual)),
        residual_relerr = Float64(norm(residual) / max(norm(p_modal), eps())),
        q_tail_ratio = Float64(q_tail_ratio),
        p_tail_ratio = Float64(p_tail_ratio),
        forced_result = (
            eta = collect(ComplexF64.(result.eta)),
            pressure = collect(ComplexF64.(result.pressure)),
        ),
    )
end

function empirical_modal_pressure_map(
    params::Surferbot.FlexibleParams;
    mode_labels=nothing,
    num_modes_basis::Int=DEFAULT_NUM_MODES_BASIS,
    validate::Bool=true,
    keep_columns::Bool=true,
)
    assembled = assemble_flexible_system(params)
    basis_ctx = raw_basis_context(params, assembled.derived; num_modes_basis=num_modes_basis)
    labels, indices = resolve_mode_labels(basis_ctx, mode_labels)

    column_payloads = [prescribed_column_payload(params, assembled, basis_ctx, label) for label in labels]
    Z = hcat([payload.p_modal[indices] for payload in column_payloads]...)
    Z_diag = ComplexF64[payload.p_diag for payload in column_payloads]
    offdiag_ratio = Float64[payload.offdiag_ratio for payload in column_payloads]
    a_vec_raw      = ComplexF64[payload.a_n      for payload in column_payloads]
    a_vec_left_raw = ComplexF64[payload.a_n_left for payload in column_payloads]

    result = (
        params = params_snapshot(params),
        Fr2 = Float64(params.L_raft * params.omega^2 / params.g),
        mode_labels = labels,
        mode_indices = indices,
        basis = basis_ctx,
        raw_basis = (
            n = collect(Int.(basis_ctx.basis.n)),
            mode_type = collect(String.(basis_ctx.basis.mode_type)),
            betaL = collect(Float64.(basis_ctx.basis.betaL)),
            beta = collect(Float64.(basis_ctx.basis.beta)),
            x_raft = basis_ctx.x_raft,
            weights = basis_ctx.weights,
            Phi = Matrix{Float64}(basis_ctx.basis.Phi),
            gram = Matrix{Float64}(basis_ctx.gram),
            gram_cond = basis_ctx.gram_cond,
            gram_error_inf = basis_ctx.gram_error_inf,
        ),
        Z = Matrix{ComplexF64}(Z),
        Z_diag = collect(ComplexF64.(Z_diag)),
        offdiag_ratio = offdiag_ratio,
        a_vec_raw      = collect(ComplexF64.(a_vec_raw)),
        a_vec_left_raw = collect(ComplexF64.(a_vec_left_raw)),
        columns = keep_columns ? column_payloads : NamedTuple[],
    )

    if validate
        validation_payload = validate_pressure_map(params, result)
        return merge(result, (validation = validation_payload,))
    end
    return result
end

function slim_modal_pressure_map(result; cache_version::Int=1)
    psi_ctx = psi_basis_context(result.basis)
    transforms = basis_transforms(result.basis, psi_ctx)
    Z_raw = Matrix{ComplexF64}(result.Z)
    Z_psi = ComplexF64.(transforms.psi_from_raw * Z_raw * transforms.raw_from_psi)
    selected_beta = Float64[result.raw_basis.beta[idx] for idx in result.mode_indices]
    # a_vec: η(+ℓ) per mode in Ψ basis.  Transformation: a^Ψ = (T_{Φ←Ψ})ᵀ a^Φ
    # because η(+ℓ) = (a^Φ)ᵀ q^Φ = (a^Φ)ᵀ (raw_from_psi q^Ψ) = ((raw_from_psi)ᵀ a^Φ)ᵀ q^Ψ
    T = transforms.raw_from_psi                                   # real N×N
    a_vec      = ComplexF64.(transpose(T) * result.a_vec_raw)
    a_vec_left = ComplexF64.(transpose(T) * result.a_vec_left_raw)
    return (
        cache_version = cache_version,
        params = result.params,
        Fr2 = result.Fr2,
        mode_labels = collect(Int.(result.mode_labels)),
        mode_indices = collect(Int.(result.mode_indices)),
        beta = selected_beta,
        x_raft = result.raw_basis.x_raft,
        weights = result.raw_basis.weights,
        raw_basis = (
            Phi = Matrix{Float64}(result.raw_basis.Phi),
            gram = Matrix{Float64}(result.raw_basis.gram),
            gram_error_inf = result.raw_basis.gram_error_inf,
        ),
        psi_basis = (
            Psi = psi_ctx.Psi,
            gram = psi_ctx.gram,
            gram_error_inf = psi_ctx.gram_error_inf,
        ),
        transforms = (
            raw_from_psi = transforms.raw_from_psi,
            psi_from_raw = transforms.psi_from_raw,
        ),
        Z_raw = Z_raw,
        Z_psi = Matrix{ComplexF64}(Z_psi),
        offdiag_ratio_raw = collect(Float64.(result.offdiag_ratio)),
        a_vec      = collect(ComplexF64.(a_vec)),
        a_vec_left = collect(ComplexF64.(a_vec_left)),
    )
end

function zero_modal_pressure_map(params::Surferbot.FlexibleParams; num_modes_basis::Int=DEFAULT_NUM_MODES_BASIS)
    assembled = assemble_flexible_system(params)
    basis_ctx = raw_basis_context(params, assembled.derived; num_modes_basis=num_modes_basis)
    labels, indices = resolve_mode_labels(basis_ctx, nothing)
    raw_result = (
        params = params_snapshot(params),
        Fr2 = Float64(params.L_raft * params.omega^2 / params.g),
        mode_labels = labels,
        mode_indices = indices,
        basis = basis_ctx,
        raw_basis = (
            n = collect(Int.(basis_ctx.basis.n)),
            mode_type = collect(String.(basis_ctx.basis.mode_type)),
            betaL = collect(Float64.(basis_ctx.basis.betaL)),
            beta = collect(Float64.(basis_ctx.basis.beta)),
            x_raft = basis_ctx.x_raft,
            weights = basis_ctx.weights,
            Phi = Matrix{Float64}(basis_ctx.basis.Phi),
            gram = Matrix{Float64}(basis_ctx.gram),
            gram_cond = basis_ctx.gram_cond,
            gram_error_inf = basis_ctx.gram_error_inf,
        ),
        Z = zeros(ComplexF64, length(indices), length(indices)),
        offdiag_ratio = zeros(Float64, length(indices)),
        a_vec_raw      = zeros(ComplexF64, length(indices)),
        a_vec_left_raw = zeros(ComplexF64, length(indices)),
    )
    return slim_modal_pressure_map(raw_result)
end

function load_or_compute_modal_pressure_map(
    params::Surferbot.FlexibleParams;
    output_dir::AbstractString=DEFAULT_OUTPUT_DIR,
    cache_file::AbstractString=DEFAULT_OPERATOR_CACHE_FILE,
    num_modes_basis::Int=DEFAULT_NUM_MODES_BASIS,
)
    key = operator_cache_key(params; num_modes_basis=num_modes_basis)
    cache_file_path = cache_path(output_dir, cache_file)
    cached = load_cached_result(cache_file_path, key)
    if !isnothing(cached)
        return merge(cached, (cache_status = (loaded = true, path = cache_file_path, key = key),))
    end

    derived = derive_params(params)
    payload = if iszero(derived.d)
        zero_modal_pressure_map(params; num_modes_basis=num_modes_basis)
    else
        result = empirical_modal_pressure_map(
            params;
            num_modes_basis=num_modes_basis,
            validate=false,
            keep_columns=false,
        )
        slim_modal_pressure_map(result)
    end
    save_cached_result(cache_file_path, key, payload)
    return merge(payload, (cache_status = (loaded = false, path = cache_file_path, key = key),))
end

function default_params_from_artifact(;
    output_dir::AbstractString=DEFAULT_OUTPUT_DIR,
    sweep_file::AbstractString=DEFAULT_SWEEP_FILE,
)
    artifact = load_sweep(sweep_path(output_dir, sweep_file))
    return apply_parameter_overrides(artifact.base_params, (;))
end

function print_map_summary(result)
    println("mode      beta_n        Fr^2      real(Z_nn)      imag(Z_nn)   offdiag   ||G-I||inf")
    for (j, mode_label) in enumerate(result.mode_labels)
        @printf(
            "%4d  %10.4f  %10.4f  % .6e  % .6e  %.3e  %.3e\n",
            mode_label,
            result.raw_basis.beta[result.mode_indices[j]],
            result.Fr2,
            real(result.Z_diag[j]),
            imag(result.Z_diag[j]),
            result.offdiag_ratio[j],
            result.raw_basis.gram_error_inf,
        )
    end
end

function print_validation_summary(result)
    validation = result.validation
    println()
    @printf("forced-problem modal pressure validation: relerr = %.3e\n", validation.residual_relerr)
    @printf("discarded modal displacement tail ratio = %.3e\n", validation.q_tail_ratio)
    @printf("discarded modal pressure tail ratio     = %.3e\n", validation.p_tail_ratio)
    println("mode      real(p_m)       imag(p_m)     real((Zq)_m)    imag((Zq)_m)     relerr")
    for (j, mode_label) in enumerate(result.mode_labels)
        obs = validation.p_modal[j]
        pred = validation.p_modal_pred[j]
        relerr = abs(obs - pred) / max(abs(obs), eps())
        @printf(
            "%4d  % .6e  % .6e  % .6e  % .6e  %.3e\n",
            mode_label,
            real(obs),
            imag(obs),
            real(pred),
            imag(pred),
            relerr,
        )
    end
end

function main(;
    output_dir::AbstractString=DEFAULT_OUTPUT_DIR,
    sweep_file::AbstractString=DEFAULT_SWEEP_FILE,
    cache_file::AbstractString=DEFAULT_CACHE_FILE,
    mode_labels=nothing,
    num_modes_basis::Int=DEFAULT_NUM_MODES_BASIS,
    use_cache::Bool=true,
)
    params = default_params_from_artifact(; output_dir=output_dir, sweep_file=sweep_file)
    labels_hint = isnothing(mode_labels) ? collect(0:(num_modes_basis - 1)) : collect(Int.(mode_labels))
    key = cache_key(params, labels_hint, num_modes_basis)
    cache_file_path = cache_path(output_dir, cache_file)

    result = if use_cache
        cached = load_cached_result(cache_file_path, key)
        isnothing(cached) ? nothing : cached
    else
        nothing
    end

    if isnothing(result)
        result = empirical_modal_pressure_map(
            params;
            mode_labels=mode_labels,
            num_modes_basis=num_modes_basis,
        )
        save_cached_result(cache_file_path, key, result)
        result = merge(result, (cache_status = (loaded = false, path = cache_file_path),))
    else
        result = merge(result, (cache_status = (loaded = true, path = cache_file_path),))
    end

    print_map_summary(result)
    print_validation_summary(result)
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    PrescribedWnDiagonalImpedance.main()
end
