module Surferbot

using SparseArrays
using LinearAlgebra
using ForwardDiff

include("fd.jl")
include("integration.jl")
include("analysis.jl")
include("modal.jl")
include("postprocess.jl")
include("sweep.jl")
include("migration.jl")
include("video.jl")
include("utils.jl")

using .Analysis: beam_asymmetry,
                 beam_edge_metrics,
                 default_coupled_motor_position_EI_sweep,
                 default_uncoupled_motor_position_EI_sweep,
                 symmetric_antisymmetric_ratio,
                 extract_lowest_beam_curve
using .FD: getNonCompactFDMWeights, getNonCompactFDmatrix, getNonCompactFDmatrix2D
using .Integration: simpson_weights
using .Modal: ModalDecomposition,
              RawFreefreeBasis,
              build_raw_freefree_basis,
              psi_to_w_transform,
              coefficients_in_w_basis,
              decompose_raft_freefree_modes,
              trapz_weights,
              freefree_betaL_roots,
              freefree_mode_shape,
              weighted_mgs
using .Migration: matlab_motor_position_ei_sources,
                  load_motor_position_ei_export,
                  artifact_from_motor_position_ei_export
using .PostProcess: calculate_surferbot_outputs
using .Sweep: SweepSummary,
              SweepArtifact,
              apply_parameter_overrides,
              expand_parameter_grid,
              summarize_result,
              sweep_parameters,
              save_sweep,
              load_sweep
using .RunVideo: SurferbotRunRecord,
                 normalize_run,
                 write_provenance_json,
                 render_surferbot_run
using .Utils: dispersion_k, gaussian_load, solve_tensor_system

export FlexibleParams,
       FlexibleResult,
       getNonCompactFDMWeights,
       getNonCompactFDmatrix,
       getNonCompactFDmatrix2D,
       simpson_weights,
       beam_asymmetry,
       beam_edge_metrics,
       default_coupled_motor_position_EI_sweep,
       default_uncoupled_motor_position_EI_sweep,
       symmetric_antisymmetric_ratio,
       extract_lowest_beam_curve,
       matlab_motor_position_ei_sources,
       load_motor_position_ei_export,
       artifact_from_motor_position_ei_export,
       ModalDecomposition,
       RawFreefreeBasis,
       build_raw_freefree_basis,
       psi_to_w_transform,
       coefficients_in_w_basis,
       decompose_raft_freefree_modes,
       trapz_weights,
       freefree_betaL_roots,
       freefree_mode_shape,
       weighted_mgs,
       SweepSummary,
       SweepArtifact,
       apply_parameter_overrides,
       expand_parameter_grid,
       summarize_result,
       sweep_parameters,
       save_sweep,
       load_sweep,
       SurferbotRunRecord,
       normalize_run,
       write_provenance_json,
       render_surferbot_run,
       dispersion_k,
       gaussian_load,
       solve_tensor_system,
       derive_params,
       assemble_flexible_system,
       flexible_solver,
       flexible_surferbot_v2_julia

"""
    FlexibleParams{T<:Real}

Physical, geometric, and numerical parameters for the flexible Surferbot solver.

# Fields
- `sigma`: Surface tension.
- `rho`: Fluid density.
- `omega`: Angular frequency of forcing.
- `nu`: Kinematic viscosity.
- `g`: Gravitational acceleration.
- `L_raft`: Length of the raft.
- `motor_position`: Horizontal position of the actuator.
- `d`: Beam width or thickness parameter.
- `EI`: Flexural rigidity.
- `rho_raft`: Linear mass density of the raft.
- `L_domain`: Length of the fluid domain.
- `domain_depth`: Depth of the fluid domain.
- `n`: Number of grid points on the raft.
- `M`: Number of grid points in the vertical direction.
- `ooa`: Order of accuracy for finite differences.
- `motor_inertia`: Inertia of the actuator motor.
- `motor_force`: Specified forcing amplitude.
- `forcing_width`: Width of the Gaussian forcing profile.
- `bc`: Boundary condition symbol (`:radiative` or `:neumann`).
"""
Base.@kwdef struct FlexibleParams{T<:Real}
    sigma::T = 72.2e-3
    rho::T = 1000.0
    omega::T = 2 * pi * 80.0
    nu::T = 1e-6
    g::T = 9.81
    L_raft::T = 0.05
    motor_position::T = 0.6 / 5 * 0.05
    d::Union{Nothing, T} = 0.03
    EI::Union{T, AbstractVector{T}} = 3.0e9 * 3e-2 * 9e-4^3 / 12
    rho_raft::Union{T, AbstractVector{T}} = 0.052
    L_domain::Union{Nothing, T} = nothing
    domain_depth::Union{Nothing, T} = nothing
    n::Union{Nothing, Int} = nothing
    M::Union{Nothing, Int} = nothing
    ooa::Int = 4
    motor_inertia::T = 0.13e-3 * 2.5e-3
    motor_force::Union{Nothing, T} = nothing
    forcing_width::T = 0.05
    bc::Symbol = :radiative
end

"""
    FlexibleResult{T<:Real}

Primary outputs of the flexible Surferbot solve.

# Fields
- `U`: Drift speed.
- `power`: Mean actuator power.
- `thrust`: Mean wave-driven thrust.
- `x`: Horizontal grid coordinates.
- `z`: Vertical grid coordinates.
- `phi`: Potential field matrix.
- `phi_z`: Vertical derivative of potential.
- `eta`: Reconstructed free-surface elevation.
- `pressure`: Reconstructed pressure field.
- `max_curvature`: Maximum dimensionless raft curvature.
- `wave_steepness`: Maximum wave steepness.
- `metadata`: NamedTuple containing arguments and optional system matrix.
"""
struct FlexibleResult{T<:Real}
    U::T
    power::T
    thrust::T
    x::Vector{T}
    z::Vector{T}
    phi::Matrix{Complex{T}}
    phi_z::Matrix{Complex{T}}
    eta::Vector{Complex{T}}
    pressure::Vector{Complex{T}}
    max_curvature::T
    wave_steepness::T
    metadata::NamedTuple
end

"""
    derive_params(params::FlexibleParams)

Calculate derived scales, nondimensional groups, and grid parameters.

# Arguments
- `params`: Input configuration object.

# Returns
- A NamedTuple containing all derived quantities and grids.
"""
function derive_params(params::FlexibleParams{T}) where {T<:Real}
    motor_force = isnothing(params.motor_force) ? params.motor_inertia * params.omega^2 : params.motor_force
    d = isnothing(params.d) ? T(0.05) : params.d
    motor_position = clamp(params.motor_position, -params.L_raft / 2, params.L_raft / 2)

    L_c = params.L_raft
    t_c = 1 / params.omega
    rho_raft_scalar = params.rho_raft isa AbstractVector ? minimum(params.rho_raft) : params.rho_raft
    m_c = rho_raft_scalar * L_c
    F_c = m_c * L_c / t_c^2

    EI_scalar = params.EI isa AbstractVector ? minimum(params.EI) : params.EI
    nd_groups = (
        Gamma = params.rho * params.L_raft^2 / rho_raft_scalar,
        Fr = sqrt(params.L_raft * params.omega^2 / params.g),
        Re = params.L_raft^2 * params.omega / params.nu,
        kappa = EI_scalar / (rho_raft_scalar * params.L_raft^4 * params.omega^2),
        We = rho_raft_scalar * params.L_raft * params.omega^2 / params.sigma,
        Lambda = d / params.L_raft,
    )

    current_depth = isnothing(params.domain_depth) ? T(2.5) * params.g / params.omega^2 : params.domain_depth
    k = dispersion_k(params.omega, params.g, current_depth, params.nu, params.sigma, params.rho)
    
    if isnothing(params.domain_depth)
        while isnan(k) || tanh(real(k) * current_depth) < T(0.99)
            current_depth *= T(1.05)
            k = dispersion_k(params.omega, params.g, current_depth, params.nu, params.sigma, params.rho)
        end
    end
    domain_depth = current_depth

    res = 80
    k_real = real(k)
    n = if isnothing(params.n)
        n_guess = max(res, ceil(Int, res / (2 * pi / k_real) * params.L_raft))
        n_guess + mod(n_guess, 2) + 1
    else
        params.n
    end

    M = isnothing(params.M) ? ceil(Int, res * k_real * domain_depth) : params.M
    L_domain = if isnothing(params.L_domain)
        min(3 * params.L_raft, round(20 * 2 * pi / k_real + params.L_raft; sigdigits=2))
    else
        params.L_domain
    end

    N = round(Int, (n - 1) * L_domain / params.L_raft) + 1
    L_domain_adim = T(N / n)

    x = collect(range(-L_domain_adim / 2, stop = L_domain_adim / 2, length = N))
    dx = x[2] - x[1]

    z = collect(range(-domain_depth, stop = 0.0, length = M)) ./ L_c
    dz = abs(z[2] - z[1])

    x_contact = abs.(x) .<= params.L_raft / (2 * L_c)
    x_free = abs.(x) .> params.L_raft / (2 * L_c)
    x_free[1] = false
    x_free[end] = false

    loads = motor_force / F_c * gaussian_load(motor_position / L_c, params.forcing_width, x[x_contact])

    nb_contact = count(x_contact)
    if params.EI isa AbstractVector
        @assert length(params.EI) == nb_contact "EI vector length ($(length(params.EI))) must match contact nodes ($nb_contact)"
    end
    if params.rho_raft isa AbstractVector
        @assert length(params.rho_raft) == nb_contact "rho_raft vector length ($(length(params.rho_raft))) must match contact nodes ($nb_contact)"
    end
    EI_vec = params.EI isa AbstractVector ? params.EI : fill(EI_scalar, nb_contact)
    rho_raft_vec = params.rho_raft isa AbstractVector ? params.rho_raft : fill(rho_raft_scalar, nb_contact)
    kappa_vec   = EI_vec ./ (rho_raft_scalar .* params.L_raft^4 .* params.omega^2)
    inertia_vec = rho_raft_vec ./ rho_raft_scalar

    return (
        params = params,
        motor_force = motor_force,
        d = d,
        motor_position = motor_position,
        L_c = L_c,
        t_c = t_c,
        m_c = m_c,
        F_c = F_c,
        nd_groups = nd_groups,
        domain_depth = domain_depth,
        k = k,
        n = n,
        M = M,
        L_domain = L_domain,
        N = N,
        x = x,
        z = z,
        dx = dx,
        dz = dz,
        x_contact = x_contact,
        x_free = x_free,
        loads = loads,
        nb_contact = nb_contact,
        kappa_vec   = kappa_vec,
        inertia_vec = inertia_vec,
    )
end

"""
    assemble_flexible_system(params::FlexibleParams)

Assemble the coupled fluid-raft linear system.

# Arguments
- `params`: Input configuration object.

# Returns
- A NamedTuple containing the system matrix `A`, vector `b`, and indexing information.
"""
function assemble_flexible_system(params::FlexibleParams{T}) where {T<:Real}
    derived = derive_params(params)
    NP = derived.N * derived.M
    nb_contact = derived.nb_contact
    system_size = 2 * NP + nb_contact
    BCtype = String(derived.params.bc)
    Fr = derived.nd_groups.Fr
    Gamma = derived.nd_groups.Gamma
    We = derived.nd_groups.We
    Lambda = derived.nd_groups.Lambda
    Re = derived.nd_groups.Re
    I_NP = sparse(T(1) * I, NP, NP)
    I_CC = sparse(T(1) * I, nb_contact, nb_contact)

    topMask = falses(derived.M, derived.N)
    topMask[end, 2:(end - 1)] .= true
    freeMask = repeat(reshape(derived.x_free, 1, derived.N), derived.M, 1) .& topMask
    contactMask = repeat(reshape(derived.x_contact, 1, derived.N), derived.M, 1) .& topMask

    bottomMask = falses(derived.M, derived.N)
    bottomMask[1, 2:(end - 1)] .= true
    rightEdgeMask = falses(derived.M, derived.N)
    rightEdgeMask[:, end] .= true
    leftEdgeMask = falses(derived.M, derived.N)
    leftEdgeMask[:, 1] .= true
    bulkMask = falses(derived.M, derived.N)
    bulkMask[2:(end - 1), 2:(end - 1)] .= true

    idxFreeSurf = findall(vec(freeMask))
    idxContact = findall(vec(contactMask))
    idxBulk = findall(vec(bulkMask))
    idxBottom = findall(vec(bottomMask))
    idxLeftEdge = findall(vec(leftEdgeMask))
    idxRightEdge = findall(vec(rightEdgeMask))

    half_free = length(idxFreeSurf) ÷ 2
    first_contact = first(idxContact)
    last_contact = last(idxContact)
    idxLeftFreeSurf = vcat(derived.M, idxFreeSurf[1:half_free], first_contact)
    idxRightFreeSurf = vcat(last_contact, idxFreeSurf[(half_free + 1):end], NP)
    nbLeft = length(idxLeftFreeSurf)

    S11 = spzeros(Complex{T}, NP, NP)
    S12 = spzeros(Complex{T}, NP, NP)
    S13 = spzeros(Complex{T}, NP, nb_contact)
    S21 = spzeros(Complex{T}, NP, NP)
    S22 = spzeros(Complex{T}, NP, NP)
    S23 = spzeros(Complex{T}, NP, nb_contact)
    S31 = spzeros(Complex{T}, nb_contact, NP)
    S32 = spzeros(Complex{T}, nb_contact, NP)
    S33 = spzeros(Complex{T}, nb_contact, nb_contact)

    L = idxLeftFreeSurf
    DxFree = getNonCompactFDmatrix(nbLeft, T(1.0), 1, derived.params.ooa)
    DxxFree = getNonCompactFDmatrix(nbLeft, T(1.0), 2, derived.params.ooa)
    S11[L[2:(end - 1)], L] = derived.dx^2 .* I_NP[L[2:(end - 1)], L] .+ (Complex{T}(4im) / Re) .* DxxFree[2:(end - 1), :]
    S12[L[2:(end - 1)], L] = (-derived.dx^2 / Fr^2) .* I_NP[L[2:(end - 1)], L] .+ (T(1) / (We * Gamma)) .* DxxFree[2:(end - 1), :]

    R = idxRightFreeSurf
    S11[R[2:(end - 1)], R] = derived.dx^2 .* I_NP[R[2:(end - 1)], R] .+ (Complex{T}(4im) / Re) .* DxxFree[2:(end - 1), :]
    S12[R[2:(end - 1)], R] = (-derived.dx^2 / Fr^2) .* I_NP[R[2:(end - 1)], R] .+ (T(1) / (We * Gamma)) .* DxxFree[2:(end - 1), :]

    CC = idxContact
    DxRaft = getNonCompactFDmatrix(nb_contact, T(1.0), 1, derived.params.ooa)
    Dx2Raft = getNonCompactFDmatrix(nb_contact, T(1.0), 2, derived.params.ooa)

    inertia_diag = Diagonal(Complex{T}.(1im .* derived.inertia_vec .+ 1im * Gamma * Lambda / Fr^2))
    S11[CC, CC] = (Complex{T}(1im) * Lambda * Gamma * derived.dx^2) .* I_NP[CC, CC] .- (T(2) * Gamma * Lambda / Re) .* Dx2Raft
    S12[CC, CC] = derived.dx^2 .* inertia_diag
    S13[CC, :] = Dx2Raft

    boundary_contact = [1, nb_contact]
    S11[CC[boundary_contact], :] .= 0
    S12[CC[boundary_contact], :] .= 0
    S13[CC[boundary_contact], :] .= 0

    S13[CC[1], :] = DxRaft[1, :]
    S12[CC[1], L] = (-Complex{T}(1im) * derived.dx * Lambda / We) .* DxFree[end, :]
    S13[CC[end], :] = DxRaft[end, :]
    S12[CC[end], R] = (-Complex{T}(1im) * derived.dx * Lambda / We) .* DxFree[1, :]

    S32[:, CC] = Complex{T}(1im) .* Dx2Raft
    S33 .= Diagonal(derived.dx^2 ./ derived.kappa_vec)
    S32[boundary_contact, :] .= 0
    S33[boundary_contact, :] .= 0
    S33[1, 1] = 1
    S33[end, end] = 1

    Dx, Dz = getNonCompactFDmatrix2D(derived.N, derived.M, T(1.0), T(1.0), 1, derived.params.ooa)
    Dxx, Dzz = getNonCompactFDmatrix2D(derived.N, derived.M, T(1.0), T(1.0), 2, derived.params.ooa)
    ratio_sq = (derived.dx / derived.dz)^2
    S11[idxBulk, :] = Dxx[idxBulk, :] + Dzz[idxBulk, :] * ratio_sq
    S12[idxBottom, :] = I_NP[idxBottom, :]

    S12[idxLeftEdge, :] .= 0
    S12[idxRightEdge, :] .= 0
    if startswith(lowercase(BCtype), "n")
        S11[idxLeftEdge, :] = Dx[idxLeftEdge, :]
        S11[idxRightEdge, :] = Dx[idxRightEdge, :]
    elseif startswith(lowercase(BCtype), "r")
        S11[idxLeftEdge, :] = (-Complex{T}(1im) * derived.k * derived.params.L_raft * derived.dx) .* I_NP[idxLeftEdge, :] .+ Dx[idxLeftEdge, :]
        S11[idxRightEdge, :] = (Complex{T}(1im) * derived.k * derived.params.L_raft * derived.dx) .* I_NP[idxRightEdge, :] .+ Dx[idxRightEdge, :]
    end

    S21 = Dz
    S22 = -derived.dz .* I_NP

    b = zeros(Complex{T}, system_size)
    b[CC] = -derived.dx^2 .* Complex{T}.(derived.loads)
    b[CC[boundary_contact]] .= 0

    A = [
        S11 S12 S13
        S21 S22 S23
        S31 S32 S33
    ]

    return (
        A = A,
        b = b,
        derived = derived,
        masks = (
            topMask = topMask,
            freeMask = freeMask,
            contactMask = contactMask,
            bottomMask = bottomMask,
            leftEdgeMask = leftEdgeMask,
            rightEdgeMask = rightEdgeMask,
            bulkMask = bulkMask,
        ),
        indices = (
            idxBulk = idxBulk,
            idxBottom = idxBottom,
            idxLeftEdge = idxLeftEdge,
            idxRightEdge = idxRightEdge,
            idxContact = idxContact,
            idxLeftFreeSurf = idxLeftFreeSurf,
            idxRightFreeSurf = idxRightFreeSurf,
        ),
    )
end

"""
    flexible_solver(params; return_system=false)

Solve the coupled flexible Surferbot problem.

# Arguments
- `params`: Input configuration object.
- `return_system`: Whether to include the assembled system in the result (default: false).

# Returns
- A `FlexibleResult` object, or a tuple `(result, system)` if `return_system` is true.
"""
function flexible_solver(params::FlexibleParams{T}; return_system::Bool=false) where {T<:Real}
    system = assemble_flexible_system(params)
    solution = solve_tensor_system(system.A, system.b)
    NP = system.derived.N * system.derived.M
    solution = solution[1:(2 * NP)]

    phi = reshape(solution[1:NP], system.derived.M, system.derived.N)
    phi_z = reshape(solution[(NP + 1):(2 * NP)], system.derived.M, system.derived.N)

    x = system.derived.x .* system.derived.L_c
    z = system.derived.z .* system.derived.L_c

    args = (
        sigma = params.sigma,
        rho = params.rho,
        omega = params.omega,
        nu = params.nu,
        g = params.g,
        L_raft = params.L_raft,
        d = system.derived.d,
        EI = params.EI,
        rho_raft = params.rho_raft,
        nd_groups = system.derived.nd_groups,
        x_contact = system.derived.x_contact,
        x = x,
        loads = system.derived.loads .* system.derived.F_c ./ system.derived.L_c,
        motor_position = params.motor_position,
        N = system.derived.N,
        M = system.derived.M,
        dx = system.derived.dx .* system.derived.L_c,
        dz = system.derived.dz .* system.derived.L_c,
        t_c = system.derived.t_c,
        L_c = system.derived.L_c,
        m_c = system.derived.m_c,
        k = system.derived.k,
        ooa = params.ooa,
    )

    U, power, thrust, eta, p, max_curvature, wave_steepness = calculate_surferbot_outputs(args, phi, phi_z, getNonCompactFDmatrix, getNonCompactFDmatrix2D)

    result = FlexibleResult{T}(
        U,
        power,
        thrust,
        x,
        z,
        Matrix(phi .* system.derived.L_c^2 ./ system.derived.t_c),
        Matrix(phi_z .* system.derived.L_c ./ system.derived.t_c),
        collect(eta),
        collect(p),
        max_curvature,
        wave_steepness,
        (
            args = merge(args, (pressure = p, power = power, thrust = thrust, phi_z = phi_z .* system.derived.L_c ./ system.derived.t_c, max_curvature = max_curvature, wave_steepness = wave_steepness)),
            system = return_system ? system : nothing,
        ),
    )

    return return_system ? (result = result, system = system) : result
end

"""
    flexible_surferbot_v2_julia(; kwargs...)

Convenience wrapper for `flexible_solver` with keyword arguments.

# Returns
- A tuple `(U, x, z, phi, eta, args)`.
"""
function flexible_surferbot_v2_julia(; kwargs...)
    result = flexible_solver(FlexibleParams{Float64}(; kwargs...))
    args = result.metadata.args
    return result.U, result.x, result.z, result.phi, result.eta, args
end

end # module
