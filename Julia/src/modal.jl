module Modal

using LinearAlgebra
using Printf

export ModalDecomposition,
       RawFreefreeBasis,
       build_raw_freefree_basis,
       psi_to_w_transform,
       coefficients_in_w_basis,
       decompose_raft_freefree_modes,
       trapz_weights,
       freefree_betaL_roots,
       freefree_mode_shape,
       weighted_mgs

"""
    ModalDecomposition

Container for the free-free modal projection of a raft displacement field.

The fields mirror the MATLAB helper `decompose_raft_freefree_modes`.

# Fields
- `n`: retained mode labels (`0`, `1`, ...)
- `mode_type`: `"rigid"` or `"elastic"` for each retained mode
- `betaL`: nondimensional modal wavenumbers
- `beta`: dimensional modal wavenumbers
- `q`: modal coefficients in the orthonormalized Psi basis
- `q_w`: modal coefficients in the raw analytical W basis
- `Q`: generalized pressure coefficients in Psi basis
- `Q_w`: generalized pressure coefficients in W basis
- `F`: generalized load coefficients in Psi basis
- `F_w`: generalized load coefficients in W basis
- `balance_residual`: per-mode beam-balance mismatch
- `energy_frac`: normalized `|q_n|^2` using Psi-basis (orthonormal) coefficients — NOT `|q_w_n|^2`; see implementation note in `decompose_raft_freefree_modes`
- `eta_recon`: reconstructed raft displacement
- `recon_rel_err`: weighted relative reconstruction error
- `x_raft`: raft-node coordinates
- `Psi`: weighted-orthonormal basis on the raft grid
- `Phi`: raw analytical basis on the raft grid
- `gram_cond`: condition number of the Gram matrix
"""
struct ModalDecomposition
    n::Vector{Int}
    mode_type::Vector{String}
    betaL::Vector{Float64}
    beta::Vector{Float64}
    q::Vector{ComplexF64} # Legacy orthonormal basis (psi)
    q_w::Vector{ComplexF64} # Raw analytical basis (w/phi)
    Q::Vector{ComplexF64}
    Q_w::Vector{ComplexF64}
    F::Vector{ComplexF64}
    F_w::Vector{ComplexF64}
    balance_residual::Vector{ComplexF64}
    energy_frac::Vector{Float64}
    eta_recon::Vector{ComplexF64}
    recon_rel_err::Float64
    x_raft::Vector{Float64}
    Psi::Matrix{Float64}
    Phi::Matrix{Float64}
    gram_cond::Float64
end

"""
    RawFreefreeBasis

Discrete raft-grid basis built from rigid modes and sampled free-free shapes.

# Fields
- `n`: mode labels
- `mode_type`: "rigid" or "elastic"
- `betaL`: nondimensional wavenumbers
- `beta`: dimensional wavenumbers
- `x_raft`: grid points
- `w`: integration weights
- `Phi`: basis matrix
- `gram_cond`: Gram matrix condition number
"""
struct RawFreefreeBasis
    n::Vector{Int}
    mode_type::Vector{String}
    betaL::Vector{Float64}
    beta::Vector{Float64}
    x_raft::Vector{Float64}
    w::Vector{Float64}
    Phi::Matrix{Float64}
    gram_cond::Float64
end

"""
    trapz_weights(x::AbstractVector{<:Real})

Compute weights for trapezoidal quadrature.

# Arguments
- `x`: Grid points (must be strictly increasing).

# Returns
- Vector of weights.
"""
function trapz_weights(x::AbstractVector{<:Real})
    xvec = collect(float.(x))
    n = length(xvec)
    if n == 1
        return ones(Float64, 1)
    end
    dx = diff(xvec)
    any(dx .<= 0) && error("x_raft must be strictly increasing for trapz weights.")
    w = zeros(Float64, n)
    w[1] = dx[1] / 2
    w[end] = dx[end] / 2
    if n > 2
        @inbounds for i in 2:(n - 1)
            w[i] = (xvec[i + 1] - xvec[i - 1]) / 2
        end
    end
    return w
end

"""
    freefree_betaL_roots(n::Integer)

Compute the positive elastic roots of the free-free beam characteristic equation.

The equation is `cosh(betaL) * cos(betaL) = 1`.

# Arguments
- `n`: Number of roots to find.

# Returns
- Vector of roots.
"""
function freefree_betaL_roots(n::Integer)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    roots = zeros(Float64, n)
    f(y) = cosh(y) * cos(y) - 1
    for k in 1:n
        a = k * pi
        b = (k + 1) * pi
        roots[k] = find_zero_bisection(f, a, b)
    end
    return roots
end

"""
    freefree_mode_shape(xi::AbstractVector{<:Real}, L::Real, betaL::Real)

Evaluate the free-free elastic mode shape on a given grid.

# Arguments
- `xi`: Grid coordinates in `[0, L]`.
- `L`: Length of the beam.
- `betaL`: Nondimensional wavenumber root.

# Returns
- Vector of mode shape values, normalized to unit L-infinity norm.
"""
function freefree_mode_shape(xi::AbstractVector{<:Real}, L::Real, betaL::Real)
    beta = betaL / L
    bx = beta .* collect(float.(xi))
    alpha = (sin(betaL) - sinh(betaL)) / (cosh(betaL) - cos(betaL))
    psi = (sin.(bx) .+ sinh.(bx)) .+ alpha .* (cos.(bx) .+ cosh.(bx))
    scale = maximum(abs.(psi))
    return isfinite(scale) && scale > 0 ? psi ./ scale : psi
end

"""
    weighted_mgs(Phi::AbstractMatrix{<:Real}, w::AbstractVector{<:Real})

Perform weighted modified Gram-Schmidt orthonormalization.

The inner product is defined as `<u, v>_W = u' * (v .* w)`.

# Arguments
- `Phi`: Input basis matrix.
- `w`: Integration weights.

# Returns
- A tuple `(Psi, keep)` where `Psi` is the orthonormal basis and `keep` is a boolean mask of retained columns.
"""
function weighted_mgs(Phi::AbstractMatrix{<:Real}, w::AbstractVector{<:Real})
    nrow, ncol = size(Phi)
    length(w) == nrow || throw(DimensionMismatch("weight vector length must match Phi rows"))
    wvec = collect(float.(w))
    phi = Matrix{Float64}(Phi)
    psi = zeros(Float64, nrow, ncol)
    keep = falses(ncol)
    count = 0

    col_norms = sqrt.(max.(real.(sum(phi .* (phi .* wvec), dims=1)), 0))
    tol = 1e-10 * max(1.0, maximum(col_norms))

    for j in 1:ncol
        v = copy(view(phi, :, j))
        for k in 1:count
            proj = dot(view(psi, :, k), v .* wvec)
            @views v .-= psi[:, k] .* proj
        end
        nv = sqrt(max(real(dot(v, v .* wvec)), 0.0))
        if isfinite(nv) && nv > tol
            count += 1
            @views psi[:, count] .= v ./ nv
            keep[j] = true
        end
    end

    return psi[:, 1:count], keep
end

"""
    build_raw_freefree_basis(x_raft, L_raft; num_modes=8, include_rigid=true)

Construct the discrete rigid-plus-elastic basis before orthonormalization.

# Arguments
- `x_raft`: Grid points on the raft.
- `L_raft`: Length of the raft.
- `num_modes`: Number of modes to generate (default: 8).
- `include_rigid`: Whether to include rigid-body modes (default: true).

# Returns
- A `RawFreefreeBasis` object.
"""
function build_raw_freefree_basis(
    x_raft::AbstractVector{<:Real},
    L_raft::Real;
    num_modes::Int=8,
    include_rigid::Bool=true,
)
    x = collect(float.(x_raft))
    Nr = length(x)
    Nr >= 3 || error("Need at least 3 raft points for modal decomposition.")
    w = trapz_weights(x)

    n_requested = min(num_modes, Nr)
    n_rigid = include_rigid ? min(2, n_requested) : 0
    n_elastic = n_requested - n_rigid

    xi = x .+ L_raft / 2
    phi_raw = zeros(Float64, Nr, n_requested)
    n_list = collect(0:(n_requested - 1))
    mode_type = fill("", n_requested)
    betaL = zeros(Float64, n_requested)
    beta = zeros(Float64, n_requested)

    col = 0
    if n_rigid >= 1
        col += 1
        phi_raw[:, col] .= 1.0
        mode_type[col] = "rigid"
    end
    if n_rigid >= 2
        col += 1
        phi_raw[:, col] .= xi .- L_raft / 2
        mode_type[col] = "rigid"
    end
    if n_elastic > 0
        betaL_el = freefree_betaL_roots(n_elastic)
        for j in 1:n_elastic
            col += 1
            betaL[col] = betaL_el[j]
            beta[col] = betaL[col] / L_raft
            phi_raw[:, col] .= freefree_mode_shape(xi, L_raft, betaL[col])
            mode_type[col] = "elastic"
        end
    end

    _, keep = weighted_mgs(phi_raw, w)
    Phi = phi_raw[:, keep]
    n_list = n_list[keep]
    mode_type = mode_type[keep]
    betaL = betaL[keep]
    beta = beta[keep]
    G = Phi' * (Phi .* w)
    return RawFreefreeBasis(
        collect(Int.(n_list)),
        collect(mode_type),
        collect(betaL),
        collect(beta),
        x,
        w,
        Matrix{Float64}(Phi),
        cond(G),
    )
end

"""
    psi_to_w_transform(Phi::AbstractMatrix{<:Real}, Psi::AbstractMatrix{<:Real}, w::AbstractVector{<:Real})

Compute the transformation matrix from orthonormal `Psi` basis to raw `Phi` basis.

# Arguments
- `Phi`: Raw analytical basis matrix.
- `Psi`: Orthonormal basis matrix.
- `w`: Integration weights.

# Returns
- Matrix `T` such that `Phi * c_w ≈ Psi * c_psi` with `c_w = T * c_psi`.
"""
function psi_to_w_transform(
    Phi::AbstractMatrix{<:Real},
    Psi::AbstractMatrix{<:Real},
    w::AbstractVector{<:Real},
)
    size(Phi, 1) == size(Psi, 1) || throw(DimensionMismatch("Phi and Psi must have the same row count"))
    length(w) == size(Phi, 1) || throw(DimensionMismatch("weight vector length must match basis rows"))
    G = Matrix{Float64}(Phi' * (Phi .* w))
    B = Matrix{Float64}(Phi' * (Psi .* w))
    return G \ B
end

"""
    coefficients_in_w_basis(coeff_psi, Phi, Psi, w)

Convert coefficients from the orthonormal `Psi` basis to the raw `Phi` basis.

# Arguments
- `coeff_psi`: Coefficients in the `Psi` basis.
- `Phi`: Raw analytical basis matrix.
- `Psi`: Orthonormal basis matrix.
- `w`: Integration weights.

# Returns
- Coefficients in the `Phi` basis.
"""
function coefficients_in_w_basis(
    coeff_psi::AbstractVector,
    Phi::AbstractMatrix{<:Real},
    Psi::AbstractMatrix{<:Real},
    w::AbstractVector{<:Real},
)
    T = psi_to_w_transform(Phi, Psi, w)
    return ComplexF64.(T * ComplexF64.(coeff_psi))
end

"""
    decompose_raft_freefree_modes(x, eta, pressure, loads, args; num_modes=8, include_rigid=true, verbose=true)

Project raft displacement field onto a free-free beam basis.

# Arguments
- `x`: Full surface grid coordinates.
- `eta`: Full surface displacement field.
- `pressure`: Raft pressure field.
- `loads`: Raft load field.
- `args`: Metadata/parameters from the solver.
- `num_modes`: Number of modes to retain (default: 8).
- `include_rigid`: Whether to include rigid modes (default: true).
- `verbose`: Whether to print a summary table (default: true).

# Returns
- A `ModalDecomposition` object.
"""
function decompose_raft_freefree_modes(
    x::AbstractVector,
    eta::AbstractVector,
    pressure::AbstractVector,
    loads::AbstractVector,
    args;
    num_modes::Int=8,
    include_rigid::Bool=true,
    verbose::Bool=true,
)
    contact = collect(Bool.(args.x_contact))
    x_all = collect(float.(x))
    eta_all = ComplexF64.(eta)
    length(contact) == length(x_all) || throw(DimensionMismatch("x_contact and x length mismatch"))
    length(eta_all) == length(x_all) || throw(DimensionMismatch("eta and x length mismatch"))

    x_raft = x_all[contact]
    eta_raft = eta_all[contact]
    Nr = length(x_raft)
    Nr >= 3 || error("Need at least 3 raft points for modal decomposition.")

    if args.EI isa AbstractVector && !allequal(args.EI)
        @warn "decompose_raft_freefree_modes: EI is spatially varying; free-free mode shapes assume uniform EI. Modal decomposition is approximate."
    end
    if args.rho_raft isa AbstractVector && !allequal(args.rho_raft)
        @warn "decompose_raft_freefree_modes: rho_raft is spatially varying; free-free mode shapes assume uniform rho_raft. Modal decomposition is approximate."
    end

    p_raft = ComplexF64.(pressure)
    f_raft = ComplexF64.(loads)
    length(p_raft) == Nr || throw(DimensionMismatch("pressure must have one value per raft point"))
    length(f_raft) == Nr || throw(DimensionMismatch("loads must have one value per raft point"))

    w = trapz_weights(x_raft)
    Weta = eta_raft .* w
    Wdp = (args.d .* p_raft) .* w
    Wf = f_raft .* w

    n_requested = min(num_modes, Nr)
    n_rigid = include_rigid ? min(2, n_requested) : 0
    n_elastic = n_requested - n_rigid

    xi = x_raft .+ args.L_raft / 2
    phi_raw = zeros(Float64, Nr, n_requested)
    n_list = collect(0:(n_requested - 1))
    mode_type = fill("", n_requested)
    betaL = zeros(Float64, n_requested)
    beta = zeros(Float64, n_requested)

    col = 0
    if n_rigid >= 1
        col += 1
        phi_raw[:, col] .= 1.0
        mode_type[col] = "rigid"
    end
    if n_rigid >= 2
        col += 1
        phi_raw[:, col] .= xi .- args.L_raft / 2
        mode_type[col] = "rigid"
    end
    if n_elastic > 0
        betaL_el = freefree_betaL_roots(n_elastic)
        for j in 1:n_elastic
            col += 1
            betaL[col] = betaL_el[j]
            beta[col] = betaL[col] / args.L_raft
            phi_raw[:, col] .= freefree_mode_shape(xi, args.L_raft, betaL[col])
            mode_type[col] = "elastic"
        end
    end

    Psi, keep = weighted_mgs(phi_raw, w)
    isempty(Psi) && error("No valid modes remained after orthonormalization.")

    n_list = n_list[keep]
    mode_type = mode_type[keep]
    betaL = betaL[keep]
    beta = beta[keep]
    Phi = phi_raw[:, keep]

    M = Psi' * (Psi .* w)
    rhs_q = Psi' * Weta
    rhs_Q = Psi' * Wdp
    rhs_F = Psi' * Wf

    gram_cond = cond(M)
    use_pinv = !isfinite(gram_cond) || gram_cond > 1e10
    solver = use_pinv ? pinv(M) : M
    q = use_pinv ? solver * rhs_q : solver \ rhs_q
    Q = use_pinv ? solver * rhs_Q : solver \ rhs_Q
    F = use_pinv ? solver * rhs_F : solver \ rhs_F

    # Calculate coefficients in raw analytical basis W_n (Phi)
    # (Phi' W Phi) q_w = (Phi' W eta)
    G = Phi' * (Phi .* w)
    q_w = G \ (Phi' * Weta)
    Q_w = G \ (Phi' * Wdp)
    F_w = G \ (Phi' * Wf)

    beta4 = beta .^ 4
    EI_scalar = args.EI isa AbstractVector ? minimum(args.EI) : args.EI
    rho_raft_scalar = args.rho_raft isa AbstractVector ? minimum(args.rho_raft) : args.rho_raft
    balance_residual = (EI_scalar .* beta4 .- rho_raft_scalar .* args.omega^2) .* q_w .- (Q_w .- F_w)
    eta_recon = Psi * q

    recon_num = sqrt(max(real(dot(eta_raft - eta_recon, (eta_raft - eta_recon) .* w)), 0.0))
    recon_den = sqrt(max(real(dot(eta_raft, eta_raft .* w)), 0.0))
    recon_rel_err = recon_den > 0 ? recon_num / recon_den : NaN

    # !! IMPORTANT: energy_frac is computed from the Psi-basis (orthonormal) coefficients q,
    # NOT from the raw W-basis coefficients q_w.
    #
    # The W-basis rigid rotation shape is φ₁(x) = x, which has L2-norm² = L³/12 ≈ 1e-5 m³,
    # roughly 5000× smaller than the elastic mode norms² ≈ L ≈ 0.05 m. Solving G q_w = rhs
    # inflates the rotation coefficient by that factor, making |q_w_rotation|² dominate
    # energy_frac even when the raft is visibly bending. This is a units artifact: the rotation
    # coefficient is dimensionless (rad) while elastic coefficients are in metres.
    #
    # The Psi basis is w-orthonormal by construction (weighted_mgs), so |q_n|² is the true
    # fraction of weighted-L2 raft deflection energy in mode n. Use q, not q_w, here.
    @debug "energy_frac computed from Psi-basis |q|² (orthonormal), not W-basis |q_w|²" *
           " — see comment above for why q_w would give spuriously large rigid-rotation fraction"
    q_energy = abs2.(q)
    q_energy_sum = sum(q_energy)
    energy_frac = q_energy_sum > 0 ? q_energy ./ q_energy_sum : zeros(Float64, length(q_w))

    modal = ModalDecomposition(
        collect(Int.(n_list)),
        collect(mode_type),
        collect(betaL),
        collect(beta),
        collect(ComplexF64.(q)),
        collect(ComplexF64.(q_w)),
        collect(ComplexF64.(Q)),
        collect(ComplexF64.(Q_w)),
        collect(ComplexF64.(F)),
        collect(ComplexF64.(F_w)),
        collect(ComplexF64.(balance_residual)),
        collect(Float64.(energy_frac)),
        collect(ComplexF64.(eta_recon)),
        recon_rel_err,
        collect(Float64.(x_raft)),
        Matrix{Float64}(Psi),
        Matrix{Float64}(Phi),
        gram_cond,
    )

    verbose && print_modal_log(modal)
    return modal
end

"""
    decompose_raft_freefree_modes(result; kwargs...)

Convenience wrapper for `decompose_raft_freefree_modes` using a `FlexibleResult`.
"""
function decompose_raft_freefree_modes(result; kwargs...)
    args = result.metadata.args
    return decompose_raft_freefree_modes(result.x, result.eta, args.pressure, args.loads, args; kwargs...)
end

"""
    find_zero_bisection(f, a::Real, b::Real; maxiter::Int=200, atol::Float64=1e-12)

Root-finding using the bisection method.

# Arguments
- `f`: Function to find the zero of.
- `a`, `b`: Bracketing interval.
- `maxiter`: Maximum iterations (default: 200).
- `atol`: Absolute tolerance (default: 1e-12).

# Returns
- The approximate root.
"""
function find_zero_bisection(f, a::Real, b::Real; maxiter::Int=200, atol::Float64=1e-12)
    fa = f(a)
    fb = f(b)
    fa == 0 && return float(a)
    fb == 0 && return float(b)
    signbit(fa) == signbit(fb) && error("Root is not bracketed in [$a, $b].")

    left = float(a)
    right = float(b)
    fleft = fa
    for _ in 1:maxiter
        mid = (left + right) / 2
        fmid = f(mid)
        if abs(fmid) <= atol || abs(right - left) <= atol
            return mid
        end
        if signbit(fleft) == signbit(fmid)
            left = mid
            fleft = fmid
        else
            right = mid
        end
    end
    return (left + right) / 2
end

"""
    print_modal_log(modal::ModalDecomposition)

Print a formatted summary table of the modal decomposition.
"""
function print_modal_log(modal::ModalDecomposition)
    println()
    println("Modal decomposition (free-free beam basis)")
    println("n | type    | betaL      | |q_n|      | arg(q_n)   | |Q_n|      | |F_n|      | |r_n|")
    println("----------------------------------------------------------------------------------------")
    for i in eachindex(modal.n)
        println(
            lpad(modal.n[i], 1), " | ",
            rpad(modal.mode_type[i], 7), " | ",
            lpad(@sprintf("%.5f", modal.betaL[i]), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.q[i])), 10), " | ",
            lpad(@sprintf("%.3f", angle(modal.q[i])), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.Q[i])), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.F[i])), 10), " | ",
            lpad(@sprintf("%.3e", abs(modal.balance_residual[i])), 10),
        )
    end
    println(
        "Diagnostics: sum(energy_frac)=",
        @sprintf("%.6f", sum(modal.energy_frac)),
        ", recon_rel_err=", @sprintf("%.3e", modal.recon_rel_err),
        ", cond(Psi'W Psi)=", @sprintf("%.3e", modal.gram_cond),
    )
end

end
