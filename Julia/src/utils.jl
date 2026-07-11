module Utils

using LinearAlgebra
using SpecialFunctions
using SparseArrays
using ForwardDiff

export solve_tensor_system, gaussian_load, two_point_load, dispersion_k

"""
    erf_libm(x::Float64)

Direct call to the C math library's error function.
"""
erf_libm(x::Float64) = ccall((:erf, Base.Math.libm), Float64, (Float64,), x)

"""
    _extract_value(x)

Recursively extract the underlying numeric value from Dual or Complex types.
"""
_extract_value(x::Real) = ForwardDiff.value(x)
_extract_value(x::Complex) = Complex(_extract_value(real(x)), _extract_value(imag(x)))
_extract_value(x::AbstractArray) = _extract_value.(x)

"""
    _extract_partials(x, p_idx)

Extract the partial derivative with respect to parameter `p_idx`.
"""
_extract_partials(x::ForwardDiff.Dual, p_idx) = p_idx <= ForwardDiff.npartials(x) ? ForwardDiff.partials(x)[p_idx] : 0.0
_extract_partials(x::Real, p_idx) = 0.0
_extract_partials(x::Complex, p_idx) = Complex(_extract_partials(real(x), p_idx), _extract_partials(imag(x), p_idx))
_extract_partials(x::AbstractArray, p_idx) = _extract_partials.(x, p_idx)

"""
    solve_tensor_system(A, b; tol=1e-5, maxiter=1000)

Solve the linear system `A * x = b`.

Handles dense and sparse matrices. Includes special logic for `SparseMatrixCSC`
containing `ForwardDiff.Dual` numbers to avoid UMFPACK stack overflows.

# Arguments
- `A`: System matrix (dense or sparse).
- `b`: Right-hand side vector.
- `tol`: Convergence tolerance (default: 1e-5).
- `maxiter`: Maximum iterations (default: 1000).

# Returns
- The solution vector `x`.
"""
function solve_tensor_system(A, b; tol=1e-5, maxiter=1000)
    if size(A, 1) != size(b, 1)
        throw(ArgumentError("b.shape must equal A.shape[0]"))
    end

    if A isa SparseMatrixCSC && (eltype(A) <: ForwardDiff.Dual || eltype(A) <: Complex{<:ForwardDiff.Dual} ||
                                eltype(b) <: ForwardDiff.Dual || eltype(b) <: Complex{<:ForwardDiff.Dual})
        return _solve_sparse_dual(A, b)
    end

    x = nothing
    try
        x = A \ b
    catch err
        if err isa LinearAlgebra.SingularException
            x = qr(Matrix(A)) \ b
        else
            rethrow(err)
        end
    end
    
    return x
end

"""
    _solve_sparse_dual(A::SparseMatrixCSC, b)

Efficiently solve sparse linear systems with Dual numbers by separating values and partials.

# Arguments
- `A`: Sparse matrix with Dual components.
- `b`: RHS vector with Dual components.

# Returns
- The solution vector `x` with reconstructed Dual numbers.
"""
function _solve_sparse_dual(A::SparseMatrixCSC, b)
    A0 = _extract_value(A)
    b0 = _extract_value(b)
    
    fact = lu(A0)
    x0 = fact \ b0
    
    first_dual = nothing
    if eltype(A) <: ForwardDiff.Dual
        first_dual = A.nzval[1]
    elseif eltype(A) <: Complex{<:ForwardDiff.Dual}
        first_dual = real(A.nzval[1])
    elseif eltype(b) <: ForwardDiff.Dual
        first_dual = b[1]
    elseif eltype(b) <: Complex{<:ForwardDiff.Dual}
        first_dual = real(b[1])
    end
    
    if isnothing(first_dual)
        return x0
    end
    
    p = ForwardDiff.npartials(first_dual)
    T_dual_real = typeof(first_dual)
    T_val = eltype(x0)
    
    x_part_raw = zeros(T_val, length(x0), p)
    for p_idx in 1:p
        Ap_nzval = _extract_partials(A.nzval, p_idx)
        Ap = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, Ap_nzval)
        bp = _extract_partials(b, p_idx)
        
        rhs_p = bp - Ap * x0
        x_part_raw[:, p_idx] = fact \ rhs_p
    end
    
    x = map(1:length(x0)) do i
        re_dual = T_dual_real(real(x0[i]), ForwardDiff.Partials(Tuple(real(x_part_raw[i, :]))))
        if x0 isa Vector{<:Complex}
             im_dual = T_dual_real(imag(x0[i]), ForwardDiff.Partials(Tuple(imag(x_part_raw[i, :]))))
             return Complex(re_dual, im_dual)
        else
             return re_dual
        end
    end
    
    return x
end

"""
    gaussian_load(x0::T, sigma::T, x::AbstractVector{T}) where {T<:Real}

Evaluate a normalized Gaussian point load on a given grid.

# Arguments
- `x0`: Center of the load.
- `sigma`: Standard deviation (width).
- `x`: Grid points.

# Returns
- Vector of load values.
"""
function gaussian_load(x0::T, sigma::T, x::AbstractVector{T}) where {T<:Real}
    a = T(-0.5)
    b = T(0.5)
    xcol = collect(x)
    phi = exp.(-T(0.5) .* ((xcol .- x0) ./ sigma).^2)
    Z = sigma * sqrt(T(pi) / 2) * (erf((b - x0) / (sqrt(T(2)) * sigma)) - erf((a - x0) / (sqrt(T(2)) * sigma)))
    return reshape((1 / Z) .* phi, length(xcol))
end

"""
    two_point_load(x0::T, x::AbstractVector{T}) where {T<:Real}

Represent a unit point load at `x0` using linear interpolation onto the two
nearest grid points. Satisfies ∑ loads·dx = 1 and ∑ loads·x·dx = x0 exactly,
making it the correct discrete analog of a Dirac delta.
"""
function two_point_load(x0::T, x::AbstractVector{T}) where {T<:Real}
    dx = x[2] - x[1]
    i2 = clamp(searchsortedfirst(x, x0), 2, length(x))
    i1 = i2 - 1
    w2 = (x0 - x[i1]) / (x[i2] - x[i1])
    w1 = one(T) - w2
    loads = zeros(T, length(x))
    loads[i1] = w1 / dx
    loads[i2] = w2 / dx
    return loads
end

"""
    dispersion_k(omega::T, g::T, H::T, nu::T, sigma::T, rho::T; k0=Complex{T}(1.0 + 0.0im), num_steps=500) where {T<:Real}

Solve the complex dispersion relation for water waves using Newton's method.

The relation includes gravity, surface tension, and kinematic viscosity.

# Arguments
- `omega`: Angular frequency.
- `g`: Gravity.
- `H`: Water depth.
- `nu`: Kinematic viscosity.
- `sigma`: Surface tension.
- `rho`: Fluid density.
- `k0`: Initial guess for wavenumber (default: 1.0).
- `num_steps`: Iterations for Newton refinement (default: 500).
- `deep`: If `true`, solve the H-independent deep-water limit (tanh(kH) -> 1)
  directly instead of using `H`. Cheaper and exact, since it needs no depth
  at all; used to get a depth-independent reference wavenumber.

# Returns
- The complex wavenumber `k`.
"""
function dispersion_k(omega::T, g::T, H::T, nu::T, sigma::T, rho::T;
                      k0=Complex{T}(min(omega^2 / g, (rho * omega^2 / sigma)^(T(1)/3))),
                      num_steps=500, deep::Bool=false) where {T<:Real}
    function dispersion_eq(k)
        tanh_kH = deep ? one(k) : tanh(k * H)
        lhs = k * tanh_kH * g
        rhs = (-sigma / rho) * k^3 * tanh_kH + omega^2 - 4im * nu * omega * k^2
        return lhs - rhs
    end

    k = k0
    for _ in 1:num_steps
        f = dispersion_eq(k)
        dk = T(1e-8)
        f_plus = dispersion_eq(k + dk)
        f_minus = dispersion_eq(k - dk)
        df = (f_plus - f_minus) / (2 * dk)
        k = k - f / df
    end
    return k
end

end # module
