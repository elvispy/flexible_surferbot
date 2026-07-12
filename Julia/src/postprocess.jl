module PostProcess

using LinearAlgebra

export calculate_surferbot_outputs

"""
    calculate_surferbot_outputs(args, phi, phi_z, getNonCompactFDmatrix, getNonCompactFDmatrix2D)

Compute thrust, drift speed, input power, and surface fields from the harmonic solution.

# Arguments
- `args`: Metadata and parameters used for the simulation.
- `phi`: Potential field matrix.
- `phi_z`: Vertical derivative of the potential field matrix.
- `getNonCompactFDmatrix`: Function handle to generate 1D FD matrices.
- `getNonCompactFDmatrix2D`: Function handle to generate 2D FD matrices.

# Returns
- A tuple `(U, power, thrust, eta, p, max_curvature, wave_steepness)`.
"""
function calculate_surferbot_outputs(args, phi, phi_z, getNonCompactFDmatrix, getNonCompactFDmatrix2D)
    Nx = args.Nx
    Nz = args.Nz
    dx_adim = args.dx / args.L_c
    dz_adim = args.dz / args.L_c
    F_c = args.m_c * args.L_c / args.t_c^2
    f_adim = zero(args.loads)

    contact_mask = args.x_contact
    Nr = count(contact_mask)

    D1r = Matrix(getNonCompactFDmatrix(Nr, 1.0, 1, args.ooa)) / dx_adim
    D2r = Matrix(getNonCompactFDmatrix(Nr, 1.0, 2, args.ooa)) / dx_adim^2

    eta_adim = (1 / (im * args.omega * args.t_c)) .* vec(phi_z[end, :])
    eta_raft = eta_adim[contact_mask]
    eta_x_raft = D1r * eta_raft

    xL = findfirst(contact_mask)
    xR = findlast(contact_mask)
    eta_x_surf_L = dot(D1r[end, (end - 9):end], eta_adim[(xL - 9):xL])
    eta_x_surf_R = dot(D1r[1, 1:10], eta_adim[xR:(xR + 9)])

    phi_surf = vec(phi[end, :])
    phi_raft = phi_surf[contact_mask]

    P1_r = (im * args.nd_groups.Gamma) .* phi_raft .- (2 * args.nd_groups.Gamma / args.nd_groups.Re) .* (D2r * phi_raft)
    p_adim = -(im * args.nd_groups.Gamma / args.nd_groups.Fr^2) .* eta_raft .+ P1_r

    Q_adim = f_adim .- args.nd_groups.Lambda .* p_adim
    x_contact = args.x[contact_mask] ./ args.L_c
    integrand = real.(Q_adim) .* real.(eta_x_raft) .+ imag.(Q_adim) .* imag.(eta_x_raft)
    thrust_adim = -trapz(x_contact, integrand) / 2
    thrust_adim += args.sigma * args.d / F_c / 4 * (abs(eta_x_surf_L)^2 - abs(eta_x_surf_R)^2)
    thrust = real(thrust_adim * F_c)

    thrust_factor = 4 / 9 * args.nu * (args.rho * args.d)^2 * args.L_raft
    U = cbrt(thrust^2 / thrust_factor)

    power = real(-(0.5 * args.omega * args.L_c * F_c) * trapz(x_contact, imag.(eta_raft) .* args.loads))
    eta = eta_adim .* args.L_c
    p = p_adim .* F_c / args.L_c^2
    
    # Calculate curvature for penalty (dimensionless)
    eta_xx_raft = D2r * eta_raft
    max_curvature = maximum(abs.(eta_xx_raft))
    
    # Calculate wave steepness (ak)
    k_real = real(args.k)
    max_a = maximum(abs.(eta))
    wave_steepness = max_a * k_real
    
    return U, power, thrust, eta, p, max_curvature, wave_steepness
end

"""
    trapz(x::AbstractVector, y::AbstractVector)

Perform numerical integration using the trapezoidal rule.

# Arguments
- `x`: Coordinates of the grid points.
- `y`: Values of the function at the grid points.

# Returns
- The approximate integral value.
"""
function trapz(x::AbstractVector, y::AbstractVector)
    total = zero(promote_type(eltype(x), eltype(y)))
    for i in 1:(length(x) - 1)
        total += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    end
    return total
end

end # module
