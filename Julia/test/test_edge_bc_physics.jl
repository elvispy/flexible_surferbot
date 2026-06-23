using Test
using LinearAlgebra
using Surferbot

@testset "Edge BC — Newton's 2nd law (right edge sign)" begin
    # At sigma_scale = 1000, Lambda/We ≈ 0.066 — edge condition dominates.
    # Right edge: M_x and eta_x at the free surface must be CO-DIRECTED (cos_theta > 0).
    # Left  edge: they are ANTI-DIRECTED (cos_theta < 0) — this is correct per paper.
    #
    # This test FAILS before Fix 2 (sign error) and PASSES after.
    # Method: cos_theta = Re(Mx * conj(eta_x)) / (|Mx| * |eta_x|)
    #   eta from kinematic BC (independent of edge condition)
    #   M_x from bending moment block (independent of eta)
    params = FlexibleParams(
        sigma          = 0.0722 * 1000.0,
        rho            = 1000.0,
        omega          = 2π * 80.0,
        nu             = 1e-6,
        g              = 9.81,
        L_raft         = 0.05,
        motor_position = -0.003,
        d              = 0.03,
        EI             = 1.0e-4,
        rho_raft       = 0.052,
        motor_inertia  = 0.13e-3 * 2.5e-3,
        L_domain       = 0.14,
    )

    sys = assemble_flexible_system(params)
    sol = solve_tensor_system(sys.A, sys.b)

    derived = sys.derived
    NP      = derived.N * derived.M
    nb_c    = derived.nb_contact

    phi_z_vec = sol[(NP + 1):(2 * NP)]
    M_vec     = sol[(2 * NP + 1):(2 * NP + nb_c)]

    # eta from kinematic BC: eta ≈ phi_z / (i*omega*t_c)
    eta_vec = phi_z_vec ./ (im * derived.params.omega * derived.t_c)

    ooa    = derived.params.ooa
    nb_L   = length(sys.indices.idxLeftFreeSurf)
    DxRaft = getNonCompactFDmatrix(nb_c,  Float64(1.0), 1, ooa)
    DxFree = getNonCompactFDmatrix(nb_L,  Float64(1.0), 1, ooa)

    # eta_x at left edge: last stencil of left free surface
    eta_x_L = DxFree[end, :] ⋅ eta_vec[sys.indices.idxLeftFreeSurf]
    # eta_x at right edge: first stencil of right free surface
    eta_x_R = DxFree[1,   :] ⋅ eta_vec[sys.indices.idxRightFreeSurf]

    Mx_L = DxRaft[1,   :] ⋅ M_vec
    Mx_R = DxRaft[end, :] ⋅ M_vec

    cos_L = real(Mx_L * conj(eta_x_L)) / (abs(Mx_L) * abs(eta_x_L) + eps())
    cos_R = real(Mx_R * conj(eta_x_R)) / (abs(Mx_R) * abs(eta_x_R) + eps())

    @info "Edge BC sign check  cos_L=$(round(cos_L, sigdigits=4))  cos_R=$(round(cos_R, sigdigits=4))"

    # Left edge: force OPPOSES slope (anti-aligned)
    @test cos_L < -0.9
    # Right edge: force FOLLOWS slope (aligned) — fails before Fix 2
    @test cos_R > 0.9
end
