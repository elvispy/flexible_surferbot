using Test
using Surferbot

@testset "graded EI" begin
    base = FlexibleParams(
        sigma = 0.0,
        rho = 1000.0,
        nu = 1e-6,
        g = 9.81,
        L_raft = 0.05,
        motor_position = 0.01,
        d = 0.025,
        EI = 1e-4,
        rho_raft = 0.052,
        domain_depth = 0.5,
        n = 41,
        Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3,
        bc = :radiative,
        omega = 2π * 10,
    )

    # 1. Scalar EI and constant-vector EI produce identical results
    result_scalar = flexible_solver(base)
    nb = Surferbot.derive_params(base).nb_contact
    params_vec = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = fill(1e-4, nb),
        rho_raft = 0.052, domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    result_vec = flexible_solver(params_vec)
    @test result_scalar.thrust ≈ result_vec.thrust rtol = 1e-10
    @test result_scalar.U     ≈ result_vec.U     rtol = 1e-10

    # 2. Non-constant EI runs and produces finite output
    EI_graded = collect(LinRange(0.5e-4, 2e-4, nb))
    params_graded = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = EI_graded,
        rho_raft = 0.052, domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    result_graded = flexible_solver(params_graded)
    @test isfinite(result_graded.thrust)
    @test isfinite(result_graded.U)

    # 3. Size mismatch raises AssertionError
    params_bad = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = [1e-4, 2e-4],
        rho_raft = 0.052, domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    @test_throws AssertionError flexible_solver(params_bad)

    # 4. Modal decomposition warns for non-constant EI
    @test_logs (:warn, r"spatially varying") decompose_raft_freefree_modes(result_graded; verbose = false)
end
