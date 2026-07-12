using Test
using Surferbot

const BASE = FlexibleParams(
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

const NB = Surferbot.derive_params(BASE).nb_contact

@testset "graded EI" begin
    # 1. Scalar EI and constant-vector EI produce identical results
    result_scalar = flexible_solver(BASE)
    params_vec = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = fill(1e-4, NB),
        rho_raft = 0.052, domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    result_vec = flexible_solver(params_vec)
    @test result_scalar.thrust ≈ result_vec.thrust rtol = 1e-10
    @test result_scalar.U     ≈ result_vec.U     rtol = 1e-10

    # 2. Non-constant EI runs and produces finite output
    EI_graded = collect(LinRange(0.5e-4, 2e-4, NB))
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

@testset "graded rho_raft" begin
    # 1. Scalar rho_raft and constant-vector rho_raft produce identical results
    result_scalar = flexible_solver(BASE)
    params_vec = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = 1e-4, rho_raft = fill(0.052, NB),
        domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    result_vec = flexible_solver(params_vec)
    @test result_scalar.thrust ≈ result_vec.thrust rtol = 1e-10
    @test result_scalar.U     ≈ result_vec.U     rtol = 1e-10

    # 2. Non-constant rho_raft runs and produces finite output
    rho_graded = collect(LinRange(0.03, 0.08, NB))
    params_graded = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = 1e-4, rho_raft = rho_graded,
        domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    result_graded = flexible_solver(params_graded)
    @test isfinite(result_graded.thrust)
    @test isfinite(result_graded.U)

    # 3. Size mismatch raises AssertionError
    params_bad = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = 1e-4, rho_raft = [0.04, 0.06],
        domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    @test_throws AssertionError flexible_solver(params_bad)

    # 4. Modal decomposition warns for non-constant rho_raft
    @test_logs (:warn, r"spatially varying") decompose_raft_freefree_modes(result_graded; verbose = false)

    # 5. Partially-rigid raft (front 80% Inf EI, flexible tail) gives cm/s-scale drift
    n_rigid = round(Int, 0.8 * NB)
    EI_partial = vcat(fill(Inf, n_rigid), fill(1e-4, NB - n_rigid))
    params_partial = FlexibleParams(
        sigma = 72.2e-3, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.025,
        EI = EI_partial, rho_raft = 0.052,
        domain_depth = 0.5, n = 41, Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, omega = 2π * 10,
    )
    result_partial = flexible_solver(params_partial)
    @test isfinite(result_partial.U)
    @test 1e-3 < abs(result_partial.U) < 1.0
end
