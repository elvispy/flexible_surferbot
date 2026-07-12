using Test
using Surferbot

@testset "flexible_solver" begin
    params = FlexibleParams(
        sigma = 0.0,
        rho = 1000.0,
        nu = 1e-6,
        g = 10 * 9.81,
        L_raft = 0.1,
        motor_position = 0.5 * 0.1 / 2,
        d = 0.1 / 2,
        EI = 100 * 3.0e9 * 3e-2 * (9.9e-4)^3 / 12,
        rho_raft = 0.018 * 3.0,
        domain_depth = 0.2,
        n = 41,
        Nz = 30,
        motor_inertia = 0.13e-3 * 2.5e-3,
        bc = :radiative,
        omega = 2 * pi * 10,
    )

    result = flexible_solver(params)

    @test isfinite(result.U)
    @test isfinite(result.power)
    @test isfinite(result.thrust)
    @test size(result.phi) == (length(result.z), length(result.x))
    @test size(result.phi_z) == (length(result.z), length(result.x))
    @test length(result.eta) == length(result.x)
end
