using Test
using LinearAlgebra
using Surferbot

@testset "System residual — A·x = b to solver precision" begin
    # Grid-independent algebraic property: use coarse grid for speed.
    cases = [
        ("10 Hz σ=0 inviscid", FlexibleParams(
            sigma = 0.0, nu = 0.0, omega = 2π * 10.0,
            rho = 1000.0, g = 9.81, L_raft = 0.05, motor_position = 0.01,
            d = 0.03, motor_inertia = 0.13e-3 * 2.5e-3, n = 21, M = 15,
        )),
        ("80 Hz capgrav viscous", FlexibleParams(
            sigma = 72.2e-3, nu = 1e-6, omega = 2π * 80.0,
            rho = 1000.0, g = 9.81, L_raft = 0.05, motor_position = 0.01,
            d = 0.03, motor_inertia = 0.13e-3 * 2.5e-3, n = 21, M = 15,
        )),
    ]
    for (label, params) in cases
        sys = assemble_flexible_system(params)
        sol = solve_tensor_system(sys.A, sys.b)
        res = norm(sys.A * sol - sys.b) / norm(sys.b)
        @info "$label  residual=$res"
        @test res < 1e-10
    end
end
