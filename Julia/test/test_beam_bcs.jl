using Test
using LinearAlgebra
using Surferbot

@testset "Free-end moments M(±½) = 0" begin
    # Free-free BC: bending moment vanishes at both raft tips.
    # Grid-independent — use coarse grid for speed.
    for (label, EI) in [("flexible EI=1e-4", 1e-4), ("semi-rigid EI=0.1", 0.1)]
        params = FlexibleParams(
            EI = EI, sigma = 72.2e-3, nu = 1e-6, omega = 2π * 40.0,
            rho = 1000.0, g = 9.81, L_raft = 0.05, motor_position = 0.01,
            d = 0.03, motor_inertia = 0.13e-3 * 2.5e-3, n = 21, M = 15,
        )
        sys = assemble_flexible_system(params)
        sol = solve_tensor_system(sys.A, sys.b)
        NP    = sys.derived.N * sys.derived.M
        nb_c  = sys.derived.nb_contact
        M_vec = sol[(2NP + 1):(2NP + nb_c)]
        M_scale = maximum(abs.(M_vec)) + eps()
        @info "$label  |M_left|/max=$(abs(M_vec[1])/M_scale)  |M_right|/max=$(abs(M_vec[end])/M_scale)"
        @test abs(M_vec[1])   < 1e-10 * M_scale
        @test abs(M_vec[end]) < 1e-10 * M_scale
    end
end
