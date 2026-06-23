using Test
using LinearAlgebra
using Surferbot

@testset "Symmetry under symmetric forcing (motor at centre, σ=0)" begin
    # Motor at x=0, pure gravity waves (σ=0): the system is exactly reflection-symmetric.
    # When the edge BC is inactive (Λ/We=0 ⟹ σ=0), the matrix A commutes with the
    # reflection permutation P, so the unique solution for symmetric b satisfies P·x = x.
    # With σ>0 the ±Λ/We edge terms are directional and break this exact symmetry
    # (physical effect, not a bug), so the test only covers the σ=0 case.
    params = FlexibleParams(
        sigma = 0.0, rho = 1000.0, nu = 0.0, g = 9.81,
        L_raft = 0.05, motor_position = 0.0, d = 0.03, L_domain = 0.14,
        motor_inertia = 0.13e-3 * 2.5e-3, omega = 2π * 40.0, n = 21, M = 15,
    )
    result = flexible_solver(params)

    # η on raft is symmetric (even): η(xᵢ) = η(x_{n+1−i})
    contact = Bool.(result.metadata.args.x_contact)
    eta_c = result.eta[contact]
    sym_err = norm(eta_c .- reverse(eta_c)) / (norm(eta_c) + eps())
    @info "Symmetry error on raft (σ=0): $sym_err"
    @test sym_err < 1e-8

    # Thrust must vanish for σ=0: T = Sxx ∝ (gk/4)(|η_L|² − |η_R|²) = 0
    # Tolerance ~1e-15 N/m: machine-precision residual from symmetric η (error ≤ 1e-8)
    @test abs(result.thrust) < 1e-15
end

@testset "Thrust sign reverses with motor position" begin
    # Motor right of centre → positive/negative thrust; left → opposite.
    # Any global sign error in the load vector or thrust formula breaks this.
    # Use ν>0 so drag is finite and U is well-defined; use coarse grid for speed.
    make = pos -> FlexibleParams(
        sigma = 72.2e-3, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = pos, d = 0.03,
        motor_inertia = 0.13e-3 * 2.5e-3, omega = 2π * 40.0, n = 21, M = 15,
    )
    r_plus  = flexible_solver(make(+0.010))
    r_minus = flexible_solver(make(-0.010))
    @test sign(r_plus.thrust) == -sign(r_minus.thrust)
    # U is the steady-state speed magnitude; it is always positive in this solver.
    # The meaningful sign check is on thrust, not U.
    @test isfinite(r_plus.U) && isfinite(r_minus.U)
end

@testset "Zero forcing → zero fields" begin
    # motor_inertia = 0 → b = 0 → x = 0 → η = 0, T = 0.
    params = FlexibleParams(
        sigma = 72.2e-3, rho = 1000.0, nu = 1e-6, g = 9.81,
        L_raft = 0.05, motor_position = 0.01, d = 0.03,
        motor_inertia = 0.0, omega = 2π * 40.0, n = 21, M = 15,
    )
    result = flexible_solver(params)
    @test maximum(abs.(result.eta)) < 1e-20
    @test abs(result.thrust) < 1e-20
end
