using Test
using Surferbot

# grid = :deep picks depth, domain width and grid from a tolerance eps = deep_tol:
#   H >= pi L/sqrt(192 eps), L_domain >= pi L/sqrt(48 eps), 4kH(1+B)/((1+3B) sinh 2kH) <= eps
# (derived from the finite-depth Dirichlet-to-Neumann symbol q tanh(qH); see docs/deep_water.md).
# On the 20-point sample of the paper's EI x x_M/L map the thrust error from a finite domain
# width is 0.1 (median) to ~1 (max) times (pi^2/48)(L/L_domain)^2.

const _DG_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

@testset "grid = :deep geometry follows the tolerance" begin
    for eps_d in (1e-2, 1e-3)
        p = FlexibleParams(grid = :deep, deep_tol = eps_d, motor_position = -0.012, EI = _DG_EI0, nu = 0.0)
        dd = Surferbot.derive_params(p)
        L = p.L_raft
        @test dd.domain_depth >= pi * L / sqrt(192 * eps_d) - 1e-12
        @test dd.L_domain >= pi * L / sqrt(48 * eps_d) - 1e-12
        # wave side: finite-depth radiation-stress fraction below eps
        k = real(dd.k); B = p.sigma * k^2 / (p.rho * p.g)
        @test 4 * k * dd.domain_depth * (1 + B) / ((1 + 3B) * sinh(2 * k * dd.domain_depth)) <= eps_d
        @test dd.Nz < 120
        @test minimum(abs.(dd.x .- 0.5)) < 1e-9
        @test minimum(abs.(dd.x .+ 0.5)) < 1e-9
    end
    p2 = FlexibleParams(grid = :deep, domain_depth = 0.02, nu = 0.0)
    @test Surferbot.derive_params(p2).domain_depth == 0.02
    @test Surferbot.derive_params(FlexibleParams(nu = 0.0)).domain_depth < FlexibleParams().L_raft
    # gravity-dominated (low Froude number): the wave-side condition can bind
    pg = FlexibleParams(grid = :deep, deep_tol = 1e-3, g = 12633.0, nu = 0.0)
    ddg = Surferbot.derive_params(pg)
    @test real(ddg.k) * ddg.domain_depth >= 4.3
end

@testset "grid = :deep thrust agrees with a tighter-tolerance reference" begin
    # eps = 1e-2 against eps = 1e-3 (depth 3x, domain width 3x). Half horizontal resolution
    # for cost; the tolerance 1e-2 is the design bound (typical error is ~10x smaller).
    n0 = Surferbot.derive_params(FlexibleParams(nu = 0.0)).n
    n = round(Int, n0 / 2); n += iseven(n) ? 1 : 0
    for (label, eif, xM) in (("rigid", Inf, -0.012), ("flexible EIx1", 1.0, -0.020), ("flexible EIx3e-4", 3e-4, -0.012))
        pd = FlexibleParams(grid = :deep, deep_tol = 1e-2, n = n, motor_position = xM, EI = _DG_EI0 * eif, nu = 0.0)
        pr = FlexibleParams(grid = :deep, deep_tol = 1e-3, n = n, motor_position = xM, EI = _DG_EI0 * eif, nu = 0.0)
        Td = solve_cached(pd).thrust
        Tr = solve_cached(pr).thrust
        err = abs(Td - Tr) / abs(Tr)
        @info "deep grid [$label]: thrust $Td vs reference $Tr, rel err $err"
        @test err < 1e-2
        GC.gc()
    end
end
