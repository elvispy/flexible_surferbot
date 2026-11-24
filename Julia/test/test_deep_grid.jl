using Test
using Surferbot

# grid = :deep sets H = X / min(k1, k2) and L_domain = 2H, with k1 the deep-water
# gravity-capillary wavenumber and k2 = 1/(L_raft/2) (see docs/deep_water.md). X = 4.5 leaves
# e^-X ~ 1% of the field at the bottom (and eps_Z = (pi^2/48)/X^2 ~ 1% for the raft).

const _DG_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

@testset "grid = :deep geometry follows X" begin
    for X in (4.5, 14.0)
        p = FlexibleParams(grid = :deep, deep_X = X, motor_position = -0.012, EI = _DG_EI0, nu = 0.0)
        dd = Surferbot.derive_params(p)
        k1 = real(Surferbot.dispersion_k(p.omega, p.g, one(p.g), p.nu, p.sigma, p.rho; deep = true))
        H = X / min(k1, 2 / p.L_raft)
        @test dd.domain_depth ≈ H
        @test dd.L_domain >= 2H - 1e-12
        @test real(dd.k) * dd.domain_depth >= X - 0.05
        @test dd.Nz < 120
        @test minimum(abs.(dd.x .- 0.5)) < 1e-9
        @test minimum(abs.(dd.x .+ 0.5)) < 1e-9
    end
    # reference case: lateral walls exactly at the depth, about 55 vertical points
    d0 = Surferbot.derive_params(FlexibleParams(grid = :deep, nu = 0.0))
    @test d0.L_domain ≈ 2 * d0.domain_depth rtol = 1e-2
    @test 45 <= d0.Nz <= 65
    p2 = FlexibleParams(grid = :deep, domain_depth = 0.02, nu = 0.0)
    @test Surferbot.derive_params(p2).domain_depth == 0.02
    @test Surferbot.derive_params(FlexibleParams(nu = 0.0)).domain_depth < FlexibleParams().L_raft
    # gravity-dominated (low Froude number): k1 H >= X still holds
    pg = FlexibleParams(grid = :deep, g = 12633.0, nu = 0.0)
    ddg = Surferbot.derive_params(pg)
    @test real(ddg.k) * ddg.domain_depth >= 4.5 - 0.05
end

@testset "grid = :deep thrust agrees with a tighter-tolerance reference" begin
    # X = 4.5 against X = 14 (depth and domain width 3x). Half horizontal resolution
    # for cost; the 1e-2 bound is the design value (typical error is ~10x smaller).
    n0 = Surferbot.derive_params(FlexibleParams(nu = 0.0)).n
    n = round(Int, n0 / 2); n += iseven(n) ? 1 : 0
    for (label, eif, xM) in (("rigid", Inf, -0.012), ("flexible EIx1", 1.0, -0.020), ("flexible EIx3e-4", 3e-4, -0.012))
        pd = FlexibleParams(grid = :deep, deep_X = 4.5, n = n, motor_position = xM, EI = _DG_EI0 * eif, nu = 0.0)
        pr = FlexibleParams(grid = :deep, deep_X = 14.0, n = n, motor_position = xM, EI = _DG_EI0 * eif, nu = 0.0)
        Td = solve_cached(pd).thrust
        Tr = solve_cached(pr).thrust
        err = abs(Td - Tr) / abs(Tr)
        @info "deep grid [$label]: thrust $Td vs reference $Tr, rel err $err"
        @test err < 1e-2
        GC.gc()
    end
end
