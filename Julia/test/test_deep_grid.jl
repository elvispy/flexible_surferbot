using Test
using Surferbot

# grid = :deep: depth max(tanh criterion, 3 L), geometric vertical grid (k*dz_surface = 0.1,
# growth 1.1), raft edges on grid nodes. Measured on the 20-point sample of the paper's
# EI x x_M/L map: thrust error against H = 8 L is 5e-4 (worst case) and 3e-5 (median) at
# H = 3 L; 6e-3 and 3e-4 at 2 L.

const _DG_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

@testset "grid = :deep geometry" begin
    p = FlexibleParams(grid = :deep, motor_position = -0.012, EI = _DG_EI0, nu = 0.0)
    dd = Surferbot.derive_params(p)
    L = p.L_raft
    @test dd.domain_depth >= 3 * L - 1e-12
    @test dd.Nz < 100                                  # geometric grid, not ~1e4 uniform points
    @test minimum(abs.(dd.x .- 0.5)) < 1e-9            # right raft edge on a node
    @test minimum(abs.(dd.x .+ 0.5)) < 1e-9            # left raft edge on a node
    # An explicit depth is respected.
    p2 = FlexibleParams(grid = :deep, domain_depth = 0.02, nu = 0.0)
    @test Surferbot.derive_params(p2).domain_depth == 0.02
    # The legacy grid is untouched.
    pl = FlexibleParams(nu = 0.0)
    @test Surferbot.derive_params(pl).domain_depth < L
end

@testset "grid = :deep thrust is converged in depth and vertical resolution" begin
    for (label, eif, xM) in (("rigid", Inf, -0.012), ("flexible EIx1", 1.0, -0.020), ("flexible EIx3e-4", 3e-4, -0.012))
        pd = FlexibleParams(grid = :deep, motor_position = xM, EI = _DG_EI0 * eif, nu = 0.0)
        k = real(Surferbot.derive_params(pd).k)
        # Reference: twice the depth, 4x finer surface spacing, slower growth.
        pr = FlexibleParams(grid = :deep, motor_position = xM, EI = _DG_EI0 * eif, nu = 0.0,
                            domain_depth = 6 * pd.L_raft, dz_surface = 0.025 / k, z_growth = 1.05)
        Td = solve_cached(pd).thrust
        Tr = solve_cached(pr).thrust
        err = abs(Td - Tr) / abs(Tr)
        @info "deep grid [$label]: thrust $Td vs reference $Tr, rel err $err"
        @test err < 2e-3
        GC.gc()
    end
end
