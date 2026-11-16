using Test
using Surferbot

# Bath-depth convergence of the thrust. The default depth is chosen so that tanh(k H) >= 0.99
# for the radiated capillary wave; whether that is deep enough for the raft itself (a source
# of horizontal extent L >> H) is exactly what this test decides.
#
# Depth and the number of vertical points are scaled together (dz fixed), as in
# MATLAB/test/test_thrust_vs_depth.m. Horizontal resolution is held fixed and coarse
# (ratios of thrust at different depths are insensitive to it).
#
# Heavy: the deepest case needs several GB. Enabled with SURFERBOT_HEAVY_TESTS=1
# (run on the cluster, not in default CI).

const _DC_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12
const _DC_TOL = 0.01          # allowed relative thrust change per depth doubling

function _thrust_at_depth(; ei_factor, x_M, depth_factor, scale = 0.5)
    p0 = FlexibleParams(motor_position = x_M, EI = _DC_EI0 * ei_factor, nu = 0.0)
    dd = Surferbot.derive_params(p0)
    n = round(Int, dd.n * scale); n += iseven(n) ? 1 : 0
    Nz = round(Int, dd.Nz * scale * depth_factor)
    p = FlexibleParams(motor_position = x_M, EI = _DC_EI0 * ei_factor, nu = 0.0, n = n, Nz = Nz,
                       domain_depth = depth_factor * dd.domain_depth)
    r = solve_cached(p)
    return r.thrust
end

if get(ENV, "SURFERBOT_HEAVY_TESTS", "0") == "1"
    @testset "Thrust converges in bath depth (rigid and flexible, reference omega, g, sigma)" begin
        for (label, eif, xM) in (("rigid, xM=-0.012", Inf, -0.012),
                                 ("flexible EIx3e-4, xM=-0.012", 3e-4, -0.012),
                                 ("flexible EIx1, xM=-0.020", 1.0, -0.020))
            T = [_thrust_at_depth(ei_factor = eif, x_M = xM, depth_factor = df) for df in (1.0, 2.0, 4.0)]
            GC.gc()
            ch = [abs(T[i + 1] / T[i] - 1) for i in 1:2]
            @info "depth convergence [$label]: thrust(x1,x2,x4) = $T, change per doubling = $ch"
            # The depth criterion is tested on its own claim: doubling the default depth
            # should not change the thrust by more than the tolerance.
            @test ch[1] < _DC_TOL
            # Convergence must be monotone: later doublings change it less.
            @test ch[2] < ch[1]
        end
    end
else
    @info "Skipping depth-convergence test (set SURFERBOT_HEAVY_TESTS=1 to run)"
end
