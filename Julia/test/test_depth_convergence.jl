using Test
using Surferbot

# Bath-depth convergence of the thrust. The default depth is chosen so that tanh(k H) >= 0.99
# for the radiated capillary wave; whether that is deep enough for the raft itself (a source
# of horizontal extent L >> H) is what this test decides.
#
# Uses the geometric vertical grid (dz_surface, z_growth), so a depth of order L costs
# ~100 vertical points instead of ~1e4. The surface spacing is held fixed across depths.
# Horizontal resolution is held fixed and coarse (thrust ratios between depths are
# insensitive to it).
#
# Measured on the unpatched solver (20 points of the paper's EI x x_M/L map): doubling the
# default depth changes thrust by 6%..600% at 19 of 20 points; the change falls below 1%
# between x16 and x32 (H ~ 1.1 L) for stiff rafts.

const _DC_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

function _thrust_graded(; ei_factor, x_M, depth_factor, scale = 0.5, growth = 1.05)
    p0 = FlexibleParams(motor_position = x_M, EI = _DC_EI0 * ei_factor, nu = 0.0)
    dd = Surferbot.derive_params(p0)
    n = round(Int, dd.n * scale); n += iseven(n) ? 1 : 0
    dz0 = dd.domain_depth / (round(Int, dd.Nz * scale) - 1)
    p = FlexibleParams(motor_position = x_M, EI = _DC_EI0 * ei_factor, nu = 0.0, n = n,
                       domain_depth = depth_factor * dd.domain_depth,
                       dz_surface = dz0, z_growth = growth)
    return solve_cached(p).thrust
end

@testset "Thrust converges in bath depth" begin
    for (label, eif, xM) in (("rigid, xM=-0.012", Inf, -0.012),
                             ("flexible EIx3e-4, xM=-0.012", 3e-4, -0.012),
                             ("flexible EIx1, xM=-0.020", 1.0, -0.020))
        T = Dict(df => _thrust_graded(ei_factor = eif, x_M = xM, depth_factor = df)
                 for df in (1.0, 2.0, 16.0, 32.0))
        shallow = abs(T[2.0] / T[1.0] - 1)
        deep = abs(T[32.0] / T[16.0] - 1)
        @info "depth convergence [$label]: change x1->x2 = $shallow, x16->x32 = $deep"
        # Deep end: converged to 2% by H ~ 1.1 L.
        @test deep < 0.02
        # The paper's default depth: doubling it should change the thrust by less than 1%.
        # It does not (known failure of the default depth criterion).
        @test_broken shallow < 0.01
    end
end
