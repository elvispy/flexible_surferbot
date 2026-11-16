using Test
using LinearAlgebra
using Surferbot

# Mirror symmetry of the full finite-difference solver at nonzero surface tension.
#
# A symmetric raft in a symmetric fluid, driven by a point motor at x_M, must give
# the mirror-image response when the motor is moved to -x_M. Nothing in these tests
# uses a sign convention from the manuscript: they only use the reflection x -> -x.
#
# Cases marked @test_broken document a known failure (right-edge capillary shear
# condition is not the mirror image of the left-edge one). Convert them to @test when
# the edge condition is fixed; Julia reports an unexpected pass until then.

const _EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

_params(xm; sf = 1.0, nu = 0.0, kwargs...) = FlexibleParams(
    motor_position = xm, EI = _EI0 * 3e-4, sigma = 72.2e-3 * sf, nu = nu; kwargs...,
)

_raft_eta(r) = r.eta[Bool.(r.metadata.args.x_contact)]

@testset "Mirror equivalence: motor at -x_M is the reflection of motor at +x_M" begin
    for sf in (1.0, 300.0)
        rp = flexible_solver(_params(+0.012; sf = sf))
        rm = flexible_solver(_params(-0.012; sf = sf))
        # Thrust is odd in x_M.
        thrust_err = abs(rp.thrust + rm.thrust) / max(abs(rp.thrust), abs(rm.thrust))
        # Raft displacement of the mirrored solve is the reversed displacement.
        ep, em = _raft_eta(rp), _raft_eta(rm)
        eta_err = norm(ep .- reverse(em)) / norm(ep)
        @info "mirror equivalence sigma x$sf: thrust_err=$thrust_err eta_err=$eta_err"
        @test_broken thrust_err < 1e-6
        @test_broken eta_err < 1e-6
    end
end

@testset "Symmetric forcing gives zero thrust at large surface tension" begin
    sf = 300.0
    r0 = flexible_solver(_params(0.0; sf = sf))
    rref = flexible_solver(_params(0.012; sf = sf))
    contact = Bool.(r0.metadata.args.x_contact)
    eta_c = r0.eta[contact]
    sym_err = norm(eta_c .- reverse(eta_c)) / norm(eta_c)
    thrust_ratio = abs(r0.thrust) / abs(rref.thrust)
    @info "centred motor sigma x$sf: sym_err=$sym_err thrust_ratio=$thrust_ratio"
    @test_broken sym_err < 1e-8
    @test_broken thrust_ratio < 1e-8
end

@testset "Thrust is continuous in surface tension as sigma -> 0" begin
    t0 = flexible_solver(_params(-0.012; sf = 0.0)).thrust
    t1 = flexible_solver(_params(-0.012; sf = 1e-3)).thrust
    t2 = flexible_solver(_params(-0.012; sf = 1e-2)).thrust
    # The capillary terms scale linearly in sigma, so the offset shrinks with it.
    @test abs(t1 - t0) < abs(t2 - t0) + eps()
    @test abs(t1 - t0) / abs(t0) < 0.01
end
