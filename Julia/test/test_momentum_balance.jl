using Test
using Surferbot

# Momentum balance: mean thrust on the raft must equal the net momentum flux carried away
# by the waves. The flux is computed directly from the solver's own velocity potential on
# a vertical cut (no plane-wave assumption),
#     S(x) = (rho/2) * int (|phi_x|^2 - |phi_z|^2) dz + (rho*g/4)|eta|^2 + (3/4) sigma |eta_x|^2,
# and compared with the near-field thrust, T/d = S(left) - S(right).
#
# Setup notes (measured): the cut must lie in the far field of a long domain, otherwise the
# first evanescent vertical mode contaminates it; the bath must be deep enough that the
# finite-depth part of S is negligible, otherwise the comparison also depends on depth.
# Coarse grids are enough: the residual is converged to ~0.2% from 0.35x to 1x resolution.

const _MB_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

function _momentum_case(; ei_factor, sigma_factor, depth_factor, L_domain, scale = 0.5)
    p0 = FlexibleParams(motor_position = -0.012, EI = _MB_EI0 * ei_factor,
                        sigma = 72.2e-3 * sigma_factor, nu = 0.0)
    dd = Surferbot.derive_params(p0)
    n = round(Int, dd.n * scale); n += iseven(n) ? 1 : 0
    Nz = round(Int, dd.Nz * scale * depth_factor)
    p = FlexibleParams(motor_position = -0.012, EI = _MB_EI0 * ei_factor,
                       sigma = 72.2e-3 * sigma_factor, nu = 0.0, n = n, Nz = Nz,
                       domain_depth = depth_factor * dd.domain_depth, L_domain = L_domain)
    r = solve_cached(p)
    a = r.metadata.args
    x, z, phi, phiz, eta = r.x, r.z, r.phi, r.phi_z, r.eta
    rho, g, sig = Float64(a.rho), Float64(a.g), Float64(a.sigma)
    dx, dz = x[2] - x[1], abs(z[2] - z[1])
    function S(i)
        D(f) = (-f[i + 2] + 8f[i + 1] - 8f[i - 1] + f[i - 2]) / (12dx)
        u2 = [abs2(D(view(phi, j, :))) for j in 1:size(phi, 1)]
        w2 = abs2.(phiz[:, i])
        integrand = u2 .- w2
        I = sum(integrand[1:(end - 1)] .+ integrand[2:end]) / 2 * dz
        return 0.5rho * I + 0.25rho * g * abs2(eta[i]) + 0.75sig * abs2(D(eta))
    end
    Lx = x[end]
    stations = (0.95, 0.85, 0.7)
    dS = [S(argmin(abs.(x .+ f * Lx))) - S(argmin(abs.(x .- f * Lx))) for f in stations]
    T = r.thrust / Float64(a.d)
    return (T = T, dS = dS)
end

@testset "Momentum balance: rigid raft, deep bath, long domain" begin
    for sf in (1.0, 30.0, 100.0)
        c = _momentum_case(ei_factor = Inf, sigma_factor = sf, depth_factor = 2.0, L_domain = 0.45)
        spread = (maximum(c.dS) - minimum(c.dS)) / abs(c.T)
        err = abs(c.T - c.dS[1]) / abs(c.T)
        @info "rigid momentum: sigma x$sf  T/d=$(c.T)  rel_err=$err  station_spread=$spread"
        @test spread < 5e-3   # flux is x-independent in the free region
        @test err < 1e-3
        GC.gc()
    end
end

@testset "Momentum balance: flexible raft at the reference sigma" begin
    c = _momentum_case(ei_factor = 3e-4, sigma_factor = 1.0, depth_factor = 2.0, L_domain = nothing)
    err = abs(c.T - c.dS[1]) / abs(c.T)
    @info "flexible momentum: sigma x1  rel_err=$err"
    @test err < 2e-3
    GC.gc()
end

@testset "Momentum balance: flexible raft at large sigma (open residual)" begin
    # Observed: the residual is stable in domain length and depth but nonzero, and its sign
    # depends on the right-edge condition. Cause not yet identified; do not relax the
    # tolerance, fix the model or explain the residual.
    for sf in (30.0, 100.0)
        c = _momentum_case(ei_factor = 3e-4, sigma_factor = sf, depth_factor = 2.0, L_domain = 0.45)
        err = abs(c.T - c.dS[1]) / abs(c.T)
        @info "flexible momentum: sigma x$sf  rel_err=$err"
        @test_broken err < 1e-3
        GC.gc()
    end
end
