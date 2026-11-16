using Test
using Surferbot

# Energy conservation as a function of surface tension.
#
# For an undamped (nu = 0) model the mean power delivered by the motor must equal the
# mean power radiated to the two far fields,
#     P_rad = d * c_g * E * (|eta_L|^2 + |eta_R|^2),   E = (rho*g + sigma*k^2)/2,
# with c_g the finite-depth group velocity, so P_in / P_rad = 1. Because the edge forces do work
# on the raft, a wrong sign in the capillary shear condition at either edge makes this ratio
# drift with sigma (3% to 65% by sigma x100 for the former right-edge sign). The test takes no
# sign convention from the manuscript.

const _EN_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

function _energy_ratio(sigma_factor, ei_factor)
    p00 = FlexibleParams(motor_position = -0.012, EI = _EN_EI0 * ei_factor,
                         sigma = 72.2e-3 * sigma_factor, nu = 0.0)
    dd = Surferbot.derive_params(p00)
    H = dd.domain_depth
    p0 = FlexibleParams(motor_position = -0.012, EI = _EN_EI0 * ei_factor,
                        sigma = 72.2e-3 * sigma_factor, nu = 0.0,
                        dz_surface = 0.1 / real(dd.k), z_growth = 1.1)
    r = solve_cached(p0)
    a = r.metadata.args
    m = beam_edge_metrics(r)
    k = Float64(real(a.k))
    rho, g, sig, d = Float64(a.rho), Float64(a.g), Float64(a.sigma), Float64(a.d)
    w(kk) = sqrt((g * kk + sig * kk^3 / rho) * tanh(kk * H))
    h = 1e-4 * k
    cg = (w(k + h) - w(k - h)) / (2h)
    E = 0.5 * (rho * g + sig * k^2)
    Prad = d * cg * E * (abs2(m.eta_left_domain) + abs2(m.eta_right_domain))
    return Float64(r.power) / Prad
end

@testset "Absolute energy balance: -power/(F_c/L_c) equals radiated power" begin
    q = _energy_ratio(1.0, 1.0)
    GC.gc()
    @info "absolute energy ratio P_in/P_rad = $q"
    @test abs(q - 1) < 2e-3
end

@testset "Energy balance is independent of surface tension" begin
    for ei_factor in (3e-4, 1.0)
        q1 = _energy_ratio(1.0, ei_factor)
        GC.gc()
        for sf in (10.0, 100.0)
            q = _energy_ratio(sf, ei_factor)
            GC.gc()
            drift = abs(q / q1 - 1)
            @info "energy ratio drift: EI x$ei_factor, sigma x$sf = $drift"
            @test drift < 1e-3
        end
    end
end
