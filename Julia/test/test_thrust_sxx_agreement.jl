using Test
using Surferbot

@testset "Thrust/Sxx agreement across frequency and surface tension" begin
    inviscid_cases = [
        (omega = 2 * pi * 10.0,  sigma = 72.2e-3, label = "10 Hz capgrav inviscid"),
        (omega = 2 * pi * 40.0,  sigma = 72.2e-3, label = "40 Hz capgrav inviscid"),
        (omega = 2 * pi * 80.0,  sigma = 72.2e-3, label = "80 Hz cap inviscid"),
    ]

    for c in inviscid_cases
        params = FlexibleParams(
            sigma         = c.sigma,
            rho           = 1000.0,
            nu            = 0.0,
            g             = 9.81,
            L_raft        = 0.05,
            d             = 0.03,
            omega         = c.omega,
            motor_position = 0.015,
            motor_inertia  = 0.13e-3 * 2.5e-3,
        )
        result = flexible_solver(params)
        args   = result.metadata.args

        T_over_d = result.thrust / Float64(args.d)
        m    = beam_edge_metrics(result)
        k    = Float64(real(args.k))
        pref = Float64(args.rho) * Float64(args.g) / 4 + 3/4 * Float64(args.sigma) * k^2
        Sxx  = pref * (abs2(m.eta_left_domain) - abs2(m.eta_right_domain))

        denom   = max(abs(T_over_d), abs(Sxx), 1e-30)
        rel_err = abs(T_over_d - Sxx) / denom
        @info "$(c.label)  T/d=$(T_over_d)  Sxx=$(Sxx)  rel_err=$(rel_err)"
        @test rel_err < 0.10
    end

    # Viscous check: Sxx != T/d for viscous waves (attenuation reduces far-field amplitude),
    # so instead verify small nu barely perturbs thrust relative to inviscid baseline.
    r_inv = flexible_solver(FlexibleParams(
        sigma=72.2e-3, rho=1000.0, nu=0.0, g=9.81, L_raft=0.05, d=0.03,
        omega=2*pi*40.0, motor_position=0.015, motor_inertia=0.13e-3*2.5e-3,
    ))
    r_vis = flexible_solver(FlexibleParams(
        sigma=72.2e-3, rho=1000.0, nu=1e-6, g=9.81, L_raft=0.05, d=0.03,
        omega=2*pi*40.0, motor_position=0.015, motor_inertia=0.13e-3*2.5e-3,
    ))
    T_inv = r_inv.thrust / Float64(r_inv.metadata.args.d)
    T_vis = r_vis.thrust / Float64(r_vis.metadata.args.d)
    visc_rel_diff = abs(T_vis - T_inv) / abs(T_inv)
    @info "40 Hz viscous vs inviscid  T/d_inv=$(T_inv)  T/d_vis=$(T_vis)  rel_diff=$(visc_rel_diff)"
    @test visc_rel_diff < 0.01
end

@testset "Rigid motor-at-one-end: thrust pushes toward the OTHER end" begin
    # Anchored to physical intuition alone, not to any formula in the paper:
    # forcing near one end of a rigid raft should recoil the raft toward the
    # opposite end. This is the one check in this file that pins an absolute
    # sign without referencing any of the (previously buggy) radiation-stress
    # or asymmetry-factor formulas.
    make = pos -> FlexibleParams(
        sigma = 72.2e-3, rho = 1000.0, nu = 0.0, g = 9.81,
        L_raft = 0.05, d = 0.03, EI = Inf,
        motor_position = pos, motor_inertia = 0.13e-3 * 2.5e-3,
        omega = 2π * 40.0, n = 21, Nz = 15,
    )
    r_right_end = flexible_solver(make(+0.024))   # motor near the RIGHT end
    r_left_end  = flexible_solver(make(-0.024))   # motor near the LEFT end
    @info "Rigid, motor at right end: thrust=$(r_right_end.thrust)"
    @info "Rigid, motor at left end:  thrust=$(r_left_end.thrust)"
    @test r_right_end.thrust < 0   # pushes LEFT, away from the motor
    @test r_left_end.thrust  > 0   # pushes RIGHT, away from the motor
end

@testset "alpha and F_T share sign (the bug this guards against)" begin
    # A previous bug had `beam_asymmetry` return the opposite sign from the
    # (independently, near-field-derived) thrust -- every internal
    # self-consistency check still "agreed with itself" under that bug, so a
    # cross-quantity check against F_T is what actually catches it.
    cases = [
        (EI = Inf,                                motor_position =  0.024, label = "rigid, motor right"),
        (EI = Inf,                                motor_position = -0.024, label = "rigid, motor left"),
        (EI = 3.0e9 * 3e-2 * 9e-4^3 / 12,          motor_position =  0.015, label = "flexible, motor right"),
        (EI = 3.0e9 * 3e-2 * 9e-4^3 / 12,          motor_position = -0.015, label = "flexible, motor left"),
    ]
    for c in cases
        params = FlexibleParams(
            sigma = 72.2e-3, rho = 1000.0, nu = 0.0, g = 9.81,
            L_raft = 0.05, d = 0.03, EI = c.EI,
            motor_position = c.motor_position, motor_inertia = 0.13e-3 * 2.5e-3,
            omega = 2π * 40.0, n = 21, Nz = 15,
        )
        result = flexible_solver(params)
        m = beam_edge_metrics(result)
        alpha = beam_asymmetry(m.eta_left_domain, m.eta_right_domain)
        @info "$(c.label)  thrust=$(result.thrust)  alpha=$(alpha)"
        @test sign(result.thrust) == sign(alpha)
    end
end
