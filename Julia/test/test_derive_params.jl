using Test
using Surferbot

@testset "derive_params" begin
    params = FlexibleParams(
        omega = 2 * pi * 10.0,
        L_raft = 0.05,
        motor_position = 10.0,
    )

    derived = derive_params(params)

    @test isapprox(derived.d, params.d; atol=1e-12)
    @test isapprox(derived.motor_force, params.motor_inertia * params.omega^2; rtol=1e-12)
    @test derived.motor_position == params.L_raft / 2
    @test derived.domain_depth > 0.0
    @test derived.k isa Complex
    @test derived.n isa Int
    @test derived.Nz isa Int
    @test derived.Nx isa Int
    @test length(derived.x) == derived.Nx
    @test length(derived.z) == derived.Nz
    @test count(derived.x_contact) + count(derived.x_free) == derived.Nx - 2
    x_contact = derived.x[derived.x_contact]
    dx_contact = diff(x_contact)
    weights = vcat(0.5 * dx_contact[1], 0.5 .* (dx_contact[1:(end - 1)] .+ dx_contact[2:end]), 0.5 * dx_contact[end])
    @test isapprox(sum(derived.loads .* weights), derived.motor_force / derived.F_c; rtol=0.05)

    nd = derived.nd_groups
    @test isapprox(nd.Gamma, params.rho * params.L_raft^2 / params.rho_raft; rtol=1e-12)
    @test isapprox(nd.Lambda, derived.d / params.L_raft; rtol=1e-12)
end

@testset "derive_params depth selection converges at 80 Hz capillary regime" begin
    params = FlexibleParams(
        sigma = 72.2e-3,
        rho = 1000.0,
        nu = 0.0,
        g = 9.81,
        L_raft = 0.05,
        omega = 2 * pi * 80.0,
    )
    derived = derive_params(params)
    @test tanh(real(derived.k) * derived.domain_depth) >= 0.99
end

@testset "Dispersion relation ω² = (gk + σk³/ρ)·tanh(kh)" begin
    for (label, params) in [
        ("10 Hz gravity", FlexibleParams(sigma=0.0,     rho=1000.0, nu=0.0, g=9.81, omega=2π*10.0)),
        ("80 Hz capgrav", FlexibleParams(sigma=72.2e-3, rho=1000.0, nu=0.0, g=9.81, omega=2π*80.0)),
    ]
        derived = derive_params(params)
        k = real(derived.k)
        h = derived.domain_depth
        lhs = params.omega^2
        rhs = (params.g * k + params.sigma / params.rho * k^3) * tanh(k * h)
        @test abs(lhs - rhs) / lhs < 1e-10
    end
end
