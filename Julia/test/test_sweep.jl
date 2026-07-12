using Test
using Surferbot

@testset "sweep layer" begin
    base = FlexibleParams(
        motor_position = 0.01,
        EI = 2.0,
        n = 11,
        Nz = 5,
        domain_depth = 0.1,
        L_domain = 0.15,
    )

    overridden = apply_parameter_overrides(base, (motor_position = 0.02, EI = 3.0))
    @test overridden.motor_position == 0.02
    @test overridden.EI == 3.0
    @test overridden.n == base.n

    grid = (motor_position = [0.0, 0.01], EI = [1.0, 2.0, 3.0])
    combos = expand_parameter_grid(grid)
    @test length(combos) == 6
    @test first(combos) == (motor_position = 0.0, EI = 1.0)
    @test last(combos) == (motor_position = 0.01, EI = 3.0)

    fake_solver(params) = (
        U = params.motor_position + params.EI,
        power = -params.EI,
        thrust = params.motor_position - params.EI,
        eta = ComplexF64[1 + 0im, 2 + 0im, 3 + 0im, 4 + 0im],
    )
    fake_beam_metrics(result) = (
        eta_left_beam = 2 + 0im,
        eta_right_beam = 3 + 0im,
        eta_left_domain = 1 + 0im,
        eta_right_domain = 4 + 0im,
        eta_beam_ratio = 2 / 3,
        eta_domain_ratio = 1 / 4,
    )

    artifact = sweep_parameters(
        base,
        grid;
        solver=fake_solver,
        beam_metrics_fn=fake_beam_metrics,
        label="test_sweep",
    )
    @test artifact.label == "test_sweep"
    @test size(artifact.summaries) == (2, 3)
    summary = artifact.summaries[2, 3]
    @test summary.U == 3.01
    @test summary.power == -3.0
    @test summary.power_input == 3.0
    @test summary.thrust == -2.99
    @test summary.eta_left_beam == 2 + 0im
    @test summary.eta_right_domain == 4 + 0im

    mktempdir() do tmp
        path = joinpath(tmp, "sweep.jld2")
        save_sweep(path, artifact)
        loaded = load_sweep(path)
        @test loaded.label == artifact.label
        @test loaded.parameter_axes == artifact.parameter_axes
        @test size(loaded.summaries) == size(artifact.summaries)
        @test loaded.summaries[1, 1].thrust == artifact.summaries[1, 1].thrust
    end
end
