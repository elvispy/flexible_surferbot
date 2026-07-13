using Test
using Surferbot

@testset "analysis helpers" begin
    @test Surferbot.beam_asymmetry(1 + 0im, 1 + 0im) == 0
    @test Surferbot.beam_asymmetry(2 + 0im, 1 + 0im) > 0
    @test Surferbot.symmetric_antisymmetric_ratio(1 + 0im, 1 + 0im) > 10

    sweep = Surferbot.default_uncoupled_motor_position_EI_sweep()
    @test length(sweep.motor_position_list) == 25
    @test length(sweep.EI_list) == 57
    @test sweep.base_params.d == 0.0
    @test sweep.base_params.nu == 0.0

    mp = [0.1, 0.2, 0.3]
    EI = [1.0, 10.0]

    eta_left = ComplexF64[
        2.0  3.0
        0.8  1.7
        0.5  1.0
    ]
    eta_right = ComplexF64[
        -0.5  -1.0
        -1.2  -2.2
        -2.0  -3.0
    ]

    curve_EI, curve_mp, asymmetry, SA_ratio = Surferbot.extract_lowest_beam_curve(mp, EI, eta_left, eta_right)
    @test length(curve_EI) == 2
    @test curve_EI == EI
    @test all(0.1 .< curve_mp .< 0.3)
    @test size(asymmetry) == size(eta_left)
    @test size(SA_ratio) == size(eta_left)

    fake_result = (
        eta = ComplexF64[10, 20, 30, 40],
        metadata = (args = (x_contact = Bool[false, true, true, false],),),
    )
    metrics = Surferbot.beam_edge_metrics(fake_result)
    @test metrics.eta_left_beam == 20
    @test metrics.eta_right_beam == 30
    @test metrics.eta_left_domain == 10
    @test metrics.eta_right_domain == 40
end
