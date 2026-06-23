using Test
using Surferbot

function make_fake_video_result()
    x = Float64[0.00, 0.01, 0.02, 0.03]
    z = Float64[0.0, -0.01]
    eta = ComplexF64[0.2 + 0.05im, 0.8 + 0.1im, 0.6 - 0.05im, 0.1 + 0.02im]
    phi = reshape(ComplexF64[1, 2, 3, 4, 5, 6, 7, 8], 2, 4)
    phi_z = reshape(ComplexF64[8, 7, 6, 5, 4, 3, 2, 1], 2, 4)
    pressure = ComplexF64[0.1, 0.2, 0.3, 0.4]
    args = (
        omega = 2 * pi * 5,
        x_contact = Bool[false, true, true, false],
        motor_position = 0.01,
        thrust = 1.25,
        power = -0.75,
        L_raft = 0.05,
    )
    metadata = (args = args, system = nothing)
    return FlexibleResult(0.0125, -0.75, 1.25, x, z, phi, phi_z, eta, pressure, 0.0, 0.0, metadata)
end

@testset "video rendering" begin
    result = make_fake_video_result()
    record = normalize_run(result)
    @test record.U == result.U
    @test record.source.kind == "FlexibleResult"
    @test record.args.motor_position == 0.01
    @test length(record.x) == 4

    mktempdir() do tmp
        outputs = render_surferbot_run(result; outdir=tmp, basename="waves", fps=2, duration_periods=1, nframes=3, script_name="test_video.jl")
        @test isfile(outputs.mp4)
        @test filesize(outputs.mp4) > 0
        @test isfile(outputs.json)
        @test filesize(outputs.json) > 0
        json = read(outputs.json, String)
        @test occursin("\"output_basename\": \"waves\"", json)
        @test occursin("\"script_name\": \"test_video.jl\"", json)
        @test occursin("\"motor_position\"", json)
        @test occursin("\"git_commit\"", json)
    end
end

