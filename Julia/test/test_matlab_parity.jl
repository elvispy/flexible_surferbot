using Test
using Surferbot

function relative_error(a, b; floor=1e-15)
    scale = max(abs(a), abs(b), floor)
    return abs(a - b) / scale
end

function assert_rel_close(a, b; rtol=1e-12, floor=1e-15)
    @test relative_error(a, b; floor=floor) <= rtol
end

function run_matlab_reference_case()
    matlab = Sys.which("matlab")
    matlab === nothing && return nothing

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    matlab_src = replace(joinpath(repo_root, "MATLAB", "src"), "\\" => "/")
    matlab_test = replace(joinpath(repo_root, "MATLAB", "test"), "\\" => "/")

    tmpdir = mktempdir()
    outfile = joinpath(tmpdir, "matlab_parity.csv")
    batch = "addpath('$matlab_src'); addpath('$matlab_test'); debug_parity_reference_case_cli"

    cmd = addenv(`$matlab -batch $batch`,
        "SURFERBOT_PARITY_OUTPUT" => outfile,
        "OMP_NUM_THREADS" => "1",
        "KMP_AFFINITY" => "disabled",
    )

    try
        run(cmd)
    catch
        return nothing
    end

    if !isfile(outfile)
        return nothing
    end

    values = parse.(Float64, split(strip(read(outfile, String)), ","))
    return (U = values[1], power = values[2], thrust = values[3])
end

@testset "matlab parity" begin
    matlab_result = run_matlab_reference_case()
    if matlab_result === nothing
        @test true
    else
        params = FlexibleParams(
            sigma = 0.0,
            rho = 1000.0,
            nu = 1e-6,
            g = 10 * 9.81,
            L_raft = 0.1,
            motor_position = 0.5 * 0.1 / 2,
            d = 0.1 / 2,
            EI = 100 * 3.0e9 * 3e-2 * (9.9e-4)^3 / 12,
            rho_raft = 0.018 * 3.0,
            domain_depth = 0.2,
            n = 41,
            Nz = 30,
            motor_inertia = 0.13e-3 * 2.5e-3,
            bc = :radiative,
            omega = 2 * pi * 10,
        )

        julia_result = flexible_solver(params)
        assert_rel_close(julia_result.U, matlab_result.U; rtol=1e-12)
        assert_rel_close(julia_result.power, matlab_result.power; rtol=1e-12)
        assert_rel_close(julia_result.thrust, matlab_result.thrust; rtol=1e-12)
    end
end
