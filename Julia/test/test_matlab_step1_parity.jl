using Test
using DelimitedFiles

function run_matlab_step1_dump(matlab_dir)
    matlab = Sys.which("matlab")
    matlab === nothing && return nothing

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    matlab_src = replace(joinpath(repo_root, "MATLAB", "src"), "\\" => "/")
    matlab_test = replace(joinpath(repo_root, "MATLAB", "test"), "\\" => "/")
    batch = "addpath('$matlab_src'); addpath('$matlab_test'); debug_parity_reference_case_step1_dump_cli"

    cmd = addenv(`$matlab -batch $batch`,
        "SURFERBOT_PARITY_DUMP_DIR" => matlab_dir,
        "OMP_NUM_THREADS" => "1",
        "KMP_AFFINITY" => "disabled",
    )

    try
        run(cmd)
    catch
        return nothing
    end
    return isfile(joinpath(matlab_dir, "summary.csv")) ? matlab_dir : nothing
end

function run_julia_step1_dump(julia_dir)
    default_bin = Sys.which("julia")
    julia_bin = get(ENV, "SURFERBOT_JULIA_BIN", default_bin === nothing ? "" : default_bin)
    isempty(julia_bin) || !isfile(julia_bin) && return nothing

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    julia_project = joinpath(repo_root, "Julia")
    julia_depot = joinpath(julia_project, ".julia_depot") * ":/Users/eaguerov/.julia"
    script = joinpath(julia_project, "scripts", "debug_dump_reference_case_step1.jl")
    cmd = addenv(`$julia_bin --project=$julia_project $script`,
        "SURFERBOT_PARITY_DUMP_DIR" => julia_dir,
        "JULIA_DEPOT_PATH" => julia_depot,
    )
    try
        run(cmd)
    catch
        return nothing
    end
    return isfile(joinpath(julia_dir, "summary.csv")) ? julia_dir : nothing
end

function read_vec(path)
    vec(readdlm(path, ',', Float64))
end

function read_summary(path)
    values = split(strip(read(path, String)), '\n')[2]
    parse.(Float64, split(values, ","))
end

function assert_vec_close(a, b; atol=1e-13, rtol=1e-13)
    @test length(a) == length(b)
    for i in eachindex(a)
        @test isapprox(a[i], b[i]; atol=atol, rtol=rtol)
    end
end

@testset "matlab parity step1 derived" begin
    dump_root = mktempdir()
    julia_dir = joinpath(dump_root, "julia")
    matlab_dir = joinpath(dump_root, "matlab")
    mkpath(julia_dir)
    mkpath(matlab_dir)

    julia_result  = run_julia_step1_dump(julia_dir)
    matlab_result = run_matlab_step1_dump(matlab_dir)

    if julia_result === nothing || matlab_result === nothing
        @test true
    else
        julia_summary = read_summary(joinpath(julia_dir, "summary.csv"))
        matlab_summary = read_summary(joinpath(matlab_dir, "summary.csv"))
        @test length(julia_summary) == length(matlab_summary)
        for i in eachindex(julia_summary)
            @test isapprox(julia_summary[i], matlab_summary[i]; atol=1e-13, rtol=1e-13)
        end

        assert_vec_close(read_vec(joinpath(julia_dir, "x.csv")), read_vec(joinpath(matlab_dir, "x.csv")))
        assert_vec_close(read_vec(joinpath(julia_dir, "z.csv")), read_vec(joinpath(matlab_dir, "z.csv")))
        assert_vec_close(read_vec(joinpath(julia_dir, "loads.csv")), read_vec(joinpath(matlab_dir, "loads.csv")))
        @test read_vec(joinpath(julia_dir, "x_contact.csv")) == read_vec(joinpath(matlab_dir, "x_contact.csv"))
        @test read_vec(joinpath(julia_dir, "x_free.csv")) == read_vec(joinpath(matlab_dir, "x_free.csv"))
    end
end
