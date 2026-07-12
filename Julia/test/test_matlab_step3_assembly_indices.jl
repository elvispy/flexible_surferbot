using Test
using DelimitedFiles
using Surferbot

function run_matlab_step3_dump(matlab_dir)
    matlab = Sys.which("matlab")
    matlab === nothing && return nothing

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    matlab_src = replace(joinpath(repo_root, "MATLAB", "src"), "\\" => "/")
    matlab_test = replace(joinpath(repo_root, "MATLAB", "test"), "\\" => "/")
    batch = "addpath('$matlab_src'); addpath('$matlab_test'); debug_parity_assembly_step3_dump_cli"

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
    return isfile(joinpath(matlab_dir, "idxBulk.csv")) ? matlab_dir : nothing
end

function read_vec_int(path)
    Int.(vec(readdlm(path, ',', Float64)))
end

function read_vec_bool(path)
    Bool.(Int.(vec(readdlm(path, ',', Float64))))
end

@testset "matlab parity step3 assembly indices" begin
    dump_root = mktempdir()
    matlab_dir = joinpath(dump_root, "matlab")
    mkpath(matlab_dir)

    matlab_result = run_matlab_step3_dump(matlab_dir)
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
            L_domain = 0.3,
        )
        system = assemble_flexible_system(params)

        @test vec(system.masks.topMask) == read_vec_bool(joinpath(matlab_dir, "topMask.csv"))
        @test vec(system.masks.freeMask) == read_vec_bool(joinpath(matlab_dir, "freeMask.csv"))
        @test vec(system.masks.contactMask) == read_vec_bool(joinpath(matlab_dir, "contactMask.csv"))
        @test vec(system.masks.bottomMask) == read_vec_bool(joinpath(matlab_dir, "bottomMask.csv"))
        @test vec(system.masks.leftEdgeMask) == read_vec_bool(joinpath(matlab_dir, "leftEdgeMask.csv"))
        @test vec(system.masks.rightEdgeMask) == read_vec_bool(joinpath(matlab_dir, "rightEdgeMask.csv"))
        @test vec(system.masks.bulkMask) == read_vec_bool(joinpath(matlab_dir, "bulkMask.csv"))
        @test system.indices.idxBulk == read_vec_int(joinpath(matlab_dir, "idxBulk.csv"))
        @test system.indices.idxBottom == read_vec_int(joinpath(matlab_dir, "idxBottom.csv"))
        @test system.indices.idxLeftEdge == read_vec_int(joinpath(matlab_dir, "idxLeftEdge.csv"))
        @test system.indices.idxRightEdge == read_vec_int(joinpath(matlab_dir, "idxRightEdge.csv"))
        @test system.indices.idxContact == read_vec_int(joinpath(matlab_dir, "idxContact.csv"))
        @test system.indices.idxLeftFreeSurf == read_vec_int(joinpath(matlab_dir, "idxLeftFreeSurf.csv"))
        @test system.indices.idxRightFreeSurf == read_vec_int(joinpath(matlab_dir, "idxRightFreeSurf.csv"))
    end
end
