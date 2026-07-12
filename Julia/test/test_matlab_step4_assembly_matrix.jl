using Test
using DelimitedFiles
using Surferbot
using SparseArrays

function run_matlab_step4_dump(matlab_dir)
    matlab = Sys.which("matlab")
    matlab === nothing && return nothing

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    matlab_src = replace(joinpath(repo_root, "MATLAB", "src"), "\\" => "/")
    matlab_test = replace(joinpath(repo_root, "MATLAB", "test"), "\\" => "/")
    batch = "addpath('$matlab_src'); addpath('$matlab_test'); debug_parity_assembly_step4_dump_cli"

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
    return isfile(joinpath(matlab_dir, "A_triplets.csv")) ? matlab_dir : nothing
end

function read_triplets(path)
    data = readdlm(path, ',', Float64)
    rows = Int.(data[:, 1])
    cols = Int.(data[:, 2])
    vals = ComplexF64.(data[:, 3], data[:, 4])
    return rows, cols, vals
end

function filter_triplets(rows, cols, vals; atol=1e-16)
    keep = abs.(vals) .>= atol
    return rows[keep], cols[keep], vals[keep]
end

function relative_error(a, b; floor=1e-15)
    scale = max(abs(a), abs(b), floor)
    return abs(a - b) / scale
end

function assert_rel_close(a, b; rtol=1e-12, floor=1e-15)
    @test relative_error(a, b; floor=floor) <= rtol
end

function read_complex_vec(path)
    data = readdlm(path, ',', Float64)
    ComplexF64.(data[:, 1], data[:, 2])
end

@testset "matlab parity step4 assembly matrix" begin
    dump_root = mktempdir()
    matlab_dir = joinpath(dump_root, "matlab")
    mkpath(matlab_dir)

    matlab_result = run_matlab_step4_dump(matlab_dir)
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

        j_i, j_j, j_v = findnz(system.A)
        m_i, m_j, m_v = read_triplets(joinpath(matlab_dir, "A_triplets.csv"))
        j_i, j_j, j_v = filter_triplets(j_i, j_j, j_v)
        m_i, m_j, m_v = filter_triplets(m_i, m_j, m_v)
        @test length(j_i) == length(m_i)
        @test j_i == m_i
        @test j_j == m_j
        for k in eachindex(j_v)
            assert_rel_close(real(j_v[k]), real(m_v[k]); rtol=1e-12)
            assert_rel_close(imag(j_v[k]), imag(m_v[k]); rtol=1e-12)
        end

        m_b = read_complex_vec(joinpath(matlab_dir, "b.csv"))
        @test length(system.b) == length(m_b)
        for k in eachindex(system.b)
            assert_rel_close(real(system.b[k]), real(m_b[k]); rtol=1e-12)
            assert_rel_close(imag(system.b[k]), imag(m_b[k]); rtol=1e-12)
        end
    end
end
