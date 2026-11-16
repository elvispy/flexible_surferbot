using Test

include("test_package.jl")
include("test_derive_params.jl")
include("test_flexible_system.jl")
include("test_flexible_solver.jl")
include("test_analysis_helpers.jl")
include("test_modal_decomposition.jl")
include("test_root_tracing.jl")
include("test_capillary_coupling.jl")
include("test_migration.jl")
include("test_sweep.jl")
include("test_video.jl")
include("test_optimization_objective.jl")
include("test_optimization_gradients.jl")
include("test_graded_beam_params.jl")
include("test_thrust_sxx_agreement.jl")
include("test_edge_bc_physics.jl")
include("test_system_residual.jl")
include("test_beam_bcs.jl")
include("test_symmetry.jl")

# Heavy solves (1.5-8 GB each, minutes): off by default so a plain Pkg.test() cannot exhaust
# memory. Enable on a cluster node with
#     SURFERBOT_HEAVY_TESTS=1 julia --heap-size-hint=<80% of job memory> ...
# The guard below refuses to run them on a machine with under 24 GB of RAM.
if get(ENV, "SURFERBOT_HEAVY_TESTS", "0") == "1"
    if Sys.total_memory() >= 24 * 2^30
        include("test_mirror_symmetry.jl")
        include("test_energy_balance.jl")
        include("test_momentum_balance.jl")
        include("test_depth_convergence.jl")
    else
        @warn "Skipping heavy physics tests: less than 24 GB of RAM"
    end
else
    @info "Skipping heavy physics tests (set SURFERBOT_HEAVY_TESTS=1 to run)"
end

# Symbolic re-derivation of the analytical results reported in the paper.
include("test_paper_derivations.jl")
