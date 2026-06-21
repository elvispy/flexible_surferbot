using Surferbot

# Purpose: run a generic parameter sweep with the Julia solver and save a native
# `JLD2` sweep artifact in `Julia/output` by default.

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
    main(save_dir=joinpath(@__DIR__, "..", "output"))

Run a small generic sweep and save the resulting native Julia sweep artifact.

Inputs:
- `save_dir`: directory where the `.jld2` sweep artifact should be written.

Edit the `base_params` and `grid` definitions in this script to change the
simulation family being swept.
"""
function main(save_dir::AbstractString=joinpath(@__DIR__, "..", "output"))
    save_dir = ensure_dir(normpath(save_dir))

    # Edit `base_params` and `grid` to define the sweep you want to run.
    base_params = FlexibleParams(
        omega = 2π * 10,
        domain_depth = 0.2,
        L_domain = 0.3,
        n = 41,
        M = 30,
    )

    grid = (
        motor_position = collect(range(0.0, stop=base_params.L_raft / 2, length=5)),
        EI = base_params.EI .* 10 .^ collect(range(-1, 1; length=5)),
    )

    artifact = sweep_parameters(
        base_params,
        grid;
        solver=flexible_solver,
        beam_metrics_fn=beam_edge_metrics,
        label="generic_sweep",
        save_path=joinpath(save_dir, "jld2", "run_sweep.jld2"),
    )

    println("Saved $(joinpath(save_dir, \"jld2\", \"run_sweep.jld2\"))")
    println("grid dimensions: $(size(artifact.summaries))")
    return artifact
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output")
    main(save_dir)
end
