using Surferbot

# Purpose: generate the default uncoupled `(x_M, EI)` sweep with the Julia
# solver and save it as a native `JLD2` artifact.

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
    main(save_dir=joinpath(@__DIR__, "..", "output");
         base_params_override=nothing,
         motor_position_list_override=nothing,
         EI_list_override=nothing,
         outfile="sweep_motor_position_EI_uncoupled.jld2")

Run the default uncoupled motor-position / stiffness sweep.

Inputs:
- `save_dir`: directory where the sweep artifact is written.
- `base_params_override`: optional replacement for the preset base parameter set.
- `motor_position_list_override`: optional list of motor positions to sweep.
- `EI_list_override`: optional list of flexural rigidities to sweep.
- `outfile`: output artifact filename within `save_dir`.
"""
function main(
    save_dir::AbstractString=joinpath(@__DIR__, "..", "output");
    base_params_override=nothing,
    motor_position_list_override=nothing,
    EI_list_override=nothing,
    outfile::AbstractString="sweep_motor_position_EI_uncoupled.jld2",
)
    save_dir = ensure_dir(normpath(save_dir))
    preset = default_uncoupled_motor_position_EI_sweep()
    base_tuple = isnothing(base_params_override) ? preset.base_params : base_params_override
    base_params = FlexibleParams(; base_tuple...)
    motor_position_list = isnothing(motor_position_list_override) ? preset.motor_position_list : collect(motor_position_list_override)
    EI_list = isnothing(EI_list_override) ? preset.EI_list : collect(EI_list_override)

    println("Running Julia uncoupled motor-position EI sweep")
    println("grid: $(length(motor_position_list)) motor positions x $(length(EI_list)) EI values")

    grid = (
        motor_position = motor_position_list,
        EI = EI_list,
    )
    return sweep_parameters(
        base_params,
        grid;
        solver=flexible_solver,
        beam_metrics_fn=beam_edge_metrics,
        label="uncoupled_motor_position_EI",
        save_path=joinpath(save_dir, "jld2", outfile),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output")
    main(save_dir)
end
