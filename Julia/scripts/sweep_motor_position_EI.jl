using Surferbot
using Printf
using Base.Threads
using DelimitedFiles

"""
sweeper_modal_coefficients.jl

A memory-efficient, parallelized sweeper that streams modal coefficients 
(q, Q, F) directly to CSV. Designed to process large parameter grids (30,000+ points) 
without OOM errors. Supports Slurm Array Tasks.
"""

# Memory-efficient, parallelized sweeper for modal coefficients.
# Directly streams (q, Q, F) results to CSV to prevent OOM errors.

function generate_header(num_modes)
    header = ["log10_EI", "xM_over_L", "L_raft", "omega", "d", "rho_raft", "alpha"]
    # Add edge elevations (complex)
    append!(header, [
        "eta_1_beam_re", "eta_1_beam_im",
        "eta_1_domain_re", "eta_1_domain_im",
        "eta_end_beam_re", "eta_end_beam_im",
        "eta_end_domain_re", "eta_end_domain_im"
    ])
    for n in 0:(num_modes-1)
        append!(header, [
            "q_w$(n)_re", "q_w$(n)_im",
            "Q_w$(n)_re", "Q_w$(n)_im",
            "F_w$(n)_re", "F_w$(n)_im"
        ])
    end
    return join(header, ",")
end

function format_row(params, res, modal)
    metrics = Surferbot.Analysis.beam_edge_metrics(res)
    alpha = Surferbot.Analysis.beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)

    row_data = [
        log10(params.EI),
        params.motor_position / params.L_raft,
        params.L_raft,
        params.omega,
        isnothing(params.d) ? 0.0 : params.d,
        params.rho_raft,
        alpha
    ]

    # Append edge elevations (complex)
    append!(row_data, [
        real(metrics.eta_left_beam), imag(metrics.eta_left_beam),
        real(metrics.eta_left_domain), imag(metrics.eta_left_domain),
        real(metrics.eta_right_beam), imag(metrics.eta_right_beam),
        real(metrics.eta_right_domain), imag(metrics.eta_right_domain)
    ])
    
    # Extract W-basis coefficients
    # modal.q, modal.Q, modal.F are already in the projected basis
    for i in 1:length(modal.n)
        push!(row_data, real(modal.q[i]))
        push!(row_data, imag(modal.q[i]))
        push!(row_data, real(modal.Q[i]))
        push!(row_data, imag(modal.Q[i]))
        push!(row_data, real(modal.F[i]))
        push!(row_data, imag(modal.F[i]))
    end
    return join([@sprintf("%.12e", x) for x in row_data], ",")
end

function run_sweep(output_path, is_coupled; nx=100, nei=300, num_modes=8, task_id=nothing)
    L_raft = 0.05
    motor_position_list = collect(range(0.0, 0.49; length=nx) .* L_raft)
    # Shifted up by 1.5 decades vs. the original [-8, -2] window so that the
    # rigid-asymptote regime (log10(κ) ≳ 0.4 in the coupled case) is sampled.
    # log10(EI) ∈ [-6.5, -0.5]  ⇒  log10(κ) ∈ [-5.41, +0.59].
    logEI_range = range(-0.5, -6.5; length=nei)
    EI_list = 10.0 .^ logEI_range

    # Setup parameters
    if is_coupled
        base_params = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    else
        base_params = Surferbot.Analysis.default_uncoupled_motor_position_EI_sweep().base_params
    end

    # Determine which indices to run
    indices = isnothing(task_id) ? (1:nei) : (task_id:task_id)

    # Initialize CSV only if starting from scratch (no task_id or task_id=1)
    if isnothing(task_id) || task_id == 1
        open(output_path, "w") do io
            println(io, generate_header(num_modes))
        end
    end

    write_lock = ReentrantLock()
    
    if isnothing(task_id)
        println("Starting full sweep: $(is_coupled ? "Coupled" : "Uncoupled") -> $output_path")
        println("Grid: $nei EI points x $nx xM points ($((nei*nx)) total)")
        
        @threads for iei in indices
            process_ei_slice(iei, EI_list, motor_position_list, base_params, output_path, num_modes, write_lock, task_id, nx, nei)
        end
    else
        for iei in indices
            process_ei_slice(iei, EI_list, motor_position_list, base_params, output_path, num_modes, write_lock, task_id, nx, nei)
        end
    end
end

function process_ei_slice(iei, EI_list, motor_position_list, base_params, output_path, num_modes, write_lock, task_id, nx, nei)
    ei = EI_list[iei]
    rows_to_write = String[]
    
    for im in 1:nx
        params = Surferbot.Sweep.apply_parameter_overrides(base_params, (
            motor_position = motor_position_list[im],
            EI = ei
        ))
        
        try
            res = flexible_solver(params)
            modal = decompose_raft_freefree_modes(res; num_modes=num_modes, verbose=false)
            push!(rows_to_write, format_row(params, res, modal))
        catch e
            @warn "Failed at EI=$(ei), xM=$(motor_position_list[im]): $e"
        end
    end
    
    lock(write_lock) do
        open(output_path, "a") do io
            for row in rows_to_write
                println(io, row)
            end
        end
    end
    
    if isnothing(task_id)
        @printf("Completed EI index %3d/%3d (log10EI = %.2f)\n", iei, nei, log10(ei))
    else
        @printf("Task %d: Completed EI = %.2e\n", task_id, ei)
    end
end

function main()
    # 1. Parse CLI arguments
    # Usage: julia sweeper.jl [task_id] [uncoupled|coupled|both]
    task_id_str = get(ARGS, 1, "")
    task_id = isempty(task_id_str) ? nothing : parse(Int, task_id_str)
    
    mode = get(ARGS, 2, "both")

    output_dir = joinpath(@__DIR__, "..", "output", "csv")
    mkpath(output_dir)

    # 1. Uncoupled Sweep
    if mode == "both" || mode == "uncoupled"
        uncoupled_path = joinpath(output_dir, "sweeper_uncoupled_full_grid.csv")
        run_sweep(uncoupled_path, false; task_id=task_id)
    end

    # 2. Coupled Sweep
    if mode == "both" || mode == "coupled"
        coupled_path = joinpath(output_dir, "sweeper_coupled_full_grid.csv")
        run_sweep(coupled_path, true; task_id=task_id)
    end
    
    if isnothing(task_id)
        println("\nAll sweeps completed successfully.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
