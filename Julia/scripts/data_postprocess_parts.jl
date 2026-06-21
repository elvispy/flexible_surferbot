using Surferbot
using JLD2
using CSV
using DataFrames
using Printf
using Base.Threads

"""
    postprocess_parts.jl

This definitive version correctly post-processes the partial .jld2 files.
It fixes all previous errors by:
1. Replicating the `derive_params` logic from `Surferbot.jl` to build the
   correct, flat `args` object.
2. Defensively checking for invalid `k_real` values to prevent crashes on
   specific physical parameter points.
"""

# --- Verbatim copies from Surferbot.jl to ensure correctness ---
function gaussian_load(x_0, width, x)
    return exp.(-((x .- x_0) / width) .^ 2)
end

function reconstruct_full_args(params::Surferbot.FlexibleParams{T}) where {T<:Real}
    motor_force = isnothing(params.motor_force) ? T(1) : params.motor_force
    d = isnothing(params.d) ? T(0.05) : params.d
    motor_position = clamp(params.motor_position, -params.L_raft / 2, params.L_raft / 2)

    L_c = params.L_raft
    t_c = 1 / params.omega
    m_c = params.rho_raft * L_c
    F_c = m_c * L_c / t_c^2

    current_depth = isnothing(params.domain_depth) ? T(2.5) * params.g / params.omega^2 : params.domain_depth
    k = Surferbot.Utils.dispersion_k(params.omega, params.g, current_depth, params.nu, params.sigma, params.rho)

    if isnothing(params.domain_depth)
        for _ in 1:50
            if !isfinite(real(k)) || tanh(real(k) * current_depth) < T(0.99)
                current_depth *= T(1.05)
                k = Surferbot.Utils.dispersion_k(params.omega, params.g, current_depth, params.nu, params.sigma, params.rho)
            else
                break
            end
        end
    end
    domain_depth = current_depth
    
    k_real = real(k)
    # DEFENSIVE CHECK for invalid wavenumber
    if !isfinite(k_real) || k_real <= 0
        @warn "Invalid k_real ($k_real) for EI=$(params.EI), motor_pos=$(params.motor_position). Skipping point."
        return nothing
    end

    res_n = 80
    n_nodes = isnothing(params.n) ? (max(res_n, ceil(Int, res_n / (2 * pi / k_real) * params.L_raft)) + 1) : params.n
    M_nodes = isnothing(params.M) ? ceil(Int, res_n * k_real * domain_depth) : params.M
    L_domain = isnothing(params.L_domain) ? min(3 * params.L_raft, round(20 * 2 * pi / k_real + params.L_raft; sigdigits=2)) : params.L_domain
    N_nodes = round(Int, (n_nodes - 1) * L_domain / params.L_raft) + 1
    
    x = collect(range(-L_domain / 2, stop = L_domain / 2, length = N_nodes))
    x_contact = abs.(x) .<= params.L_raft / 2
    loads = motor_force .* gaussian_load(params.motor_position, params.forcing_width, x[x_contact])

    return (
        sigma = params.sigma, rho = params.rho, omega = params.omega, nu = params.nu, g = params.g,
        L_raft = params.L_raft, d = d, EI = params.EI, rho_raft = params.rho_raft,
        x_contact = x_contact, x = x, loads = loads, motor_position = params.motor_position,
        n = n_nodes, M = M_nodes, N = N_nodes,
        ooa = params.ooa, bc = params.bc
    )
end

function reconstruct_params(base_params, part_data, motor_pos)
    ei = part_data["EI"]
    overrides = (motor_position = motor_pos, EI = ei)
    return Surferbot.Sweep.apply_parameter_overrides(base_params, overrides)
end

function process_part(part_path::AbstractString, base_params, motor_position_list, num_modes::Int)
    try
        part_data = load(part_path)
        results_chunk = part_data["results"]
        rows = []

        for (im, res) in enumerate(results_chunk)
            if isnothing(res) || !haskey(res, :metadata)
                continue
            end
            
            motor_pos = motor_position_list[im]
            params = reconstruct_params(base_params, part_data, motor_pos)
            
            args = reconstruct_full_args(params)
            if isnothing(args)
                continue
            end

            modal = Surferbot.Modal.decompose_raft_freefree_modes(
                args.x, res.eta, res.pressure, args.loads, args; 
                num_modes=num_modes, verbose=false
            )
            
            row_data = (
                log10_EI = log10(params.EI), xM_over_L = params.motor_position / params.L_raft,
                L_raft = params.L_raft, omega = params.omega, d = params.d, rho_raft = params.rho_raft,
            )
            
            for n in 0:(num_modes-1)
                row_data = merge(row_data, NamedTuple{(
                    Symbol("q_w$(n)_re"), Symbol("q_w$(n)_im"),
                    Symbol("Q_w$(n)_re"), Symbol("Q_w$(n)_im"),
                    Symbol("F_w$(n)_re"), Symbol("F_w$(n)_im"),
                )}((
                    modal.q_w[n+1].re, modal.q_w[n+1].im,
                    modal.Q_w[n+1].re, modal.Q_w[n+1].im,
                    modal.F_w[n+1].re, modal.F_w[n+1].im
                )))
            end
            push!(rows, row_data)
        end
        return rows
    catch e
        @error "Failed to process part $part_path: $e"
        showerror(stderr, e, catch_backtrace())
        return []
    end
end

function main(parts_dir::AbstractString, output_csv::AbstractString; num_modes::Int=8)
    part_files = filter(f -> endswith(f, ".jld2"), readdir(parts_dir, join=true))
    println("Found $(length(part_files)) parts in $parts_dir")
    
    base_params = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    nx = 100 
    motor_position_list = collect(range(0.0, 0.49; length=nx) .* base_params.L_raft)

    all_rows = Vector{Any}(undef, length(part_files))
    @threads for i in eachindex(part_files)
        @printf("Processing part %d/%d: %s
", i, length(part_files), basename(part_files[i]))
        all_rows[i] = process_part(part_files[i], base_params, motor_position_list, num_modes)
    end
    
    println("Collating results...")
    flat_rows = vcat(filter(!isnothing, all_rows)...)
    flat_rows = vcat(filter(r -> !isempty(r), flat_rows)...)
    
    if isempty(flat_rows)
        @warn "No data was processed. Output CSV will be empty."
        return
    end
    
    df = DataFrame(flat_rows)
    println("Writing to $output_csv...")
    CSV.write(output_csv, df)
    println("Done. Wrote $(size(df, 1)) rows to the master CSV.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    parts_dir = get(ARGS, 1, joinpath(@__DIR__, "..", "output", "jld2"))
    output_csv = get(ARGS, 2, joinpath(@__DIR__, "..", "output", "csv", "postprocess_parts.csv"))
    main(parts_dir, output_csv)
end
