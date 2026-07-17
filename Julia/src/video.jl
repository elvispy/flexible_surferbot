module RunVideo

using Dates
using JLD2
using Printf

if !haskey(ENV, "GKSwstype") || isempty(ENV["GKSwstype"])
    ENV["GKSwstype"] = "100"
end

export SurferbotRunRecord,
       normalize_run,
       write_provenance_json,
       render_surferbot_run

function ensure_plots_backend!()
    if !isdefined(@__MODULE__, :Plots)
        @eval import Plots
    end
    plots = Base.invokelatest(getfield, @__MODULE__, :Plots)
    Base.invokelatest(plots.gr)
    return plots
end

"""
    SurferbotRunRecord

In-memory record for Surferbot simulation runs, used for video rendering.

# Fields
- `U`: Mean drift speed.
- `x`: Horizontal grid coordinates (meters).
- `eta`: Complex surface elevation amplitudes.
- `args`: Metadata and physical parameters.
- `source`: Provenance information (type and path).
"""
struct SurferbotRunRecord
    U::Float64
    x::Vector{Float64}
    eta::Vector{ComplexF64}
    args::NamedTuple
    source::NamedTuple
end

"""
    repo_root()

Return the absolute path to the repository root.
"""
repo_root() = abspath(joinpath(@__DIR__, "..", ".."))

"""
    maybe_get(nt, name::Symbol, default=nothing)

Safely extract a property from a NamedTuple, returning a default if it doesn't exist.
"""
function maybe_get(nt, name::Symbol, default=nothing)
    hasproperty(nt, name) ? getproperty(nt, name) : default
end

"""
    to_namedtuple(x)

Convert a Dict or NamedTuple to a NamedTuple.
"""
function to_namedtuple(x)
    if x isa NamedTuple
        return x
    elseif x isa AbstractDict
        keys_vec = Symbol.(collect(keys(x)))
        vals_vec = collect(values(x))
        return NamedTuple{Tuple(keys_vec)}(Tuple(vals_vec))
    else
        error("Cannot convert $(typeof(x)) to NamedTuple")
    end
end

"""
    load_saved_run(path::AbstractString)

Load a simulation result from a JLD2 file or directory.
"""
function load_saved_run(path::AbstractString)
    if isdir(path)
        for candidate in ("results.jld2", "run.jld2", "result.jld2")
            file = joinpath(path, candidate)
            if isfile(file)
                return JLD2.load(file)
            end
        end
        jld2_files = filter(f -> endswith(lowercase(f), ".jld2"), readdir(path; join=true))
        length(jld2_files) == 1 || error("Expected exactly one JLD2 file in $path or a standard run filename.")
        return JLD2.load(first(jld2_files))
    elseif isfile(path)
        return JLD2.load(path)
    else
        error("Run path not found: $path")
    end
end

"""
    normalize_run(input; source_kind="unknown", source_path=nothing)

Convert various result formats (JLD2, Dict, FlexibleResult) into a `SurferbotRunRecord`.
"""
function normalize_run(input; source_kind::AbstractString="unknown", source_path::Union{Nothing,AbstractString}=nothing)
    if input isa SurferbotRunRecord
        return input
    elseif hasproperty(input, :metadata) && hasproperty(input, :U) && hasproperty(input, :x) && hasproperty(input, :eta)
        args = to_namedtuple(input.metadata.args)
        return SurferbotRunRecord(
            float(input.U),
            Float64.(input.x),
            ComplexF64.(input.eta),
            args,
            (kind = "FlexibleResult", path = nothing),
        )
    elseif input isa NamedTuple
        required = (:U, :x, :eta, :args)
        all(hasproperty(input, key) for key in required) || error("NamedTuple input must contain U, x, eta, and args")
        args = to_namedtuple(getproperty(input, :args))
        return SurferbotRunRecord(
            float(getproperty(input, :U)),
            Float64.(getproperty(input, :x)),
            ComplexF64.(getproperty(input, :eta)),
            args,
            (kind = source_kind, path = source_path),
        )
    elseif input isa AbstractString
        loaded = load_saved_run(input)
        return normalize_run(loaded; source_kind="path", source_path=input)
    elseif input isa AbstractDict
        if haskey(input, "result")
            return normalize_run(input["result"]; source_kind=source_kind, source_path=source_path)
        elseif haskey(input, "run")
            return normalize_run(input["run"]; source_kind=source_kind, source_path=source_path)
        elseif all(haskey(input, key) for key in ("U", "x", "eta", "args"))
            payload = (
                U = input["U"],
                x = input["x"],
                eta = input["eta"],
                args = input["args"],
            )
            return normalize_run(payload; source_kind=source_kind, source_path=source_path)
        else
            error("Unsupported saved run payload. Expected result/run or U/x/eta/args keys.")
        end
    else
        error("Unsupported run input type: $(typeof(input))")
    end
end

"""
    json_escape(s::AbstractString)

Escape special characters for JSON strings.
"""
function json_escape(s::AbstractString)
    io = IOBuffer()
    for c in collect(s)
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\b'
            print(io, "\\b")
        elseif c == '\f'
            print(io, "\\f")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

"""
    write_json_value(io, value; indent=0)

Recursively write values in JSON format to an IO stream.
"""
function write_json_value(io, value; indent::Int=0)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        if isfinite(value)
            print(io, value)
        elseif isnan(value)
            print(io, "\"NaN\"")
        elseif value > 0
            print(io, "\"Inf\"")
        else
            print(io, "\"-Inf\"")
        end
    elseif value isa AbstractString
        print(io, '"', json_escape(value), '"')
    elseif value isa Symbol
        print(io, '"', json_escape(String(value)), '"')
    elseif value isa AbstractVector
        if isempty(value)
            print(io, "[]")
        else
            print(io, "[\n")
            for (i, item) in enumerate(value)
                print(io, repeat(" ", indent + 2))
                write_json_value(io, item; indent=indent + 2)
                if i < length(value)
                    print(io, ",")
                end
                print(io, "\n")
            end
            print(io, repeat(" ", indent), "]")
        end
    elseif value isa NamedTuple
        fields = collect(pairs(value))
        if isempty(fields)
            print(io, "{}")
        else
            print(io, "{\n")
            for (i, (key, item)) in enumerate(fields)
                print(io, repeat(" ", indent + 2))
                print(io, "\"", json_escape(String(key)), "\": ")
                write_json_value(io, item; indent=indent + 2)
                if i < length(fields)
                    print(io, ",")
                end
                print(io, "\n")
            end
            print(io, repeat(" ", indent), "}")
        end
    elseif value isa AbstractDict
        pairs_vec = collect(pairs(value))
        if isempty(pairs_vec)
            print(io, "{}")
        else
            print(io, "{\n")
            for (i, (key, item)) in enumerate(pairs_vec)
                print(io, repeat(" ", indent + 2))
                print(io, "\"", json_escape(String(key)), "\": ")
                write_json_value(io, item; indent=indent + 2)
                if i < length(pairs_vec)
                    print(io, ",")
                end
                print(io, "\n")
            end
            print(io, repeat(" ", indent), "}")
        end
    elseif value isa Complex
        print(io, '"', json_escape(string(value)), '"')
    else
        write_json_value(io, string(value); indent=indent)
    end
end

"""
    git_commit_hash()

Return the current Git commit hash or "unknown".
"""
function git_commit_hash()
    try
        return strip(readchomp(`git -C $(repo_root()) rev-parse HEAD`))
    catch
        return "unknown"
    end
end

"""
    build_provenance(record; output_basename, output_dir, script_name)

Build a metadata NamedTuple for provenance tracking.
"""
function build_provenance(record::SurferbotRunRecord; output_basename::AbstractString, output_dir::AbstractString, script_name::AbstractString)
    return (
        script_name = script_name,
        output_basename = output_basename,
        output_dir = output_dir,
        generated_at = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        git_commit = git_commit_hash(),
        source = record.source,
        U = record.U,
        thrust = maybe_get(record.args, :thrust, nothing),
        power = maybe_get(record.args, :power, nothing),
        args = record.args,
    )
end

"""
    write_provenance_json(path, provenance)

Write provenance metadata to a JSON file.
"""
function write_provenance_json(path::AbstractString, provenance)
    open(path, "w") do io
        write_json_value(io, provenance; indent=0)
        print(io, "\n")
    end
    return path
end

"""
    plot_frame(record, t; omega, x_contact_mask, motor_idx, show_motor)

Generate a single frame plot for the simulation at time `t`.
The raft is colour-coded by log₁₀(EI): dark = stiff, light grey = compliant.
"""
function plot_frame(record::SurferbotRunRecord, t::Real; omega::Real, x_contact_mask, motor_idx::Union{Nothing,Int}, show_motor::Bool)
    Plots = ensure_plots_backend!()
    scaleY = 1e6
    y        = real.(record.eta .* exp.(1im * omega * t)) .* scaleY
    x_scaled = record.x .* 1e2
    y_limit  = maximum(abs.(record.eta)) * scaleY * 1.1

    # ── Water surface ─────────────────────────────────────────────────────────
    p = Base.invokelatest(Plots.plot,
        x_scaled, y;
        fillrange  = -y_limit,
        fillcolor  = :steelblue,
        fillalpha  = 0.18,
        color      = :steelblue4,
        linewidth  = 2.0,
        label      = false,
        xlabel     = "x  (cm)",
        ylabel     = "η  (μm)",
        ylim       = (-y_limit, y_limit * 1.45),
        xlim       = (first(x_scaled), last(x_scaled)),
        title      = @sprintf("f = %.1f Hz     t = %.3f s     U = %.3f mm/s",
                               omega / (2π), t, record.U * 1e3),
        legend     = false,
        background_color_legend = :white,
        size       = (1400, 520),
        dpi        = 150,
        background_color = :white,
        framestyle = :box,
        grid       = false,
        guidefontsize  = 29,
        tickfontsize   = 24,
        titlefontsize  = 26,
        legendfontsize = 24,
        fontfamily = "Computer Modern",
        left_margin    = Base.invokelatest(*, 18, Plots.mm),
        top_margin     = Base.invokelatest(*, 4, Plots.mm),
        bottom_margin  = Base.invokelatest(*, 18, Plots.mm),
    )

    # ── Raft coloured by log₁₀(EI): dark = stiff, light grey = compliant ─────
    if !isempty(x_contact_mask) && any(x_contact_mask)
        x_raft = x_scaled[x_contact_mask]
        y_raft = y[x_contact_mask]
        n_raft = length(x_raft)

        EI_raw = maybe_get(record.args, :EI, nothing)
        if EI_raw !== nothing
            EI_vec = EI_raw isa AbstractVector ? Float64.(EI_raw) : fill(Float64(EI_raw), n_raft)

            # Replace Inf / NaN with a large sentinel before taking log
            EI_fin  = [isfinite(e) && e > 0 ? e : 1e30 for e in EI_vec]
            log_EI  = log10.(EI_fin)

            lo, hi  = extrema(log_EI)
            span    = hi - lo
            # norm_EI: 0 = most compliant (light grey), 1 = stiffest (dark)
            norm_EI = span < 1e-10 ? zeros(n_raft) : (log_EI .- lo) ./ span

            Base.invokelatest(Plots.scatter!, p, x_raft, y_raft;
                     marker_z          = norm_EI,
                     colormap          = [:gainsboro, :black],
                     clims             = (0.0, 1.0),
                     colorbar          = false,
                     markersize         = 7,
                     markerstrokewidth  = 0,
                     label              = false)
        else
            # Fallback for scalar or missing EI
            Base.invokelatest(Plots.plot!, p, x_raft, y_raft; color = :black, linewidth = 8, label = false)
        end
    end

    # ── Motor forcing: Re{f̂(x) e^{iωt}}, nondimensionalized, as arrows ────────
    # The motor sits above the raft and always pushes or pulls from there, so
    # each arrow occupies a fixed region just above the raft surface (never
    # dipping below it) rather than swinging above and below y0. Only the
    # arrowhead's position within that fixed region encodes the direction:
    # tip at the bottom (pointing down, toward the raft) for pressing, tip at
    # the top (pointing up, away from the raft) for pulling. Positive f acts
    # downward (paper sign convention), so a positive load is pressing
    # (negative dy in this plot's y-up coordinate).
    if show_motor && !isempty(x_contact_mask) && any(x_contact_mask)
        loads = maybe_get(record.args, :loads, nothing)
        L_c   = maybe_get(record.args, :L_c, nothing)
        m_c   = maybe_get(record.args, :m_c, nothing)
        t_c   = maybe_get(record.args, :t_c, nothing)
        if loads !== nothing && L_c !== nothing && m_c !== nothing && t_c !== nothing
            F_c          = m_c * L_c / t_c^2
            loads_nondim = loads .* L_c ./ F_c            # real, nondimensional load density
            f_t          = loads_nondim .* cos(omega * t) # Re{f̂ e^{iωt}}, f̂ real (in-phase forcing)
            peak         = maximum(abs.(loads_nondim))

            if peak > 0
                x_raft = x_scaled[x_contact_mask]
                y_raft = y[x_contact_mask]
                # Show a handful of evenly-spaced arrows across the visually
                # significant (>5% of peak) part of the Gaussian load
                # footprint, rather than one per (dense) solver grid point.
                sig_idx  = findall(abs.(loads_nondim) .> 0.05 * peak)
                n_arrows = min(length(sig_idx), 8)
                pick     = sig_idx[round.(Int, range(1, length(sig_idx); length = n_arrows))]
                arrow_scale = 0.8 * y_limit / peak
                dy = -f_t[pick] .* arrow_scale

                # Drawn manually (shaft + a small filled triangle at the tip)
                # rather than via quiver!'s built-in arrowheads: GR's quiver
                # recipe does not honor Plots.arrow()'s headlength/headwidth,
                # so the heads stay oversized regardless of what's passed.
                x_span         = last(x_scaled) - first(x_scaled)
                head_height    = 0.05 * y_limit
                head_halfwidth = 0.006 * x_span
                gap            = 0.05 * y_limit  # clearance above the raft surface

                shaft_xs = Float64[]
                shaft_ys = Float64[]
                for i in eachindex(pick)
                    x0  = x_raft[pick[i]]
                    y0  = y_raft[pick[i]]
                    dyi = dy[i]
                    actual_head  = min(head_height, abs(dyi))
                    arrow_bottom = y0 + gap
                    arrow_top    = arrow_bottom + abs(dyi)

                    if dyi >= 0
                        # Pulling: tip at the top, pointing up, away from the raft.
                        tip_y = arrow_top
                        base_y = tip_y - actual_head
                        shaft_other_y = arrow_bottom
                    else
                        # Pressing: tip at the bottom, pointing down, toward the raft.
                        tip_y = arrow_bottom
                        base_y = tip_y + actual_head
                        shaft_other_y = arrow_top
                    end

                    append!(shaft_xs, (x0, x0, NaN))
                    append!(shaft_ys, (shaft_other_y, base_y, NaN))

                    tri = Base.invokelatest(Plots.Shape,
                        [x0 - head_halfwidth, x0 + head_halfwidth, x0],
                        [base_y, base_y, tip_y])
                    Base.invokelatest(Plots.plot!, p, tri;
                             color = :orangered, linecolor = :orangered, label = false)
                end

                Base.invokelatest(Plots.plot!, p, shaft_xs, shaft_ys;
                         color     = :orangered,
                         linewidth = 1.5,
                         label     = false)
            end
        end
    end

    return p
end

"""
    render_surferbot_run(input; outdir=pwd(), basename="waves", fps=30, duration_periods=10, nframes=nothing, script_name=...)

Render a simulation run as an MP4 video with provenance metadata.

# Arguments
- `input`: Run record or path to JLD2 file.
- `outdir`: Directory to save outputs (default: current directory).
- `basename`: Base name for files (default: "waves").
- `fps`: Frames per second (default: 30).
- `duration_periods`: Number of periods to simulate (default: 10).
- `nframes`: Total number of frames (overrides `duration_periods`).
- `script_name`: Name of the script calling this function.

# Returns
- A NamedTuple `(mp4 = path, json = path)`.
"""
function render_surferbot_run(input; outdir::AbstractString=pwd(), basename::AbstractString="waves", fps::Int=30, duration_periods::Real=10, nframes::Union{Nothing,Int}=nothing, script_name::AbstractString=Base.basename(PROGRAM_FILE))
    Plots = ensure_plots_backend!()
    record = normalize_run(input)
    mkpath(outdir)
    mp4_path = joinpath(outdir, basename * ".mp4")
    json_path = joinpath(outdir, basename * ".json")
    provenance = build_provenance(record; output_basename=basename, output_dir=outdir, script_name=script_name)
    write_provenance_json(json_path, provenance)

    omega = maybe_get(record.args, :omega, nothing)
    omega === nothing && error("Run metadata must include omega for video rendering.")
    omega = float(omega)

    contact_mask = Bool[]
    if hasproperty(record.args, :x_contact)
        contact_mask = collect(Bool.(getproperty(record.args, :x_contact)))
    end

    motor_position = maybe_get(record.args, :motor_position, nothing)
    motor_idx = motor_position === nothing ? nothing : argmin(abs.(record.x .- float(motor_position)))

    total_frames = something(nframes, Int(round(duration_periods * fps)))
    tvec = range(0, stop = duration_periods * (2π / omega), length = total_frames)

    anim = Base.invokelatest(Plots.Animation)
    for t in tvec
        frame_plot = plot_frame(record, t; omega = omega, x_contact_mask = contact_mask, motor_idx = motor_idx, show_motor = motor_idx !== nothing)
        Base.invokelatest(Plots.frame, anim, frame_plot)
    end

    Base.invokelatest(Plots.mp4, anim, mp4_path; fps = fps)
    return (mp4 = mp4_path, json = json_path)
end

end # module
