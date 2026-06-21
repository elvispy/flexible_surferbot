using Surferbot, Printf

# Purpose: render a stationary-domain MP4 animation for one Surferbot run,
# writing `basename.mp4` plus a same-stem provenance JSON sidecar.
#
# Usage — two modes:
#
#   1. From a saved run file:
#      julia --project=. scripts/plot_surferbot_run.jl path/to/run.jld2 [OPTIONS]
#
#   2. From physical parameters (runs the solver inline):
#      julia --project=. scripts/plot_surferbot_run.jl --xM 0.13 --EI 5e-4 [OPTIONS]
#
# Options (all modes):
#   --outdir   DIR        output directory  (default: output/figures)
#   --basename NAME       stem for .mp4 and .json  (default: auto from params or "run")
#   --fps      N          frame rate  (default: 30)
#   --duration N          simulated periods  (default: 10)
#   --nframes  N          exact frame count, overrides fps×duration
#
# Options (param mode only):
#   --xM       FLOAT      motor position as fraction of raft length xM/L
#   --EI       FLOAT      bending stiffness EI [SI units]
#   --nu       FLOAT      kinematic viscosity [m²/s]  (default: 1e-6, water)

function parse_args(args)
    opts = Dict{String,Any}(
        "outdir"    => nothing,
        "basename"  => nothing,
        "fps"       => 30,
        "duration"  => 10.0,
        "nframes"   => nothing,
        "xM"        => nothing,
        "EI"        => nothing,
        "nu"        => 1e-6,
        "input"     => nothing,
    )

    # Legacy positional form: none of the args start with "--"
    # Signature: <run-file> [outdir] [basename] [fps] [duration] [nframes]
    if !any(startswith(a, "--") for a in args)
        opts["input"]    = args[1]
        length(args) >= 2 && (opts["outdir"]   = args[2])
        length(args) >= 3 && (opts["basename"] = args[3])
        length(args) >= 4 && (opts["fps"]      = parse(Int,     args[4]))
        length(args) >= 5 && (opts["duration"] = parse(Float64, args[5]))
        length(args) >= 6 && (opts["nframes"]  = parse(Int,     args[6]))
        return opts
    end

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir";       opts["outdir"]   = args[i+1]; i += 2
        elseif a == "--basename"; opts["basename"] = args[i+1]; i += 2
        elseif a == "--fps";      opts["fps"]      = parse(Int,     args[i+1]); i += 2
        elseif a == "--duration"; opts["duration"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--nframes";  opts["nframes"]  = parse(Int,     args[i+1]); i += 2
        elseif a == "--xM";       opts["xM"]       = parse(Float64, args[i+1]); i += 2
        elseif a == "--EI";       opts["EI"]       = parse(Float64, args[i+1]); i += 2
        elseif a == "--nu";       opts["nu"]       = parse(Float64, args[i+1]); i += 2
        elseif !startswith(a, "--") && opts["input"] === nothing
            opts["input"] = a; i += 1
        else
            error("Unknown argument: $a\nRun with no arguments to see usage.")
        end
    end
    return opts
end

const USAGE = """
Usage:
  From a saved run file:
    julia --project=. scripts/plot_surferbot_run.jl <run-file> [OPTIONS]

  From physical parameters:
    julia --project=. scripts/plot_surferbot_run.jl --xM <xM/L> --EI <EI> [OPTIONS]

Options (all modes):
  --outdir   DIR    output directory          (default: output/figures)
  --basename NAME   stem for .mp4 and .json  (default: auto or "run")
  --fps      N      frame rate                (default: 30)
  --duration N      simulated periods         (default: 10)
  --nframes  N      exact frame count

Options (param mode):
  --xM       FLOAT  motor position xM/L
  --EI       FLOAT  bending stiffness [SI]
  --nu       FLOAT  kinematic viscosity m²/s  (default: 1e-6, water)
"""

function solve_from_params(opts)
    defaults  = Surferbot.FlexibleParams{Float64}()
    xM_norm   = opts["xM"]
    EI        = opts["EI"]
    nu        = opts["nu"]
    motor_pos = xM_norm * Float64(defaults.L_raft)
    p         = Surferbot.FlexibleParams{Float64}(; (k => getproperty(defaults, k)
                    for k in fieldnames(Surferbot.FlexibleParams))...,
                    motor_position = motor_pos, EI = EI, nu = nu)
    @info "Solving: xM/L=$(xM_norm)  EI=$(EI)  nu=$(nu)"
    return Surferbot.flexible_solver(p)
end

function auto_basename(opts)
    if opts["xM"] !== nothing && opts["EI"] !== nothing
        return @sprintf("surferbot_xM%.4f_EI%.2e_nu%.0e", opts["xM"], opts["EI"], opts["nu"])
    end
    return "plot_surferbot_run"
end

function main(args=ARGS)
    if isempty(args) || args[1] in ("-h", "--help")
        print(USAGE); return
    end

    opts    = parse_args(args)
    outdir  = something(opts["outdir"], joinpath(@__DIR__, "..", "output", "figures"))
    fps     = opts["fps"]
    dur     = opts["duration"]
    nframes = opts["nframes"]

    param_mode = opts["xM"] !== nothing || opts["EI"] !== nothing

    if param_mode
        (opts["xM"] === nothing || opts["EI"] === nothing) &&
            error("--xM and --EI must both be provided in param mode.")
        bname = something(opts["basename"], auto_basename(opts))
        res   = solve_from_params(opts)
        render_surferbot_run(res; outdir, basename=bname, fps, duration_periods=dur,
                             nframes, script_name=Base.basename(@__FILE__))
    else
        opts["input"] === nothing && error("Provide a run file or use --xM/--EI.\n$USAGE")
        bname = something(opts["basename"], "plot_surferbot_run")
        render_surferbot_run(opts["input"]; outdir, basename=bname, fps,
                             duration_periods=dur, nframes,
                             script_name=Base.basename(@__FILE__))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
