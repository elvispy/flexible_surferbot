"""
plot_large_response_video.jl

Supplementary animation of the very large wave response seen in the thrust map
near the flexible resonance lines with the motor almost at the raft end.

Operating point located by scanning the total radiated intensity
I = |eta(-l)|^2 + |eta(+l)|^2 over (kappa, xM), then refined against the full
finite-difference model:

    xM/L  = -0.49
    kappa = 1.0213e-4          (0.1% from the odd coupled resonance 1.0202e-4)
    I     = 2233 x the SurferBot reference point (kappa=6.87e-3, xM/L=-0.12)

Two things worth knowing about this point.  The response is dominated by the
odd elastic mode n=5, driven at resonance with only radiation damping to limit
it, and the motor sits where the free-free modes have their largest amplitude,
so the forcing projects onto them maximally.  And the predicted steepness is
ak = 7.66 against 0.24 at the reference point, so the linear small-slope model
is being extrapolated well past its own validity here; the animation shows what
the linear theory predicts, not what a real raft would do.

Same conventions as the other supplementary videos: non-dimensional, 60 fps,
20 s, ten forcing periods.
"""

using Surferbot
using Printf

const KAPPA = 1.0212620339077332e-4
const XM_OVER_L = -0.49

function main()
    output_dir = joinpath(@__DIR__, "..", "output", "figures")
    mkpath(output_dir)

    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    EI_scale = Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2
    L = Float64(bp.L_raft)

    p = Surferbot.Sweep.apply_parameter_overrides(bp,
            (EI = KAPPA * EI_scale, motor_position = XM_OVER_L * L))
    result = Surferbot.flexible_solver(p)

    eta = ComplexF64.(result.eta)
    I = abs2(eta[1]) + abs2(eta[end])
    alpha = (abs2(eta[1]) - abs2(eta[end])) / I
    @printf "kappa=%.4e  xM/L=%.2f  I=%.4e  alpha=%+.4f  ak=%.3f  max|eta|=%.3e m\n" KAPPA XM_OVER_L I alpha result.wave_steepness maximum(abs.(eta))

    paths = Surferbot.render_surferbot_run(result;
        outdir           = output_dir,
        basename         = "large_response_end_forcing",
        fps              = 60,
        duration_periods = 10,
        seconds          = 20,
        nondim           = true,
        script_name      = Base.basename(@__FILE__))
    println("Saved: $(paths.mp4)")
end

main()
