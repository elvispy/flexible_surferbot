"""
plot_fig6_flexibility_grid_videos.jl

Supplementary-material animations for the three columns of the paper's
figure 6 (fig:flexibility_grid, plot_kappa_snapshot_grid_flexibility.pdf):
the flexibility sweep at the fixed SurferBot forcing position xM/L=-0.12,
at the three highlighted stiffnesses (KAPPA_HIGHLIGHTS, paper_theme.jl):
  (a) kappa = 1.9953e-3  (mode 3 dominant, large peak thrust)
  (b) kappa = 6.8665e-3  (intermediate, mixed modes, alpha ~ -1)
  (e) kappa = 1.7575e-2  (mode 2 dominant, large peak thrust)
Same convention as the existing fig3b_rigid_validation.mp4 and
non_uniform_surferbot.mp4: Surferbot.flexible_solver + render_surferbot_run,
fps=30, duration_periods=10.
"""

using Surferbot
using Printf

const KAPPA_HIGHLIGHTS = [1.9952623149688789e-3, 6.8665e-3, 1.7575106248547922e-2]
const LABELS = ["a", "b", "e"]

function main()
    output_dir = joinpath(@__DIR__, "..", "output", "figures")
    mkpath(output_dir)

    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    EI_scale = Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2

    for (label, kappa) in zip(LABELS, KAPPA_HIGHLIGHTS)
        EI = kappa * EI_scale
        p = Surferbot.Sweep.apply_parameter_overrides(bp, (; EI = EI))
        result = Surferbot.flexible_solver(p)

        m = Surferbot.Analysis.beam_edge_metrics(result)
        alpha = Surferbot.Analysis.beam_asymmetry(m.eta_left_beam, m.eta_right_beam)
        @printf "fig6%s  kappa=%.4e  U=%.4f mm/s  alpha=%+.4f\n" label kappa result.U*1e3 alpha

        basename = @sprintf("fig6%s_kappa_snapshot_%.4e", label, kappa)
        paths = Surferbot.render_surferbot_run(result;
            outdir           = output_dir,
            basename         = basename,
            fps              = 30,
            duration_periods = 10,
            script_name      = Base.basename(@__FILE__))
        println("Saved: $(paths.mp4)")
    end
end

main()
