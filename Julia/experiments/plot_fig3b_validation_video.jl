"""
Supplementary-material animation for the rigid-limit validation case shown
in the paper's figure 3(b) (plot_fig4_Aguero2026.jl): EI=Inf, motor at
xM/L=-0.12, forcing_width=0.05*L_raft (main.tex, sec. "Reference rigid-raft
response"). Renders fig3b_rigid_validation.mp4.
"""

using Surferbot

function main()
    output_dir = joinpath(@__DIR__, "..", "output", "figures")
    mkpath(output_dir)

    params = Surferbot.FlexibleParams(
        sigma          = 0.0722,
        rho            = 1000.0,
        omega          = 2π * 80.0,
        nu             = 1e-6,
        g              = 9.81,
        L_raft         = 0.05,
        motor_position = -0.006,
        d              = 0.03,
        EI             = Inf,
        rho_raft       = 0.052,
        motor_inertia  = 0.13e-3 * 2.5e-3 * 2.0,
        L_domain       = 0.14,
        forcing_width  = 0.05 * 0.05,
    )

    result = Surferbot.flexible_solver(params)

    paths = Surferbot.render_surferbot_run(result;
        outdir           = output_dir,
        basename         = "fig3b_rigid_validation",
        fps              = 30,
        duration_periods = 10,
        script_name      = Base.basename(@__FILE__))

    println("Saved: $(paths.mp4)")
end

main()
