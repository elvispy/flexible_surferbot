"""
This scripts is a plot of a single simulation of a surferbot case with a nonuniform EI distribution.

80% of the raft should be stiff (EI = Inf), and 20% should be compliant. The density can be constant,
and the motor should be sitting on the stiff part, but close to the joint between stiff and compliant.

EI_soft ≈ 1e-6 N·m²  — rubber-like (≈ 5000× below the default surferbot EI).
Motor sits 2 mm inside the stiff zone from the joint.
"""

using Surferbot
using Printf

function main()
    output_dir = joinpath(@__DIR__, "..", "output", "figures")
    mkpath(output_dir)

    # ── Base parameters (coupled default) ──────────────────────────────────────
    bp = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    fparams_scalar = Surferbot.Sweep.apply_parameter_overrides(bp, (;))

    # ── Count contact nodes (grid depends only on L_raft, not EI) ──────────────
    derived_scalar = Surferbot.derive_params(fparams_scalar)
    nb = derived_scalar.nb_contact

    # ── Graded EI: first 80% rigid (Inf), last 20% rubber-like ─────────────────
    n_stiff  = round(Int, 0.8 * nb)
    EI_soft  = 1e-6   # N·m²  (natural rubber cross-section; ≈ 5000× below default)
    EI_vec   = vcat(fill(Inf, n_stiff), fill(EI_soft, nb - n_stiff))

    # ── Motor: 2 mm inside the stiff zone from the joint ───────────────────────
    # Joint is at x = L_raft*(0.8 - 0.5) = 0.015 m from beam centre (for L=0.05 m).
    L        = Float64(bp.L_raft)
    x_joint  = L * (0.8 - 0.5)          # +0.015 m from centre
    x_motor  = x_joint - 0.002          # +0.013 m from centre (stiff side)

    # ── Assemble and solve ──────────────────────────────────────────────────────
    p      = Surferbot.Sweep.apply_parameter_overrides(bp, (EI = EI_vec, motor_position = x_motor, nu = 1e-6))
    result = Surferbot.flexible_solver(p)

    # ── Console diagnostics ─────────────────────────────────────────────────────
    m = Surferbot.Analysis.beam_edge_metrics(result)
    α = Surferbot.Analysis.beam_asymmetry(m.eta_left_beam, m.eta_right_beam)
    @printf "U    = %.4f mm/s\n"   result.U * 1e3
    @printf "α    = %+.4f\n"       α
    @printf "|η_L| = %.4e m\n"     abs(m.eta_left_beam)
    @printf "|η_R| = %.4e m\n"     abs(m.eta_right_beam)
    @printf "n_stiff = %d / %d contact nodes (%.0f%%)\n" n_stiff nb 100*n_stiff/nb
    @printf "x_motor / L = %.4f\n" x_motor / L

    # ── Render animation ────────────────────────────────────────────────────────
    paths = Surferbot.render_surferbot_run(result;
        outdir          = output_dir,
        basename        = "non_uniform_surferbot",
        fps             = 60,
        duration_periods = 10,
        seconds         = 20,
        nondim          = true,
        script_name     = Base.basename(@__FILE__))

    println("Saved: $(paths.mp4)")
end

main()
