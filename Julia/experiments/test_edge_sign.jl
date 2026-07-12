"""
test_edge_sign.jl

Physical Newton's law test for the raft-edge boundary condition.

The question (as stated by the user):
  "Find a surface with, say, positive slope. Surface tension as a force should be
   pulling upwards. Does the shear force M_x actually pull upwards or downwards?"

At the RIGHT edge (+L/2), when eta_x(+L/2⁺) > 0:
  • surface slopes upward away from the raft
  • surface tension pulls the right edge UPWARD  → F = +d·σ·eta_x > 0
  → M_x(+½) and eta_x must be CO-DIRECTED (cos_theta = +1)

At the LEFT edge (-L/2), when eta_x(-L/2⁻) > 0:
  • surface slopes upward TOWARD the raft (from the left, high side is the raft side)
  • surface tension pulls the left edge DOWNWARD  → F = -d·σ·eta_x < 0
  → M_x(-½) and eta_x must be ANTI-DIRECTED (cos_theta = -1)

Expected:
  Left edge:  cos_theta ≈ -1  (correct for code AND paper)
  Right edge: cos_theta ≈ +1  (correct for paper; code has WRONG sign → cos_theta ≈ -1)

The test uses:
  - eta_x from the kinematic BC: eta = phi_z / (i·ω) (independent of the edge condition)
  - M_x from the bending moment block of the full solution (independent of eta)
  - cos_theta = Re(Mx · conj(eta_x)) / |Mx| / |eta_x|  (no reference to the paper equation)

Usage:
    julia --project=Julia/Project.toml Julia/experiments/test_edge_sign.jl
"""

using Surferbot
using LinearAlgebra

function run_test(; sigma_scale = 1.0)
    sigma  = 0.0722 * sigma_scale
    params = FlexibleParams(
        sigma          = sigma,
        rho            = 1000.0,
        omega          = 2π * 80.0,
        nu             = 1e-6,
        g              = 9.81,
        L_raft         = 0.05,
        motor_position = -0.003,
        d              = 0.03,
        EI             = 1.0e-4,
        rho_raft       = 0.052,
        motor_inertia  = 0.13e-3 * 2.5e-3,
        L_domain       = 0.14,
    )

    sys = assemble_flexible_system(params)
    sol = solve_tensor_system(sys.A, sys.b)

    derived  = sys.derived
    NP       = derived.Nx * derived.Nz
    nb_c     = derived.nb_contact

    phi_z_vec = sol[(NP + 1):(2 * NP)]
    M_vec     = sol[(2 * NP + 1):(2 * NP + nb_c)]

    # eta from kinematic BC: K̄·eta = phi_z, K̄≈i → eta ≈ phi_z/i (same as postprocess.jl line 36)
    t_c     = derived.t_c
    omega   = derived.params.omega
    eta_vec = phi_z_vec ./ (im * omega * t_c)   # = -i * phi_z in ND

    ooa    = derived.params.ooa
    nb_L   = length(sys.indices.idxLeftFreeSurf)
    DxRaft = getNonCompactFDmatrix(nb_c, Float64(1.0), 1, ooa)
    DxFree = getNonCompactFDmatrix(nb_L,  Float64(1.0), 1, ooa)

    # eta_x at LEFT edge: backward stencil at right end of left-free array
    # (= slope of eta just to the LEFT of the raft, from the free-surface side)
    eta_L   = eta_vec[sys.indices.idxLeftFreeSurf]
    eta_x_L = DxFree[end, :] ⋅ eta_L    # unit-spacing → actual eta_x × dx

    # eta_x at RIGHT edge: forward stencil at left end of right-free array
    # (= slope of eta just to the RIGHT of the raft, from the free-surface side)
    eta_R   = eta_vec[sys.indices.idxRightFreeSurf]
    eta_x_R = DxFree[1, :]   ⋅ eta_R    # unit-spacing → actual eta_x × dx

    # M_x at LEFT/RIGHT edge from the beam moment block
    Mx_L = DxRaft[1,   :] ⋅ M_vec      # forward at left  end of raft
    Mx_R = DxRaft[end, :] ⋅ M_vec      # backward at right end of raft

    # Normalised cosine between M_x and eta_x:
    #   +1 → force in SAME direction as slope  (aligned)
    #   -1 → force in OPPOSITE direction        (anti-aligned)
    cos_L = real(Mx_L * conj(eta_x_L)) / (abs(Mx_L) * abs(eta_x_L) + eps())
    cos_R = real(Mx_R * conj(eta_x_R)) / (abs(Mx_R) * abs(eta_x_R) + eps())

    We  = derived.nd_groups.We
    Lam = derived.nd_groups.Lambda

    println("σ_scale = $sigma_scale   (Λ/We = $(round(Lam/We, sigdigits=3)))")
    println("  Left  edge  cos_theta = $(round(cos_L, sigdigits=4))   expected ≈ -1  (force OPPOSES slope)  $(abs(cos_L + 1) < 0.1 ? "✓" : "✗ WRONG")")
    println("  Right edge  cos_theta = $(round(cos_R, sigdigits=4))   expected ≈ +1  (force FOLLOWS slope)  $(abs(cos_R - 1) < 0.1 ? "✓" : "✗ WRONG — surface tension pulls the WRONG way")")
    println()
end

println("=== Physical force-direction test ===")
println("Right edge expected: cos_theta ≈ +1 (force UP when slope UP).")
println("Both edges anti-aligned → sign error at right edge.")
println()
for scale in [1.0, 10.0, 100.0, 1000.0, 10_000.0]
    run_test(sigma_scale = scale)
end
