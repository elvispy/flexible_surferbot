"""
test_kappa_fix_precision.jl

Machine-precision regression for Fix 1 (raft balance K̄ viscous) and Fix 2 (edge BC).

At ν=0 (Re→∞) and σ=0 (We→∞):
  - Fix 1: (2/Re)·Diag(inertia_coeff)·Dx2Raft = 0  → S12[CC,CC] unchanged
  - Fix 2: edge_visc = 2/(Re·dx²) = 0, Λ/We = 0   → S13 edge rows differ only by factor i

Both fixes reduce to a scalar rescaling of the constraint rows → solutions must agree to
machine precision (‖η_new − η_old‖ / ‖η_old‖ ≤ 1e-12).

Usage:
    julia --project=Julia/Project.toml Julia/experiments/test_kappa_fix_precision.jl
"""

using Surferbot
using LinearAlgebra
using SparseArrays

function run()
    params = FlexibleParams(
        sigma          = 0.0,      # We→∞: edge condition = 0
        nu             = 0.0,      # Re→∞: viscous terms = 0
        rho            = 1000.0,
        omega          = 2π * 80.0,
        g              = 9.81,
        L_raft         = 0.05,
        motor_position = -0.003,
        d              = 0.03,
        EI             = 1.0e-4,
        rho_raft       = 0.052,
        motor_inertia  = 0.13e-3 * 2.5e-3,
        L_domain       = 0.14,
    )

    sys    = assemble_flexible_system(params)
    A_new  = copy(sys.A)
    b      = sys.b
    idxC   = sys.indices.idxContact

    derived = sys.derived
    NP      = derived.Nx * derived.Nz
    nb_c    = derived.nb_contact

    # Reconstruct old edge rows: divide by i to undo i·DxRaft → DxRaft
    # Old: S13[CC[1],:] = DxRaft[1,:]  → = A_new[CC[1], 2NP+1:2NP+nb_c] / i
    A_old = copy(A_new)
    col_range = (2 * NP + 1):(2 * NP + nb_c)
    A_old[idxC[1],   col_range] ./= im
    A_old[idxC[end], col_range] ./= im

    x_new = A_new \ b
    x_old = A_old \ b

    # Extract η via kinematic BC: eta = phi_z / (i·ω·t_c)
    phi_z_new = x_new[(NP + 1):(2 * NP)]
    phi_z_old = x_old[(NP + 1):(2 * NP)]
    t_c = derived.t_c
    omega = derived.params.omega
    eta_new = phi_z_new ./ (im * omega * t_c)
    eta_old = phi_z_old ./ (im * omega * t_c)

    rel_err = norm(eta_new .- eta_old) / norm(eta_old)
    println("Machine-precision test at ν=0, σ=0")
    println("  ‖η_new − η_old‖ / ‖η_old‖ = $(rel_err)")

    threshold = 1e-12
    passed = rel_err ≤ threshold
    println("  $(passed ? "PASS ✓" : "FAIL ✗")  (threshold = $threshold)")
    return passed
end

run()
