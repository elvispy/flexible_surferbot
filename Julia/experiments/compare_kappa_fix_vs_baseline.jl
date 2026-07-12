"""
compare_kappa_fix_vs_baseline.jl

Compares the 5 paper-snapshot operating points between the new (Fix 1 + Fix 2)
and a reconstructed baseline (old K̄ approximations) in a single Julia session.
Precompiles once; all 10 solves share the same loaded module.

Usage:
    julia --project=Julia/Project.toml Julia/experiments/compare_kappa_fix_vs_baseline.jl
"""

using Surferbot
using LinearAlgebra
using SparseArrays
using Printf

# ─── reconstruct old system matrix ──────────────────────────────────────────

function make_old_system(sys)
    d   = sys.derived
    NP  = d.Nx * d.Nz
    nb_c = d.nb_contact
    Re    = Float64(d.nd_groups.Re)
    Lambda = Float64(d.nd_groups.Lambda)
    We    = Float64(d.nd_groups.We)
    Fr    = Float64(d.nd_groups.Fr)
    Gamma = Float64(d.nd_groups.Gamma)
    dx    = Float64(d.dx)
    ooa   = d.params.ooa

    CC = sys.indices.idxContact
    L  = sys.indices.idxLeftFreeSurf
    R  = sys.indices.idxRightFreeSurf

    DxRaft  = getNonCompactFDmatrix(nb_c,     1.0, 1, ooa)
    Dx2Raft = getNonCompactFDmatrix(nb_c,     1.0, 2, ooa)
    DxFreeL = getNonCompactFDmatrix(length(L), 1.0, 1, ooa)   # same matrix for L and R
    DxFreeR = DxFreeL

    inertia_coeff = Float64.(d.inertia_vec) .- Gamma * Lambda / Fr^2

    A_old = copy(sys.A)
    col_M = (2*NP + 1):(2*NP + nb_c)

    # Undo Fix 1: remove (2/Re)*Diag(inertia_coeff)*Dx2Raft from S12[CC, CC]
    fix1 = (2.0 / Re) .* (Diagonal(Complex{Float64}.(inertia_coeff)) * Dx2Raft)
    for (i, ci) in enumerate(CC)
        for (j, cj) in enumerate(CC)
            A_old[ci, NP + cj] -= fix1[i, j]
        end
    end

    # Undo Fix 2: restore old edge rows (DxRaft, no i factor; old S12 sign/dx)
    A_old[CC[1],   col_M]    = DxRaft[1, :]
    A_old[CC[end], col_M]    = DxRaft[end, :]
    A_old[CC[1],   NP .+ L]  = (-im * dx * Lambda / We) .* DxFreeL[end, :]
    A_old[CC[end], NP .+ R]  = (-im * dx * Lambda / We) .* DxFreeR[1,  :]

    return A_old
end

# ─── extract eta from solution vector ────────────────────────────────────────

function extract_eta(x, derived)
    NP = derived.Nx * derived.Nz
    phi_z = reshape(x[(NP+1):(2*NP)], derived.Nz, derived.Nx)
    omega = Float64(derived.params.omega)
    t_c   = Float64(derived.t_c)
    return (1.0 / (im * omega * t_c)) .* vec(phi_z[end, :])
end

# ─── Sxx thrust estimate (inviscid far-field) ─────────────────────────────────

function sxx_thrust(eta, d, params)
    k     = Float64(real(d.k))
    rho   = Float64(params.rho)
    g     = Float64(params.g)
    sigma = Float64(params.sigma)
    depth = Float64(d.d)
    pref  = rho * g / 4 + 3/4 * sigma * k^2
    return pref * (abs2(eta[1]) - abs2(eta[end])) * depth
end

# ─── main ────────────────────────────────────────────────────────────────────

function run()
    bp        = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    EI_scale  = Float64(bp.rho_raft) * Float64(bp.L_raft)^4 * Float64(bp.omega)^2
    xM_sb     = abs(Float64(bp.motor_position)) / Float64(bp.L_raft)

    ops = [
        (kappa=1.71103172e-3, xM=xM_sb,  label="(a) κ=1.71e-3  xM=SB  "),
        (kappa=5.43e-3,       xM=xM_sb,  label="(b) κ=5.43e-3  xM=SB  "),
        (kappa=5.43e-3,       xM=0.183,  label="(c) κ=5.43e-3  xM=0.183"),
        (kappa=5.43e-3,       xM=0.272,  label="(d) κ=5.43e-3  xM=0.272"),
        (kappa=2.22e-2,       xM=xM_sb,  label="(e) κ=2.22e-2  xM=SB  "),
    ]

    L_domain = 0.10   # same as plot_kappa_snapshot.jl build_params

    println("\n", "─"^118)
    @printf("%-26s  %7s  %7s  %8s  %9s  %9s  %9s  %7s\n",
            "Case", "η_n(µm)", "η_o(µm)", "Δamp%", "‖Δη‖/‖η‖", "T_new(µN)", "T_old(µN)", "ΔT%")
    println("─"^118)

    for op in ops
        EI = op.kappa * EI_scale
        params_kw = Surferbot.Sweep.apply_parameter_overrides(bp, (L_domain = L_domain, EI = EI,
                     motor_position = op.xM * Float64(bp.L_raft)))

        # New (current branch)
        result, sys = flexible_solver(params_kw; return_system=true)
        eta_new = result.eta
        T_new   = result.thrust

        # Old (baseline reconstruction)
        A_old = make_old_system(sys)
        x_old = A_old \ sys.b
        eta_old_full = extract_eta(x_old, sys.derived)

        # Only the domain η values (same length as result.eta)
        eta_old = eta_old_full .* sys.derived.L_c   # nondim → metres

        # Sxx-based thrust estimate using dimensional eta_old
        T_old = sxx_thrust(eta_old, sys.derived, params_kw)

        # Metrics — split into amplitude change and total L2 change
        eta_rel   = norm(eta_new .- eta_old) / (norm(eta_old) + eps())
        amp_rel   = abs(maximum(abs.(eta_new)) - maximum(abs.(eta_old))) /
                    (maximum(abs.(eta_old)) + eps())
        max_new   = maximum(abs.(eta_new)) * 1e6   # µm
        max_old   = maximum(abs.(eta_old)) * 1e6   # µm
        T_rel     = abs(T_new - T_old) / (abs(T_old) + eps())

        @printf("%-26s  %7.1f  %7.1f  %7.4f%%  %7.2e  %9.4f  %9.4f  %6.2f%%\n",
                op.label, max_new, max_old, amp_rel*100, eta_rel, T_new*1e6, T_old*1e6, T_rel*100)
    end
    println("─"^118)
    println("\nΔamp%  = |max|η_new| − max|η_old|| / max|η_old|  (amplitude shift, visible in plot)")
    println("‖Δη‖/‖η‖ = L2 norm change over full domain (includes phase shifts)")
    println("T_old  = Sxx estimate from reconstructed old solution (inviscid far-field)")
end

run()
