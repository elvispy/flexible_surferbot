using Test
using Surferbot

@testset "capillary_endpoint_map" begin
    # Synthetic free-free-like basis: mode m has parity (-1)^m, so
    # psi_start[m] = (-1)^m * psi_end[m] and (with the validated Fix 2 sign
    # convention applied at the source -- see edge_slopes) s_vec_left[n] =
    # (-1)^n * s_vec[n]. Both individually mirror the mode's own parity, no
    # extra derivative sign flip, matching the real free-free basis and the
    # validated a_vec/a_vec_left pattern.
    modes = 0:7
    psi_end   = [1.3, 2.1, 0.7, 1.9, 2.4, 0.5, 1.1, 2.8]
    psi_start = [(-1.0)^n * psi_end[i] for (i, n) in enumerate(modes)]
    s_vec      = ComplexF64[1.0+0.5im, 0.3-0.2im, 2.1+1.1im, 0.8-0.4im,
                             1.5+0.9im, 0.6-0.1im, 1.2+0.3im, 0.4-0.6im]
    s_vec_left = ComplexF64[(-1.0)^n * s_vec[i] for (i, n) in enumerate(modes)]

    C = Surferbot.capillary_endpoint_map(psi_end, psi_start, s_vec, s_vec_left)
    @test size(C) == (8, 8)

    # The regression check: with this sign convention, C must be exactly
    # parity-block-diagonal -- zero for every opposite-parity (m,n) pair, and
    # generically nonzero for same-parity pairs. Get the sign wrong anywhere in
    # this chain (here, or in edge_slopes upstream) and this inverts (nonzero
    # cross-parity, ~zero same-parity) -- exactly the bug that silently broke
    # the A(xM=0)=0 symmetry.
    for (i, m) in enumerate(modes), (j, n) in enumerate(modes)
        if isodd(m + n)
            @test abs(C[i, j]) < 1e-10
        else
            @test abs(C[i, j]) > 1e-6
        end
    end

    # Physical consequence, reproduced directly: if forcing only excites even
    # modes (q_odd = 0, as happens for exactly-centered symmetric forcing),
    # the antisymmetric combination q' * s_vec_left-weighted quantity built
    # from C must not leak into the odd equations from even q, and vice versa
    # -- i.e. C restricted to (odd row, even col) or (even row, odd col) acting
    # on a purely-even q must contribute nothing to odd equations.
    q_even_only = ComplexF64[isodd(n) ? 0.0 : 1.0 + 0.2im for n in modes]
    Cq = C * q_even_only
    for (i, m) in enumerate(modes)
        if isodd(m)
            @test abs(Cq[i]) < 1e-10  # odd equation must see zero contribution
        end
    end
end

@testset "C_sigma end-to-end symmetry at xM=0 (real physics, not synthetic)" begin
    # The unit test above checks the extracted capillary_endpoint_map formula
    # in isolation. This checks the actual production pipeline that consumes
    # it (theoretical_modal_context_LH / solve_theoretical_modal_response,
    # C_sigma included) against the real modal-pressure-map cache and sweep
    # parameters -- so a future regression in *how* C_sigma is assembled or
    # consumed (not just in the formula itself) is still caught.
    #
    # The invariant: forcing exactly at the raft center (xM=0) is left-right
    # symmetric, so only even (symmetric) modes are excited and the
    # antisymmetric radiation amplitude A must vanish, for ANY stiffness kappa,
    # up to the discretization-limited parity residual of the coupled FD
    # pipeline (|A|/|S| ~ 1e-7 here, set by grid symmetry and small cross-parity
    # leakage in the cached impedance Z_psi). This is what the C_sigma sign bug
    # broke (A(xM=0) came out as ~1e-5 relative to |S|~1e-3, i.e. ~1e-2
    # relative) -- see the commit fixing it for the full derivation.
    repo = joinpath(@__DIR__, "..")
    include(joinpath(repo, "scripts", "plot_dimensionless_diagnostics_LH.jl"))

    output_dir = joinpath(repo, "output")
    art = load_sweep(joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))
    params = art.base_params
    shift  = log10(Float64(params.rho_raft) * Float64(params.L_raft)^4 * Float64(params.omega)^2)
    theory_ctx = theoretical_modal_context_LH(params; output_dir=output_dir)

    for log10_kappa in (-3.5, -2.0, -1.0, 0.0, 1.0)  # spans the figure's full kappa range
        EI = 10^(log10_kappa + shift)
        q    = solve_theoretical_modal_response(EI, 0.0, theory_ctx)
        diag = theoretical_endpoint_diagnostics_LH(q, theory_ctx)
        # scale-invariant: compare against |S|, which is generically O(1) at
        # xM=0 (only even modes excited), rather than an absolute tolerance.
        # Threshold 1e-6 sits above the ~1e-7 discretization residual and still
        # catches the ~1e-2 relative parity-sign bug by four orders of magnitude.
        @test abs(diag.A) / max(abs(diag.S), eps()) < 1e-6
    end
end
