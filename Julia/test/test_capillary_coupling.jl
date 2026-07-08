using Test
using Surferbot

@testset "capillary_endpoint_map" begin
    # Synthetic free-free-like basis: mode m has parity (-1)^m, so
    # psi_start[m] = (-1)^m * psi_end[m] (a position quantity mirrors with its
    # own mode parity), while s_vec_left[n] = (-1)^(n+1) * s_vec[n] (a
    # slope/derivative quantity picks up an extra sign flip under reflection).
    # This mirrors the exact relationships verified against the real free-free
    # basis and the validated a_vec/a_vec_left pattern.
    modes = 0:7
    psi_end   = [1.3, 2.1, 0.7, 1.9, 2.4, 0.5, 1.1, 2.8]
    psi_start = [(-1.0)^n * psi_end[i] for (i, n) in enumerate(modes)]
    s_vec      = ComplexF64[1.0+0.5im, 0.3-0.2im, 2.1+1.1im, 0.8-0.4im,
                             1.5+0.9im, 0.6-0.1im, 1.2+0.3im, 0.4-0.6im]
    s_vec_left = ComplexF64[(-1.0)^(n+1) * s_vec[i] for (i, n) in enumerate(modes)]

    C = Surferbot.capillary_endpoint_map(psi_end, psi_start, s_vec, s_vec_left)
    @test size(C) == (8, 8)

    # The regression check: with this sign convention, C must be exactly
    # parity-block-diagonal -- zero for every opposite-parity (m,n) pair, and
    # generically nonzero for same-parity pairs. A `+` instead of `-` in the
    # implementation would invert this (nonzero cross-parity, ~zero same-parity)
    # -- exactly the bug that silently broke the A(xM=0)=0 symmetry.
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
