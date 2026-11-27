using Test
using LinearAlgebra
using Surferbot

# The modal capillary map C^σ is checked against the slopes of the solved free surface, not
# against a sign convention. For a raft whose shape is prescribed as mode W_n, Newton's law
# on the contact lines gives the vertical force on the raft
#     F_R = +σ d η_x(R+)   (right edge),     F_L = -σ d η_x(L-)   (left edge),
# with the slopes taken just outside the raft. The generalised force on mode m is
# W_m(1/2) F_R + W_m(-1/2) F_L, and the modal raft equation (D - Z + C^σ) q = -F moves it to
# the left with the opposite sign, so
#     C^σ_mn / (σ d) = -W_m(1/2) η_x^n(R+) + W_m(-1/2) η_x^n(L-).
# The slopes below are finite differences of the free-surface profile returned by the same
# prescribed-mode solves that build the modal model.
#
# A sign error in either edge, or a global sign error, changes this matrix by O(1), so each
# is also checked as a mutation that must fail.

if !isdefined(Main, :PrescribedWnDiagonalImpedance)
    include(joinpath(@__DIR__, "..", "experiments", "prescribed_wn_diagonal_impedance.jl"))
end

@testset "C_sigma equals the capillary force computed from the solved free-surface slopes" begin
    EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12
    p0 = FlexibleParams(motor_position = -0.012, EI = EI0, sigma = 72.2e-3, nu = 0.0)
    k = real(Surferbot.derive_params(p0).k)
    p = FlexibleParams(motor_position = -0.012, EI = EI0, sigma = 72.2e-3, nu = 0.0,
                       dz_surface = 0.1 / k, z_growth = 1.1)
    derived = Surferbot.derive_params(p)
    res = Main.PrescribedWnDiagonalImpedance.empirical_modal_pressure_map(
        p; num_modes_basis = 6, validate = false)

    contact = Bool.(derived.x_contact)
    dx = derived.dx * derived.L_c
    iR = findlast(contact) + 1       # first free-surface node right of the raft
    iL = findfirst(contact) - 1      # first free-surface node left of the raft
    slope_R(eta) = (-3 * eta[iR] + 4 * eta[iR + 1] - eta[iR + 2]) / (2dx)
    slope_L(eta) = (3 * eta[iL] - 4 * eta[iL - 1] + eta[iL - 2]) / (2dx)

    Phi = res.raw_basis.Phi
    sR = [slope_R(c.eta) for c in res.columns]
    sL = [slope_L(c.eta) for c in res.columns]
    N = length(sR)
    expected = [-Phi[end, m] * sR[n] + Phi[1, m] * sL[n] for m in 1:N, n in 1:N]
    @test norm(expected) > 0

    produced = Surferbot.capillary_endpoint_map(Phi[end, :], Phi[1, :],
                                                ComplexF64.(res.s_vec_raw),
                                                ComplexF64.(res.s_vec_left_raw))
    relerr(C) = norm(C - expected) / norm(expected)
    @info "C_sigma vs free-surface slopes: relerr = $(relerr(produced))"

    # The residual is the difference between the 3-point slope used here and the higher-order
    # stencil used in the solver, on a graded grid.
    @test relerr(produced) < 0.2

    # Mutations: each wrong version of the map must be far from the force on the contact lines.
    right_flipped = [+Phi[end, m] * sR[n] + Phi[1, m] * sL[n] for m in 1:N, n in 1:N]
    left_flipped  = [-Phi[end, m] * sR[n] - Phi[1, m] * sL[n] for m in 1:N, n in 1:N]
    @info "mutations: relerr(-C)=$(relerr(-produced)) right_flipped=$(relerr(right_flipped)) left_flipped=$(relerr(left_flipped))"
    @test relerr(-produced) > 0.5
    @test relerr(right_flipped) > 0.5
    @test relerr(left_flipped) > 0.5
end
