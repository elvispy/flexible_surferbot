using Test
using Surferbot

include(joinpath(@__DIR__, "..", "experiments", "prescribed_wn_diagonal_impedance.jl"))
include(joinpath(@__DIR__, "..", "experiments", "plot_dimensionless_diagnostics.jl"))

const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

@testset "modal pressure map cache" begin
    params = FlexibleParams(
        n = 31,
        Nz = 10,
        domain_depth = 0.05,
        L_domain = 0.15,
        d = 0.03,
    )

    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        params;
        output_dir = mktempdir(),
        num_modes_basis = 8,
    )

    @test payload.cache_status.loaded == false
    @test payload.mode_labels == collect(0:7)
    @test size(payload.Z_raw) == (8, 8)
    @test size(payload.Z_psi) == (8, 8)

    q_psi = ComplexF64[1.0 + 0im, -0.3 + 0.2im, 0.15 - 0.05im, 0.0 + 0.1im, -0.08 + 0im, 0.03 - 0.02im, 0.01 + 0.01im, -0.02 + 0.005im]
    q_raw = payload.transforms.raw_from_psi * q_psi
    p_psi_from_raw = payload.transforms.psi_from_raw * (payload.Z_raw * q_raw)
    @test payload.Z_psi * q_psi ≈ p_psi_from_raw atol = 1e-8 rtol = 1e-8

    cached = ModalPressureMap.load_or_compute_modal_pressure_map(
        params;
        output_dir = dirname(dirname(payload.cache_status.path)),
        num_modes_basis = 8,
    )
    @test cached.cache_status.loaded == true
    @test cached.Z_raw ≈ payload.Z_raw atol = 1e-12 rtol = 1e-12

    uncoupled = FlexibleParams(
        n = 31,
        Nz = 10,
        domain_depth = 0.05,
        L_domain = 0.15,
        d = 0.0,
    )
    uncoupled_payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        uncoupled;
        output_dir = mktempdir(),
        num_modes_basis = 8,
    )
    @test maximum(abs.(uncoupled_payload.Z_raw)) == 0.0
    @test maximum(abs.(uncoupled_payload.Z_psi)) == 0.0

    theory_ctx = Main.theoretical_modal_context(uncoupled; output_dir=mktempdir())
    q = Main.solve_theoretical_modal_response(uncoupled.EI, 0.2, theory_ctx)
    xM = 0.2 * uncoupled.L_raft
    load_dist = theory_ctx.F0 .* Main.gaussian_load_nd(xM, theory_ctx.sigma_f, theory_ctx.x_raft, uncoupled.L_raft)
    forcing_rhs = theory_ctx.psi_gram \ (theory_ctx.Psi' * (load_dist .* theory_ctx.weights))
    structural = Diagonal(ComplexF64.(uncoupled.EI .* theory_ctx.beta .^ 4 .- uncoupled.rho_raft .* uncoupled.omega^2))
    @test q ≈ (structural \ (-ComplexF64.(forcing_rhs))) atol = 1e-10 rtol = 1e-10
end
