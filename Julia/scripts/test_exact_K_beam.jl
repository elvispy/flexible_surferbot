"""
test_exact_K_beam.jl

Compare eta wavefield between:
  - approx:  K̄⁻¹ ≈ -i              (current)
  - exact:   K̄(M/κ) = φ_zxx        (new)

Reports relative L2 norm ‖η_exact − η_approx‖ / ‖η_approx‖ for 10
randomly sampled (nu, motor_position) pairs. Expected order: O(1/Re).
"""

using Surferbot, LinearAlgebra, Random, Printf

Random.seed!(42)

const BASE = FlexibleParams{Float64}()
const L    = BASE.L_raft

# Sample nu log-uniformly in [1e-7, 1e-3] (Re ≈ 1e3 … 1e7)
# Sample motor_position uniformly in [0.1L, 0.5L]
function sample_params(rng)
    nu  = 10^(-7 + 4 * rand(rng))          # log-uniform 1e-7..1e-3
    xM  = L * (0.1 + 0.4 * rand(rng))      # 0.1L..0.5L
    Re  = L^2 * BASE.omega / nu
    (nu = nu, xM = xM, Re = Re)
end

@printf("%-4s  %-11s  %-8s  %-12s  %-12s  %-12s\n",
        "#", "nu", "xM/L", "Re", "rel_L2_eta", "rel_L2_thrust")
@printf("%s\n", "─"^72)

rng = MersenneTwister(42)
for i in 1:10
    s = sample_params(rng)

    p_a = FlexibleParams{Float64}(nu=s.nu, motor_position=s.xM, exact_K_beam=false)
    p_e = FlexibleParams{Float64}(nu=s.nu, motor_position=s.xM, exact_K_beam=true)

    r_a = flexible_solver(p_a)
    r_e = flexible_solver(p_e)

    rel_eta    = norm(r_e.eta    .- r_a.eta)    / norm(r_a.eta)
    rel_thrust = abs(r_e.thrust  -  r_a.thrust) / abs(r_a.thrust)

    @printf("%-4d  %-11.3e  %-8.4f  %-12.4e  %-12.4e  %-12.4e\n",
            i, s.nu, s.xM/L, s.Re, rel_eta, rel_thrust)
end
