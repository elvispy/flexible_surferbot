using Surferbot
include(joinpath(@__DIR__, "..", "src", "optimization.jl"))
using .SurferbotOptimization

# Purpose: run the current Julia gradient-based optimization demo for the
# Surferbot parameterization `(x_A, log(EI))`.

base_params = FlexibleParams(
    sigma = 1e-6,
    rho = 1000.0,
    nu = 1e-6,
    g = 10 * 9.81,
    L_raft = 0.1,
    motor_position = 0.5 * 0.1 / 2,
    d = 0.1 / 2,
    EI = 100 * 3.0e9 * 3e-2 * (9.9e-4)^3 / 12,
    rho_raft = 0.018 * 3.0,
    domain_depth = 0.2,
    n = 41,
    M = 30,
    motor_inertia = 0.13e-3 * 2.5e-3,
    bc = :radiative,
    omega = 2 * pi * 10,
    L_domain = 0.3,
)

theta0 = [base_params.motor_position, log(base_params.EI)]
opt_params = OptimizationParams(
    theta0 = theta0,
    lower = [0.0, log(base_params.EI / 100)],
    upper = [base_params.L_raft / 2, log(base_params.EI * 100)],
)

baseline = flexible_solver(base_params)
config = OptimizationConfig(
    Pmax = input_power(baseline),
    mu = 10.0,
    beta = 20.0,
    fd_step = 1e-6,
    sens_step = 1e-6,
)

result = run_thrust_optimization(base_params, opt_params, config; maxiter=10, gtol=1e-6)

println("theta = ", result.theta)
println("xA = ", result.theta[1])
println("EI = ", exp(result.theta[2]))
println("objective = ", result.objective)
println("thrust = ", result.thrust)
println("power_input = ", result.power_input)
println("gradient = ", result.gradient)
println("iterations = ", result.iterations)
println("converged = ", result.converged)
