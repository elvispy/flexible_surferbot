module SurferbotOptimization

using LinearAlgebra
using SparseArrays
using Surferbot
using ForwardDiff
using LinearSolve

export OptimizationParams,
       OptimizationConfig,
       OptimizationResult,
       softplus_penalty,
       theta_to_params,
       input_power,
       thrust_objective,
       objective_and_gradient,
       run_thrust_optimization

"""
    OptimizationParams

Container for optimization variables and simple box constraints.

# Fields
- `theta0`: Initial values for `[xA, logEI]`.
- `lower`: Lower bounds for the parameters.
- `upper`: Upper bounds for the parameters.
"""
Base.@kwdef struct OptimizationParams
    theta0::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

"""
    OptimizationConfig

Configuration for the penalized thrust optimization problem.

# Fields
- `Pmax`: Actuator power budget.
- `mu`: Power penalty weight (default: 1.0).
- `beta`: Softplus sharpness parameter (default: 50.0).
- `gamma`: Curvature penalty weight (default: 0.0).
- `kappa_max_limit`: Dimensionless curvature threshold (default: 1.0).
- `delta`: Wave steepness penalty weight (default: 0.0).
- `ak_limit`: Wave steepness threshold (default: 0.1).
- `fd_step`: Step size for postprocessed derivatives (default: 1e-6).
- `sens_step`: Step size for sensitivity assembly (default: 1e-6).
"""
Base.@kwdef struct OptimizationConfig
    Pmax::Float64
    mu::Float64 = 1.0
    beta::Float64 = 50.0
    gamma::Float64 = 0.0
    kappa_max_limit::Float64 = 1.0
    delta::Float64 = 0.0
    ak_limit::Float64 = 0.1
    fd_step::Float64 = 1e-6
    sens_step::Float64 = 1e-6
end

"""
    OptimizationResult

Summary of a completed optimization run.

# Fields
- `theta`: Final optimization variables `[xA, logEI]`.
- `objective`: Final penalized objective value.
- `thrust`: Final mean thrust.
- `power_input`: Final mean input power.
- `gradient`: Final objective gradient.
- `iterations`: Number of iterations performed.
- `converged`: Whether convergence criteria were met.
"""
struct OptimizationResult
    theta::Vector{Float64}
    objective::Float64
    thrust::Float64
    power_input::Float64
    gradient::Vector{Float64}
    iterations::Int
    converged::Bool
end

"""
    softplus(z::Real, beta::Real)

Smooth approximation of `max(z, 0)`.

# Arguments
- `z`: Input value.
- `beta`: Sharpness parameter.

# Returns
- Smoothly rectified value.
"""
function softplus(z::Real, beta::Real)
    if beta * z > 40
        return float(z)
    elseif beta * z < -40
        return exp(beta * z) / beta
    else
        return log1p(exp(beta * z)) / beta
    end
end

"""
    softplus_penalty(power_input, Pmax, mu; beta=50.0)

Calculate the smooth penalty for exceeding the power budget.

# Arguments
- `power_input`: Calculated actuator input power.
- `Pmax`: Power budget.
- `mu`: Penalty weight.
- `beta`: Sharpness parameter (default: 50.0).

# Returns
- Calculated penalty value.
"""
function softplus_penalty(power_input::Real, Pmax::Real, mu::Real; beta::Real=50.0)
    excess = power_input - Pmax
    return mu * softplus(excess, beta)^2
end

"""
    theta_to_params(theta, base_params)

Map optimization variables `theta = [xA, logEI]` to `FlexibleParams`.

# Arguments
- `theta`: Vector of optimization variables.
- `base_params`: Template parameters for constant values.

# Returns
- A new `FlexibleParams` instance.
"""
function theta_to_params(theta::AbstractVector{<:Real}, base_params::FlexibleParams)
    (base_params.EI isa AbstractVector || base_params.rho_raft isa AbstractVector) && error(
        "theta_to_params does not support spatially varying EI or rho_raft. " *
        "Optimization with graded beam params is not yet implemented.")
    @assert length(theta) == 2
    T = eltype(theta)
    return FlexibleParams{T}(;
        sigma = T(base_params.sigma),
        rho = T(base_params.rho),
        omega = T(base_params.omega),
        nu = T(base_params.nu),
        g = T(base_params.g),
        L_raft = T(base_params.L_raft),
        motor_position = theta[1],
        d = isnothing(base_params.d) ? nothing : T(base_params.d),
        EI = exp(theta[2]),
        rho_raft = T(base_params.rho_raft),
        L_domain = isnothing(base_params.L_domain) ? nothing : T(base_params.L_domain),
        domain_depth = isnothing(base_params.domain_depth) ? nothing : T(base_params.domain_depth),
        n = base_params.n,
        Nz = base_params.Nz,
        ooa = base_params.ooa,
        motor_inertia = T(base_params.motor_inertia),
        motor_force = isnothing(base_params.motor_force) ? nothing : T(base_params.motor_force),
        forcing_width = T(base_params.forcing_width),
        bc = base_params.bc,
    )
end

"""
    input_power(power::Real)
    input_power(result)

Convert solver power (negative for input) to positive input power.

# Returns
- Positive input power value.
"""
input_power(power::Real) = -float(power)
input_power(result) = input_power(result.power)

"""
    build_output_args(derived, params)

Assemble arguments for output calculation from derived states and parameters.
"""
function build_output_args(derived, params)
    return (
        sigma = params.sigma,
        rho = params.rho,
        omega = params.omega,
        nu = params.nu,
        g = params.g,
        L_raft = params.L_raft,
        d = derived.d,
        nd_groups = derived.nd_groups,
        x_contact = derived.x_contact,
        x = derived.x .* derived.L_c,
        loads = derived.loads .* derived.F_c ./ derived.L_c,
        Nx = derived.Nx,
        Nz = derived.Nz,
        dx = derived.dx .* derived.L_c,
        dz = derived.dz .* derived.L_c,
        t_c = derived.t_c,
        L_c = derived.L_c,
        m_c = derived.m_c,
        k = derived.k,
        ooa = params.ooa,
    )
end

"""
    split_state(solution::AbstractVector, derived)

Split the stacked state vector into potential and its derivative matrices.
"""
function split_state(solution::AbstractVector, derived)
    NP = derived.Nx * derived.Nz
    phi = reshape(solution[1:NP], derived.Nz, derived.Nx)
    phi_z = reshape(solution[(NP + 1):(2 * NP)], derived.Nz, derived.Nx)
    return phi, phi_z
end

"""
    evaluate_from_state(solution, derived, params, config)

Calculate objective and physical outputs from a solved state.

# Arguments
- `solution`: State vector.
- `derived`: Derived grid/system information.
- `params`: Physical parameters.
- `config`: Optimization configuration.

# Returns
- NamedTuple of objective components and physical fields.
"""
function evaluate_from_state(solution::AbstractVector, derived, params, config::OptimizationConfig)
    phi, phi_z = split_state(solution, derived)
    args = build_output_args(derived, params)
    U, power, thrust, eta, p, max_curvature, wave_steepness = Surferbot.calculate_surferbot_outputs(args, phi, phi_z, Surferbot.getNonCompactFDmatrix, Surferbot.getNonCompactFDmatrix2D)
    Pin = input_power(power)
    
    # Power penalty
    p_penalty = softplus_penalty(Pin, config.Pmax, config.mu; beta=config.beta)
    
    # Curvature penalty
    c_penalty = config.gamma * softplus(max_curvature - config.kappa_max_limit, config.beta)^2
    
    # Wave steepness penalty
    s_penalty = config.delta * softplus(wave_steepness - config.ak_limit, config.beta)^2
    
    objective = -thrust + p_penalty + c_penalty + s_penalty
    
    return (
        objective = objective,
        thrust = thrust,
        power = power,
        power_input = Pin,
        penalty = p_penalty,
        curvature_penalty = c_penalty,
        steepness_penalty = s_penalty,
        max_curvature = max_curvature,
        wave_steepness = wave_steepness,
        U = U,
        eta = eta,
        pressure = p,
        phi = phi,
        phi_z = phi_z,
        args = args,
    )
end

"""
    evaluate_primal(theta, base_params, config)

Solve the forward problem for parameters `theta`.

# Returns
- NamedTuple containing system state and postprocessed outputs.
"""
function evaluate_primal(theta::AbstractVector{<:Real}, base_params::FlexibleParams, config::OptimizationConfig)
    params = theta_to_params(theta, base_params)
    system = Surferbot.assemble_flexible_system(params)
    full_solution = Surferbot.solve_tensor_system(system.A, system.b)
    NP = system.derived.Nx * system.derived.Nz
    state = full_solution[1:(2 * NP)]
    outputs = evaluate_from_state(state, system.derived, params, config)
    return (
        params = params,
        system = system,
        full_solution = full_solution,
        state = state,
        outputs = outputs,
    )
end

"""
    thrust_objective(theta, base_params, config)

Compute the scalar penalized objective for `theta`.
"""
function thrust_objective(theta::AbstractVector{<:Real}, base_params, config::OptimizationConfig)
    eval = evaluate_primal(theta, base_params, config)
    return eval.outputs.objective
end

"""
    clamp_theta(theta, opt_params)

Clamp optimization variables to their defined bounds.
"""
function clamp_theta(theta, opt_params::OptimizationParams)
    return clamp.(theta, opt_params.lower, opt_params.upper)
end

"""
    central_step(value::Real, base_step::Real)

Compute a scale-aware step size for central differences.
"""
function central_step(value::Real, base_step::Real)
    return base_step * max(1.0, abs(float(value)))
end

"""
    directional_objective_derivative(state, state_direction, theta, theta_direction, derived, base_params, config)

Compute the directional derivative of the objective via finite differences.
"""
function directional_objective_derivative(state, state_direction, theta, theta_direction, derived, base_params, config::OptimizationConfig)
    h = config.fd_step
    theta_plus = theta .+ h .* theta_direction
    theta_minus = theta .- h .* theta_direction
    params_plus = theta_to_params(theta_plus, base_params)
    params_minus = theta_to_params(theta_minus, base_params)
    derived_plus = Surferbot.derive_params(params_plus)
    derived_minus = Surferbot.derive_params(params_minus)
    out_plus = evaluate_from_state(state .+ h .* state_direction, derived_plus, params_plus, config)
    out_minus = evaluate_from_state(state .- h .* state_direction, derived_minus, params_minus, config)
    return (out_plus.objective - out_minus.objective) / (2h)
end

"""
    objective_and_gradient(theta, base_params, config)

Calculate the objective value and its gradient using implicit differentiation.

# Returns
- `(objective, gradient, primal_evaluation)`.
"""
function objective_and_gradient(theta::AbstractVector{<:Real}, base_params, config::OptimizationConfig)
    primal = evaluate_primal(theta, base_params, config)
    A = primal.system.A
    z = primal.full_solution
    grad = zeros(Float64, length(theta))

    # Compute all assembly derivatives at once using AD. ForwardDiff.jacobian
    # does not support complex-valued outputs (it silently returns wrong,
    # un-extracted Dual values instead of erroring), so differentiate a
    # real-valued function that stacks the real/imaginary parts, then
    # reassemble a complex Jacobian with the same shape/slicing the
    # sensitivity-solve loop below expects.
    mn = length(A)
    nb_len = length(primal.system.b)
    f_real = (th) -> begin
        params = theta_to_params(th, base_params)
        system = Surferbot.assemble_flexible_system(params)
        Avec = vec(system.A)
        return vcat(real.(Avec), imag.(Avec), real.(system.b), imag.(system.b))
    end
    J_real = ForwardDiff.jacobian(f_real, theta)
    J = vcat(
        Complex.(J_real[1:mn, :],             J_real[(mn + 1):(2mn), :]),
        Complex.(J_real[(2mn + 1):(2mn + nb_len), :], J_real[(2mn + nb_len + 1):end, :]),
    )

    # Solve for sensitivities using our Dual-safe helper
    m, n_sys = size(A)
    for idx in eachindex(theta)
        dA_vec = @view J[1:(m*n_sys), idx]
        db = @view J[(m*n_sys + 1):end, idx]
        dA = reshape(dA_vec, m, n_sys)
        
        rhs = db - dA * z
        sensitivity_full = Surferbot.solve_tensor_system(A, rhs)
        
        sensitivity_state = sensitivity_full[1:length(primal.state)]
        theta_dir = zeros(Float64, length(theta))
        theta_dir[idx] = 1.0
        val = directional_objective_derivative(primal.state, sensitivity_state, theta, theta_dir, primal.system.derived, base_params, config)
        grad[idx] = ForwardDiff.value(val)
    end

    return primal.outputs.objective, grad, primal
end

"""
    backtracking_line_search(theta, direction, objective_value, gradient, base_params, config, opt_params)

Armijo-rule backtracking line search for the local optimizer.
"""
function backtracking_line_search(theta, direction, objective_value, gradient, base_params, config, opt_params)
    step = 1.0
    c1 = 1e-4
    directional = dot(gradient, direction)
    while step > 1e-8
        trial = clamp_theta(theta .+ step .* direction, opt_params)
        trial_objective = thrust_objective(trial, base_params, config)
        if trial_objective <= objective_value + c1 * step * directional
            return step, trial_objective
        end
        step *= 0.5
    end
    return 0.0, objective_value
end

"""
    lbfgs_direction(gradient, s_history, y_history)

Compute search direction using the Limited-memory BFGS algorithm.
"""
function lbfgs_direction(gradient, s_history, y_history)
    q = copy(gradient)
    m = length(s_history)
    alpha = zeros(Float64, m)
    rho = zeros(Float64, m)

    for i in m:-1:1
        rho[i] = 1 / dot(y_history[i], s_history[i])
        alpha[i] = rho[i] * dot(s_history[i], q)
        q .-= alpha[i] .* y_history[i]
    end

    if m > 0
        gamma = dot(s_history[end], y_history[end]) / dot(y_history[end], y_history[end])
        r = gamma .* q
    else
        r = copy(q)
    end

    for i in 1:m
        beta = rho[i] * dot(y_history[i], r)
        r .+= s_history[i] .* (alpha[i] - beta)
    end

    return -r
end

"""
    run_thrust_optimization(base_params, opt_params, config; maxiter=20, gtol=1e-6, memory=5)

Optimize the Surferbot design variables to maximize penalized thrust.

# Arguments
- `base_params`: Template physical parameters.
- `opt_params`: Optimization variable bounds and initial guess.
- `config`: Penalty and step size settings.
- `maxiter`: Maximum iterations (default: 20).
- `gtol`: Gradient norm tolerance (default: 1e-6).
- `memory`: L-BFGS history length (default: 5).

# Returns
- An `OptimizationResult` object.
"""
function run_thrust_optimization(base_params, opt_params::OptimizationParams, config::OptimizationConfig; maxiter::Int=20, gtol::Float64=1e-6, memory::Int=5)
    theta = clamp_theta(copy(opt_params.theta0), opt_params)
    objective, gradient, primal = objective_and_gradient(theta, base_params, config)
    s_history = Vector{Vector{Float64}}()
    y_history = Vector{Vector{Float64}}()
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        if norm(gradient) <= gtol
            converged = true
            break
        end

        direction = lbfgs_direction(gradient, s_history, y_history)
        step, _ = backtracking_line_search(theta, direction, objective, gradient, base_params, config, opt_params)
        if step == 0.0
            break
        end

        theta_next = clamp_theta(theta .+ step .* direction, opt_params)
        objective_next, gradient_next, primal_next = objective_and_gradient(theta_next, base_params, config)

        s = theta_next - theta
        y = gradient_next - gradient
        ys = dot(y, s)
        if ys > 1e-12
            push!(s_history, collect(s))
            push!(y_history, collect(y))
            if length(s_history) > memory
                popfirst!(s_history)
                popfirst!(y_history)
            end
        end

        theta = theta_next
        objective = objective_next
        gradient = gradient_next
        primal = primal_next
    end

    if norm(gradient) <= gtol
        converged = true
    end

    return OptimizationResult(
        collect(theta),
        objective,
        primal.outputs.thrust,
        primal.outputs.power_input,
        collect(gradient),
        iterations,
        converged,
    )
end

end
