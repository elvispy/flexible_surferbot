using Test
using Surferbot
using ForwardDiff
const SBO = Surferbot.SurferbotOptimization

const GRAD_BASE_PARAMS = FlexibleParams(
    sigma = 0.0,
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
    Nz = 30,
    motor_inertia = 0.13e-3 * 2.5e-3,
    bc = :radiative,
    omega = 2 * pi * 10,
    L_domain = 0.3,
)

function total_fd_gradient(theta, base_params, config; h=1e-6)
    grad = zeros(Float64, length(theta))
    for i in eachindex(theta)
        step = h * max(1.0, abs(theta[i]))
        tp = copy(theta)
        tm = copy(theta)
        tp[i] += step
        tm[i] -= step
        grad[i] = (SBO.thrust_objective(tp, base_params, config) - SBO.thrust_objective(tm, base_params, config)) / (2step)
    end
    grad
end

@testset "optimization gradients" begin
    theta = [0.012, log(GRAD_BASE_PARAMS.EI)]
    config = SBO.OptimizationConfig(Pmax = 1e-5, mu = 10.0, beta = 20.0, fd_step = 1e-6, sens_step = 1e-6)
    objective, grad, primal = SBO.objective_and_gradient(theta, GRAD_BASE_PARAMS, config)
    fd_grad = total_fd_gradient(theta, GRAD_BASE_PARAMS, config)

    @test objective isa Float64
    @test length(grad) == 2
    @test primal.outputs.thrust isa Float64

    # Two bugs fixed here (see Julia/src/optimization.jl and Julia/src/Surferbot.jl):
    # (1) objective_and_gradient's ForwardDiff.jacobian call was applied to a
    #     complex-valued function (vcat(vec(system.A), system.b)), which
    #     ForwardDiff.jacobian does not support -- it silently returned
    #     un-extracted, nested Dual garbage instead of erroring, so grad[1]
    #     (d/d motor_position) came out ~2.2x wrong. Fixed by differentiating
    #     a real-valued function (real/imag parts stacked) and reassembling
    #     the complex Jacobian afterward.
    # (2) That exposed a second bug: Surferbot.jl's nd_groups computed
    #     We = C/sigma and Re = C/nu, then used 1/We and 1/Re downstream. At
    #     sigma=0 or nu=0 (both common configurations -- the "no surface
    #     tension"/inviscid limits used here), this is fine in plain Float64
    #     (We=Inf, 1/We=0, the physically-correct vanishing-capillarity/
    #     inviscid limit) but produces Dual(Inf, NaN) under ForwardDiff, even
    #     though 1/We = sigma/C is a perfectly smooth function of sigma at
    #     sigma=0 -- the singularity was an artifact of the specific
    #     C/sigma-then-invert expression graph, not a real one. Fixed by
    #     computing inv_We = sigma/C and inv_Re = nu/C directly.
    @test isapprox(grad[1], fd_grad[1]; atol=1e-5, rtol=5e-3)
    @test isapprox(grad[2], fd_grad[2]; atol=1e-5, rtol=5e-3)
end

@testset "assemble_flexible_system AD-safe" begin
    # Isolated regression tests at the smallest reproducible unit: directly
    # differentiate assemble_flexible_system's system matrix (real/imag
    # parts) and check for finite (non-NaN) results, independent of the full
    # optimization pipeline above. Covers the removable-singularity fix (at
    # sigma=0 and nu=0), a sanity check that nu != 0 still works, and a
    # multi-material (graded, vector EI) configuration -- theta_to_params
    # itself still rejects vector EI/rho_raft (optimizing over graded beams
    # is a separate, unimplemented feature), so this checks only that
    # assemble_flexible_system's own AD path handles graded beams correctly.
    #
    # Uses a local field-override helper rather than
    # Surferbot.Sweep.apply_parameter_overrides: that helper doesn't promote
    # every FlexibleParams field to a common type, so passing a single Dual
    # override alongside otherwise-Float64 fields fails to construct the
    # (uniformly-typed) FlexibleParams{T}. dual_params mirrors
    # theta_to_params's cast-every-field style instead.
    function dual_params(base_params::FlexibleParams, field::Symbol, value)
        T = value isa AbstractVector ? eltype(value) : typeof(value)
        fields = Dict{Symbol,Any}()
        for name in fieldnames(FlexibleParams)
            v = getfield(base_params, name)
            fields[name] = if name == field
                value
            elseif v isa AbstractVector
                T.(v)
            elseif isnothing(v)
                v
            elseif v isa Symbol || v isa Integer
                v
            else
                T(v)
            end
        end
        return FlexibleParams{T}(; fields...)
    end

    function finite_jacobian(overrides_fn, value)
        f = (s) -> begin
            p = overrides_fn(s)
            sys = Surferbot.assemble_flexible_system(p)
            vcat(real.(vec(sys.A)), imag.(vec(sys.A)))
        end
        all(isfinite, ForwardDiff.derivative(f, value))
    end

    @test finite_jacobian(s -> dual_params(GRAD_BASE_PARAMS, :sigma, s), 0.0)
    @test finite_jacobian(s -> dual_params(GRAD_BASE_PARAMS, :nu, s), 0.0)
    @test finite_jacobian(s -> dual_params(GRAD_BASE_PARAMS, :nu, s), 1e-6)

    base = Surferbot.derive_params(GRAD_BASE_PARAMS)
    nb = base.nb_contact
    EI_vec = vcat(fill(GRAD_BASE_PARAMS.EI, nb ÷ 2), fill(GRAD_BASE_PARAMS.EI * 10, nb - nb ÷ 2))
    multi_material_fn = (s) -> begin
        EI_perturbed = typeof(s).(EI_vec)
        EI_perturbed[1] = s
        dual_params(GRAD_BASE_PARAMS, :EI, EI_perturbed)
    end
    @test finite_jacobian(multi_material_fn, EI_vec[1])
end
