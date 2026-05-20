"""
benham_comparison.jl

Validate that our Julia solver reproduces the Benham 2024 MATLAB results.

Parameters from MATLAB/old_code/surferbot.m (second block).
The MATLAB reference data (/tmp/benham_eta.csv) was produced with:
  sigma = 0  (surfaceTension = 0*0.073 in surferbot.m)
  nu    = 0  (kinematicViscosity = 0*1e-6 in surferbot.m)
  EI    = Inf (rigid raft)

We test sigma = 0 (match MATLAB) and sigma = 0.073 (Benham paper formulation)
to quantify the impact of surface tension on amplitude.

Motor forcing: F_z = M_raft * ω² * A,  A = 100 * 0.13e-3 * 2.5e-3 m.

Success criterion: L2 relative error on eta(x) < 1% (for sigma=0 case).
"""

using Surferbot
using Printf
using LinearAlgebra
using CSV
using DataFrames

# ─── Shared parameters ───────────────────────────────────────────────────────

const raft_mass = 0.009
const f_Hz      = 5.0
const ω         = 2π * f_Hz
const L         = 0.05
const d         = 0.03
const motor_pos = -0.3 * L / 3
const osc_amp   = 100 * 0.13e-3 * 2.5e-3
const F_motor   = raft_mass * ω^2 * osc_amp

# ─── Verify dimensionless groups ─────────────────────────────────────────────

Ω_check  = ω * sqrt(L / 9.81)
M_check  = (raft_mass / L) / (1000.0 * d * L)
xA_check = motor_pos / L
Fz_check = F_motor / (1000.0 * 9.81 * L^2 * d)

@printf "─── Dimensionless groups ────────────────────────────────────────────\n"
@printf "  Ω  = %.6f  (MATLAB: %.6f)\n"  Ω_check  2π*f_Hz*sqrt(L/9.81)
@printf "  M  = %.6f  (MATLAB: 0.120000)\n"  M_check
@printf "  xA = %.6f  (MATLAB: -0.100000)\n"  xA_check
@printf "  Fz = %.6e  (MATLAB: %.6e)\n"  Fz_check  Fz_check
@printf "────────────────────────────────────────────────────────────────────\n\n"

# ─── Load MATLAB reference ───────────────────────────────────────────────────

csv_path = "/tmp/benham_eta.csv"
isfile(csv_path) || error("Run the MATLAB script first: $csv_path not found")
df     = CSV.read(csv_path, DataFrame; header=false)
x_ml   = Float64.(df[!, 1])
eta_ml = Complex{Float64}.(df[!, 2] .+ im .* df[!, 3])

@printf "─── MATLAB reference (sigma=0, nu=0) ───────────────────────────────\n"
@printf "  |η_left_end|   = %9.4f µm\n"  abs(eta_ml[1])*1e6
@printf "  |η_right_end|  = %9.4f µm\n"  abs(eta_ml[end])*1e6
@printf "  max |η|        = %9.4f µm\n"  maximum(abs.(eta_ml))*1e6
@printf "────────────────────────────────────────────────────────────────────\n\n"

# ─── Interpolation helper ────────────────────────────────────────────────────

function interp1_linear(xs, ys, xq)
    n = length(xs)
    out = similar(ys, length(xq))
    for (j, x) in enumerate(xq)
        i = searchsortedfirst(xs, x) - 1
        i = clamp(i, 1, n - 1)
        t = (x - xs[i]) / (xs[i+1] - xs[i])
        out[j] = (1 - t) * ys[i] + t * ys[i+1]
    end
    return out
end

function compare_to_matlab(sigma_val, EI_val, label)
    params = Surferbot.FlexibleParams(
        sigma          = sigma_val,
        rho            = 1000.0,
        omega          = ω,
        nu             = 0.0,
        g              = 9.81,
        L_raft         = L,
        motor_position = motor_pos,
        d              = d,
        EI             = EI_val,
        rho_raft       = raft_mass / L,
        motor_force    = F_motor,
        L_domain       = 2 * 10 * L,
    )

    result  = Surferbot.flexible_solver(params)
    eta_jl  = result.eta
    x_jl    = result.x

    x_lo = max(x_jl[1],   x_ml[1])
    x_hi = min(x_jl[end], x_ml[end])
    mask = x_lo .<= x_jl .<= x_hi

    eta_ml_on_jl = interp1_linear(x_ml, real.(eta_ml), x_jl[mask]) .+
                   im .* interp1_linear(x_ml, imag.(eta_ml), x_jl[mask])
    eta_jl_mask  = eta_jl[mask]

    l2_err   = norm(eta_jl_mask .- eta_ml_on_jl) / norm(eta_ml_on_jl) * 100
    linf_err = maximum(abs.(eta_jl_mask .- eta_ml_on_jl)) / maximum(abs.(eta_ml_on_jl)) * 100

    @printf "─── %s ─────\n" label
    @printf("  Julia:  |η_left|=%8.4f µm  |η_right|=%8.4f µm  max=%8.4f µm\n",
        abs(eta_jl[1])*1e6, abs(eta_jl[end])*1e6, maximum(abs.(eta_jl))*1e6)
    @printf("  L2 err = %.4f %%   Linf err = %.4f %%   %s\n",
        l2_err, linf_err, l2_err < 1.0 ? "✓ PASS" : "✗ FAIL")
    @printf "────────────────────────────────────────────────────────────────────\n"
    return l2_err
end

# ─── Sweep ───────────────────────────────────────────────────────────────────

@printf "─── Julia sigma=0 (matching MATLAB) ────────────────────────────────\n\n"
compare_to_matlab(0.0, Inf, "sigma=0,     EI=Inf")
compare_to_matlab(0.0, 1e4, "sigma=0,     EI=1e4")
@printf "\n─── Julia sigma=0.073 (Benham paper) ───────────────────────────────\n\n"
compare_to_matlab(0.073, Inf, "sigma=0.073, EI=Inf")
compare_to_matlab(0.073, 1e4, "sigma=0.073, EI=1e4")
