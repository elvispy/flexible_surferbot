using Surferbot
using JLD2
using Plots
using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Statistics
import SpecialFunctions: erf

"""
Julia/experiments/plot_four_panel_diagnostics.jl

Generates four diagnostic figures:
1. Uncoupled Theoretical
2. Uncoupled Integral
3. Coupled Theoretical
4. Coupled Integral

Each figure overlays 6 zero-crossing curves on an alpha heatmap:
- S_{02} = 0
- S_{0246} = 0
- A_{13} = 0
- A_{1357} = 0
- |eta_1| = 0
- |eta_end| = 0
"""

# --- Root-Finding Utils ---

function all_positive_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]
        b = vals[i + 1]
        if a == 0
            xs[i] > 1e-6 && push!(roots, Float64(xs[i]))
        elseif a * b < 0
            # Linear interpolation for high-precision crossing
            t = a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            root > 1e-6 && push!(roots, Float64(root))
        end
    end
    return unique!(roots)
end

function complex_roots(xs::AbstractVector{<:Real}, re_vals::AbstractVector{<:Real}, im_vals::AbstractVector{<:Real}; threshold=1e-3)
    # Find crossings of Re(eta) and filter by Im(eta) near 0
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = re_vals[i]
        b = re_vals[i + 1]
        if a * b <= 0
            t = a == b ? 0.0 : a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            
            # Interpolate Imaginary part at the root
            im_root = im_vals[i] + t * (im_vals[i+1] - im_vals[i])
            if abs(im_root) < threshold
                push!(roots, Float64(root))
            end
        end
    end
    return unique!(roots)
end

# --- Modal Framework Helpers ---

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    Phi = zeros(Float64, length(xi_motor), max_mode + 1)
    
    xi_L = [0.0]
    xi_R = [L]
    Phi_L = zeros(Float64, 1, max_mode + 1)
    Phi_R = zeros(Float64, 1, max_mode + 1)

    Phi[:, 1] .= 1.0
    Phi_L[1, 1] = 1.0
    Phi_R[1, 1] = 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
        Phi_L[1, 2] = -L / 2
        Phi_R[1, 2] = L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_L[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_L, L, betaL_el[n - 1])[1]
            Phi_R[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_R, L, betaL_el[n - 1])[1]
        end
    end
    return (; xM_norm=collect(Float64.(xM_norm)), Phi, Phi_L=vec(Phi_L), Phi_R=vec(Phi_R))
end

# Placeholder for coupled pressure projection law
function Q_n_theoretical(xM, EI, params, n)
    return 0.0 + 0.0im # Discover the law later
end

# --- Root Extraction ---

function get_basis_for_plotting(params)
    fparams = if params isa Surferbot.FlexibleParams
        params
    else
        # Handle JLD2 reconstructed types or NamedTuples
        Surferbot.FlexibleParams(; 
            [k => getfield(params, k) for k in fieldnames(typeof(params)) if k in fieldnames(Surferbot.FlexibleParams)]...
        )
    end
    res = Surferbot.flexible_solver(fparams)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(res; num_modes=8, verbose=false)
    # Phi is raw analytical W, Psi is orthonormalized
    return (Phi=modal.Phi, Psi=modal.Psi, x=modal.x_raft) 
end

function get_roots_integral(csv_path, condition_name; modes=0:7)
    df = CSV.read(csv_path, DataFrame)
    L_raft = first(df.L_raft)
    basis = get_basis_for_plotting((L_raft=L_raft,))
    
    # CSV saves modal.q, modal.Q, modal.F which are PSI basis coefficients.
    # Therefore, weights MUST come from basis.Psi
    w_end = basis.Psi[end, :]
    w_start = basis.Psi[1, :]

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for group in groupby(df, :log10_EI)
        logEI = first(group.log10_EI)
        xM_slice = group.xM_over_L
        
        re_vals = Float64[]
        im_vals = Float64[]
        
        for row in eachrow(group)
            val = 0.0 + 0.0im
            if condition_name == "S02"
                for n in (0, 2)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "S0246"
                for n in (0, 2, 4, 6)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "A13"
                for n in (1, 3)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "A1357"
                for n in (1, 3, 5, 7)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "eta_1"
                for n in 0:7
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_start[n+1]
                end
            elseif condition_name == "eta_end"
                for n in 0:7
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            end
            push!(re_vals, real(val))
            push!(im_vals, imag(val))
        end
        
        roots = if startswith(condition_name, "eta")
            complex_roots(xM_slice, re_vals, im_vals)
        else
            all_positive_roots(xM_slice, re_vals)
        end
        
        for r in roots
            push!(pts_logEI, logEI)
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function gaussian_load_nd(x0, sigma, x, L_raft)
    a, b = -L_raft/2, L_raft/2
    phi = exp.(-0.5 .* ((x .- x0) ./ sigma).^2)
    Z = sigma * sqrt(pi / 2) * (erf((b - x0) / (sqrt(2) * sigma)) - erf((a - x0) / (sqrt(2) * sigma)))
    return (1 / Z) .* phi
end

function get_roots_theoretical(artifact, condition_name)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    xM_grid = collect(range(0.0, 0.49, length=1000))
    
    # Reconstruct exact Solver Basis (Psi)
    fparams = if params isa Surferbot.FlexibleParams
        params
    else
        Surferbot.FlexibleParams(; 
            [k => getfield(params, k) for k in fieldnames(typeof(params)) if k in fieldnames(Surferbot.FlexibleParams)]...
        )
    end
    res = Surferbot.flexible_solver(fparams)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(res; num_modes=8, verbose=false)
    Psi = modal.Psi
    x_raft = modal.x_raft
    w_weights = Surferbot.Modal.trapz_weights(x_raft)
    w_end = Psi[end, :]
    w_start = Psi[1, :]

    # Master equation constants
    L = params.L_raft
    Dfun(EI, beta) = EI * beta^4 - params.rho_raft * params.omega^2
    beta_roots = modal.beta # Use exact betas from solver reconstruction
    
    # Motor Source Force
    F0 = params.motor_inertia * params.omega^2
    sigma_f = 0.05 * L
    
    pts_logEI = Float64[]
    pts_xM = Float64[]

    for (iei, EI) in enumerate(EI_list)
        re_vals = Float64[]
        im_vals = Float64[]
        
        for (ix, xM_norm) in enumerate(xM_grid)
            xi_xM = xM_norm * L
            
            # 1. Project Gaussian forcing onto Psi basis
            load_dist = F0 .* gaussian_load_nd(xi_xM, sigma_f, x_raft, L)
            F_psi = Psi' * (load_dist .* w_weights)
            
            # 2. Calculate q_n
            q = zeros(ComplexF64, 8)
            for n in 0:7
                Qn = Q_n_theoretical(xi_xM, EI, params, n)
                Dn = Dfun(EI, beta_roots[n+1])
                q[n+1] = (Qn - F_psi[n+1]) / Dn
            end
            
            val = 0.0 + 0.0im
            if condition_name == "S02"
                val = q[1]*w_end[1] + q[3]*w_end[3]
            elseif condition_name == "S0246"
                val = q[1]*w_end[1] + q[3]*w_end[3] + q[5]*w_end[5] + q[7]*w_end[7]
            elseif condition_name == "A13"
                val = q[2]*w_end[2] + q[4]*w_end[4]
            elseif condition_name == "A1357"
                val = q[2]*w_end[2] + q[4]*w_end[4] + q[6]*w_end[6] + q[8]*w_end[8]
            elseif condition_name == "eta_1"
                for n in 0:7
                    val += q[n+1] * w_start[n+1]
                end
            elseif condition_name == "eta_end"
                for n in 0:7
                    val += q[n+1] * w_end[n+1]
                end
            end
            push!(re_vals, real(val))
            push!(im_vals, imag(val))
        end
        
        roots = if startswith(condition_name, "eta")
            complex_roots(xM_grid, re_vals, im_vals)
        else
            all_positive_roots(xM_grid, re_vals)
        end
        
        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

# --- GPR Smoothing Logic ---

function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector)
    n = length(values)
    mean_value = mean(values)
    centered = collect(Float64.(values .- mean_value))

    dx = diff(sort(unique(Float64.(x))))
    dy = diff(sort(unique(Float64.(y))))
    dx = dx[dx .> 0]
    dy = dy[dy .> 0]
    # Length scales: heuristic based on grid spacing
    lx = isempty(dx) ? 0.05 : max(0.02, 3 * median(dx))
    ly = isempty(dy) ? 0.15 : max(0.05, 3 * median(dy))
    sigma_f = max(std(values), 1e-3)
    noise = max(1e-4, 2e-2 * sigma_f)

    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        r2 = ((x[i] - x[j]) / lx)^2 + ((y[i] - y[j]) / ly)^2
        kij = sigma_f^2 * exp(-0.5 * r2)
        K[i, j] = kij
        K[j, i] = kij
    end
    for i in 1:n
        K[i, i] += noise^2 + 1e-10
    end

    F = cholesky(Symmetric(K))
    weights = F \ centered
    return (x = Float64.(x), y = Float64.(y), weights = weights, mean = mean_value, lx = lx, ly = ly, sigma_f2 = sigma_f^2, noise=noise)
end

function predict_gp2d(model, xq::Real, yq::Real)
    acc = 0.0
    for i in eachindex(model.weights)
        r2 = ((xq - model.x[i]) / model.lx)^2 + ((yq - model.y[i]) / model.ly)^2
        acc += model.weights[i] * (model.sigma_f2 * exp(-0.5 * r2))
    end
    return model.mean + acc
end

# --- Main Logic ---

function main()
    output_dir = joinpath(@__DIR__, "../..", "output")
    configs = [
        (name="unc_theo", coupled=false, source=:theoretical),
        (name="unc_int",  coupled=false, source=:integral),
        (name="cpl_theo", coupled=true,  source=:theoretical),
        (name="cpl_int",  coupled=true,  source=:integral)
    ]

    # Modular Toggle for GPR Smoothing
    USE_GPR = true 

    for cfg in configs
        println("Processing $(cfg.name)...")
        
        # Load Heatmap Data (JLD2 for Alpha background)
        jld2_file = cfg.coupled ? 
            "sweep_motor_position_EI_coupled_from_matlab.jld2" : 
            "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
        artifact = load_sweep(joinpath(output_dir, "jld2", jld2_file))
        
        logEI_axis = log10.(collect(Float64.(artifact.parameter_axes.EI)))
        xM_axis = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
        alpha = beam_asymmetry.(map(s->s.eta_left_beam, artifact.summaries), map(s->s.eta_right_beam, artifact.summaries))
        
        # --- Background GPR Smoothing ---
        local p_heatmap
        if USE_GPR
            println("  Fitting GPR background...")
            xtrain, ytrain, vtrain = Float64[], Float64[], Float64[]
            # Note: training feature space (x=logEI, y=xM)
            for (ie, lei) in enumerate(logEI_axis), (im, xm) in enumerate(xM_axis)
                push!(xtrain, lei)
                push!(ytrain, xm)
                push!(vtrain, alpha[im, ie])
            end
            model = fit_gp2d(xtrain, ytrain, vtrain)
            
            # Predict on a very dense grid for smoothness
            logEI_dense = collect(range(minimum(logEI_axis), maximum(logEI_axis), length=200))
            xM_dense = collect(range(minimum(xM_axis), maximum(xM_axis), length=200))
            alpha_dense = [predict_gp2d(model, lei, xm) for xm in xM_dense, lei in logEI_dense]
            
            p_heatmap = (logEI_dense, xM_dense, alpha_dense)
        else
            p_heatmap = (logEI_axis, xM_axis, alpha)
        end

        # Extract 6 Curves
        curve_names = ["S02", "S0246", "A13", "A1357", "eta_1", "eta_end"]
        results = Dict()
        
        if cfg.source == :integral
            csv_file = cfg.coupled ? "sweeper_coupled_full_grid.csv" : "sweeper_uncoupled_full_grid.csv"
            csv_path = joinpath(output_dir, "csv", csv_file)
            for cname in curve_names
                results[cname] = get_roots_integral(csv_path, cname)
            end
        else
            for cname in curve_names
                results[cname] = get_roots_theoretical(artifact, cname)
            end
        end
        
        # Plotting with expert scientific visualization
        # Okabe-Ito Colorblind-Safe Palette
        okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        
        plt_opts = (
            xlabel="log10(EI)", ylabel="x_M / L", 
            colormap=:balance, clims=(-1, 1), 
            levels=51, # High granularity for smooth gradients
            interpolate=true,
            xlims=(minimum(logEI_axis), maximum(logEI_axis)),
            ylims=(minimum(xM_axis), maximum(xM_axis)),
            legend=:topright, size=(1000, 750), margin=10Plots.mm,
            # Font sizes optimized for readability
            titlefontsize=14, guidefontsize=11, tickfontsize=9, legendfontsize=9,
            fontfamily="sans-serif"
        )
        
        # Determine Title
        fig_title = if cfg.name == "unc_theo"
            "Uncoupled Theoretical (Placeholder Qn=0)"
        elseif cfg.name == "unc_int"
            "Uncoupled Integral"
        elseif cfg.name == "cpl_theo"
            "Coupled Theoretical"
        elseif cfg.name == "cpl_int"
            "Coupled Integral"
        end

        p = heatmap(p_heatmap[1], p_heatmap[2], p_heatmap[3], title=fig_title; plt_opts...)
        
        # Define high-contrast symbols and Okabe-Ito colors
        # Curves: S02, S0246, A13, A1357, eta_1, eta_end
        curve_colors = [okabe_ito[1], okabe_ito[2], okabe_ito[3], okabe_ito[4], okabe_ito[5], okabe_ito[6]]
        markers = [:circle, :rect, :diamond, :utriangle, :star5, :star8]
        labels = ["S02=0", "S0246=0", "A13=0", "A1357=0", "|eta_1|=0", "|eta_end|=0"]
        
        for (i, cname) in enumerate(curve_names)
            res = results[cname]
            scatter!(p, res.logEI, res.xM_norm, 
                label=labels[i], color=curve_colors[i], marker=markers[i], 
                markersize=8, markerstrokewidth=0)
        end
        
        out_path = joinpath(output_dir, "figures", "plot_four_panel_diagnostics_$(cfg.name).pdf")
        savefig(p, out_path)
        println("Saved $out_path")
    end
end

main()
