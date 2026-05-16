"""
plot_thrust_beam_and_LH.jl

Two heatmaps on the (log₁₀κ, xM/L) plane per coupling case:
  1. Beam  — Δ|η|²/L²  at the raft endpoints      (eta_{1,end}_beam columns)
  2. LH    — Δ|η|²/L²  at the computational-domain ends  (Longuet-Higgins proxy)
             (eta_{1,end}_domain columns)

Δ|η|² = |η_end|² − |η_1|²  (right minus left; same sign convention as α).

Two colour variants for the LH panel:
  :signed_log — signed-log₁₀ scale (kept for reference)
  :gp         — GP posterior mean (squared-exponential kernel) on a dense 200×200
                grid, fitted on asinh-transformed values then back-transformed to
                raw Δ|η|²/L² units for a plain linear colorbar; training set uses
                only the EI values inside the displayed xlims and 75 % of the xM
                values (skip every 4th point → 75 of 100 xM values).

Source CSVs: output/csv/sweeper_{coupled,uncoupled}_full_grid.csv
"""

using CSV
using DataFrames
using Plots
using LaTeXStrings
using Printf
using Statistics
using LinearAlgebra
using Surferbot

# ─── Load and reshape ────────────────────────────────────────────────────────

function load_delta_grids(csv_path::AbstractString; pref::Float64)
    df = CSV.read(csv_path, DataFrame)

    log10_EI_vals = sort(unique(Float64.(df.log10_EI)))
    xM_vals       = sort(unique(Float64.(df.xM_over_L)))
    nEI = length(log10_EI_vals)
    nxM = length(xM_vals)

    beam_grid   = fill(NaN, nxM, nEI)
    domain_grid = fill(NaN, nxM, nEI)

    L        = Float64(first(df.L_raft))
    rho_raft = Float64(first(df.rho_raft))
    omega    = Float64(first(df.omega))

    idx = Dict{Tuple{Float64,Float64}, Tuple{Int,Int}}(
        (log10_EI_vals[j], xM_vals[i]) => (i, j)
        for j in 1:nEI for i in 1:nxM)

    for row in eachrow(df)
        k = (Float64(row.log10_EI), Float64(row.xM_over_L))
        haskey(idx, k) || continue
        i, j = idx[k]
        η1b = complex(row.eta_1_beam_re,    row.eta_1_beam_im)
        ηEb = complex(row.eta_end_beam_re,  row.eta_end_beam_im)
        η1d = complex(row.eta_1_domain_re,  row.eta_1_domain_im)
        ηEd = complex(row.eta_end_domain_re, row.eta_end_domain_im)
        beam_grid[i, j]   = (abs2(ηEb) - abs2(η1b)) / L^2
        domain_grid[i, j] = pref * (abs2(ηEd) - abs2(η1d)) / (rho_raft * L * omega^2)
    end

    return (; log10_EI = log10_EI_vals, xM = xM_vals, beam_grid, domain_grid,
              L, rho_raft, omega)
end

# ─── Colour transforms ───────────────────────────────────────────────────────

function signed_log10_grid(mat::AbstractMatrix{Float64}; eps_frac=1e-6)
    finite_vals = filter(isfinite, vec(mat))
    isempty(finite_vals) && return fill(NaN, size(mat))
    maxabs = maximum(abs, finite_vals)
    ε = maxabs * eps_frac + 1e-30
    return @. sign(mat) * log10(abs(mat) + ε)
end

# sign(x)·∛|x| — smooth through zero, moderate dynamic-range compression.
# Less aggressive than log (moderate values still get distinct colours), no
# scale parameter needed, derivative is finite everywhere except x=0.
function cbrt_grid(mat::AbstractMatrix{Float64})
    return @. sign(mat) * cbrt(abs(mat))
end

# ─── GP surrogate ────────────────────────────────────────────────────────────
# Squared-exponential kernel GP.  Fitted internally on asinh-transformed values
# (for numerical stability across the huge dynamic range), then back-transformed
# to raw units so the output can be displayed with a plain linear colorbar.
# Length scales are set to match the feature scale of the data rather than the
# fine grid spacing (which the heuristic 3×median would give).

function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector;
                  lx=0.4, ly=0.06)
    n = length(values)
    mean_value = mean(values)
    centered   = collect(Float64.(values .- mean_value))
    sigma_f    = max(std(values), 1e-3)
    noise      = max(1e-4, 2e-2 * sigma_f)

    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        r2 = ((x[i]-x[j])/lx)^2 + ((y[i]-y[j])/ly)^2
        kij = sigma_f^2 * exp(-0.5 * r2)
        K[i,j] = kij; K[j,i] = kij
    end
    K[diagind(K)] .+= noise^2 + 1e-10

    F = cholesky(Symmetric(K))
    weights = F \ centered
    return (; x=Float64.(x), y=Float64.(y), weights, mean=mean_value,
              lx, ly, sigma_f2=sigma_f^2)
end

function predict_gp2d(model, xq::Real, yq::Real)
    acc = 0.0
    for i in eachindex(model.weights)
        r2 = ((xq - model.x[i]) / model.lx)^2 + ((yq - model.y[i]) / model.ly)^2
        acc += model.weights[i] * model.sigma_f2 * exp(-0.5 * r2)
    end
    return model.mean + acc
end

# Build GP on the visible region only.
# Training set: EI values with log₁₀κ ≥ xlims[1]; xM downsampled by 25%
# (skip every 4th point → 75 of 100 xM values).
# GP is fitted on asinh-transformed values, predictions are back-transformed
# to raw Δ|η|²/L² for display with a linear colorbar.
function gp_surrogate_lh(log10_kappa_full, xM_full, delta_grid;
                          xlims=(-4.0, maximum(log10_kappa_full)),
                          out_n=200)
    # Filter EI to visible range
    ki      = findall(xlims[1] .<= log10_kappa_full .<= xlims[2])
    kappa_in = log10_kappa_full[ki]
    grid_in  = delta_grid[:, ki]

    # Keep 75 % of xM values: drop every 4th point
    xi      = [i for i in 1:length(xM_full) if mod(i, 4) != 0]
    xM_in   = xM_full[xi]
    grid_in = grid_in[xi, :]

    n_train = length(xM_in) * length(kappa_in)
    @info "GP training set: $(length(kappa_in)) κ × $(length(xM_in)) xM = $n_train points"

    # Flatten, drop NaN
    xt, yt, vt = Float64[], Float64[], Float64[]
    for (ie, ke) in enumerate(kappa_in), (im, xm) in enumerate(xM_in)
        v = grid_in[im, ie]
        isfinite(v) || continue
        push!(xt, ke); push!(yt, xm); push!(vt, v)
    end

    # asinh transform for GP fitting; remember scale for back-transform
    maxabs = maximum(abs, vt)
    s      = maxabs * 1e-6 + 1e-30
    vt_a   = asinh.(vt ./ s)

    @info "Fitting GP..."
    model = fit_gp2d(xt, yt, vt_a)

    # Predict on dense output grid and back-transform to raw units
    kappa_out = collect(range(xlims[1], xlims[2]; length=out_n))
    xM_out    = collect(range(minimum(xM_full), maximum(xM_full); length=out_n))
    @info "Predicting on $(out_n)×$(out_n) grid..."
    pred_a = [predict_gp2d(model, k, x) for x in xM_out, k in kappa_out]
    pred   = @. s * sinh(pred_a)

    return kappa_out, xM_out, pred
end

# ─── Operating-point marker ───────────────────────────────────────────────────

function operating_point(bp, shift)
    lk = log10(Float64(bp.EI)) - shift
    xm = abs(Float64(bp.motor_position)) / Float64(bp.L_raft)
    return lk, xm
end

# ─── Render one panel ─────────────────────────────────────────────────────────

function render_panel(log10_kappa, xM_axis, delta_grid, fig_title, out_base, bp, shift;
                      mode=:signed_log)
    max_logK = maximum(log10_kappa)
    XLIMS    = (-4.0, max_logK)
    YLIMS    = (0.0, 0.5)

    # Clim from raw data in visible κ range — done before transform so ticks are in real units.
    ki_vis   = findall(XLIMS[1] .<= log10_kappa .<= XLIMS[2])
    raw_vis  = filter(isfinite, vec(delta_grid[:, ki_vis]))
    clim_raw = isempty(raw_vis) ? 1.0 : quantile(abs.(raw_vis), 0.95)
    clim_raw = max(clim_raw, 1e-30)

    if mode == :signed_log
        c        = signed_log10_grid(delta_grid)
        clim_val = maximum(abs, filter(isfinite, vec(c[:, ki_vis])))
        cbtitle  = L"\mathrm{sgn}(\Delta)\cdot\log_{10}|\Delta\hat\eta^2/L^2|"
        cbticks  = :auto
        kp, xp, cp = log10_kappa, xM_axis, c

    elseif mode == :cbrt
        cp       = cbrt_grid(delta_grid)
        clim_val = cbrt(clim_raw)
        kp, xp   = log10_kappa, xM_axis
        cbtitle  = L"\Delta S_{xx}\,/\,(\rho_R L \omega^2)"

        # Colorbar ticks: decade-spaced raw values, positioned in cbrt space.
        # The uneven spacing on the colorbar axis communicates the nonlinear scale.
        log_max   = floor(Int, log10(clim_raw))
        pos_vals  = [10.0^k for k in (log_max-3):log_max]
        raw_ticks = vcat(-reverse(pos_vals), [0.0], pos_vals)
        cbrt_pos  = @. sign(raw_ticks) * cbrt(abs(raw_ticks))
        tick_labs = [v == 0.0 ? "0" : @sprintf("%.0e", v) for v in raw_ticks]
        cbticks   = (cbrt_pos, tick_labs)
    end

    plt_opts = (
        xlabel             = L"\log_{10}\,\kappa",
        ylabel             = L"x_M / L",
        colormap           = cgrad(:RdBu, rev=true),
        clims              = (-clim_val, clim_val),
        colorbar_ticks     = cbticks,
        interpolate        = true,
        xlims              = XLIMS,
        ylims              = YLIMS,
        legend             = :bottomright,
        background_color_legend = RGBA(1, 1, 1, 0.85),
        foreground_color_legend = :black,
        legendfontsize     = 11,
        colorbar           = true,
        colorbar_title     = cbtitle,
        colorbar_titlefontsize     = 11,
        colorbar_titlerotation     = 270,
        colorbar_tickfontsize      = 11,
        size               = (820, 640),
        margin             = 6Plots.mm,
        right_margin       = 16Plots.mm,
        dpi                = 220,
        titlefontsize      = 14,
        guidefontsize      = 14,
        tickfontsize       = 12,
        fontfamily         = "Computer Modern",
        framestyle         = :box,
        grid               = false,
    )

    p = heatmap(kp, xp, cp; plt_opts...)

    lk_star, xm_star = operating_point(bp, shift)
    scatter!(p, [lk_star], [xm_star];
             marker=:star5, markersize=12, color=:white,
             markerstrokecolor=:black, markerstrokewidth=1, label="SurferBot")

    savefig(p, out_base * ".pdf")
    println("Saved $(out_base).pdf")
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    csv_path = joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv")
    if !isfile(csv_path)
        @warn "sweeper_coupled_full_grid.csv not found — skipping."
        return
    end
    bp      = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    bp_fp   = Surferbot.FlexibleParams(; (k => getproperty(bp, k)
                  for k in fieldnames(Surferbot.FlexibleParams) if hasproperty(bp, k))...)
    derived = Surferbot.derive_params(bp_fp)
    k_val   = real(derived.k)
    pref    = Float64(bp.rho) * Float64(bp.g) / 4 + 3/4 * Float64(bp.sigma) * k_val^2

    println("Loading data...")
    grids = load_delta_grids(csv_path; pref)
    shift = log10(grids.rho_raft * grids.L^4 * grids.omega^2)
    log10_kappa = grids.log10_EI .- shift
    Lambda_val = @sprintf("%.2f", Float64(bp.d) / Float64(bp.L_raft))

    # Beam — signed-log only (unchanged)
    render_panel(log10_kappa, grids.xM, grids.beam_grid,
        LaTeXString("Coupled, \$\\Lambda=$Lambda_val\$ — beam \$\\Delta|\\eta|^2/L^2\$"),
        joinpath(fig_dir, "plot_thrust_beam_coupled"), bp, shift; mode=:signed_log)

    # LH — cube-root transform with real-unit colorbar ticks
    lh_title = LaTeXString("Coupled, \$\\Lambda=$Lambda_val\$ — LH \$\\Delta|\\eta|^2/L^2\$")
    println("Rendering LH cbrt...")
    render_panel(log10_kappa, grids.xM, grids.domain_grid, lh_title,
        joinpath(fig_dir, "plot_thrust_LH_coupled_cbrt"), bp, shift; mode=:cbrt)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
