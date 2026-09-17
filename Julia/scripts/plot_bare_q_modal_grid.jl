"""
plot_bare_q_modal_grid.jl

Bare modal response |q_n| vs kappa, 2x2 grid:
    rows    = even modes, odd modes
    columns = uncoupled (Lambda=0), coupled

Companion to the Uncoupled limit Lambda=0 appendix section: shows the free-free
beam's own resonant modal geometry alongside the fully coupled case, split by
parity.  Vertical lines mark the zero-thrust stiffnesses G(kappa)=0, found by
bisecting the paper's scalar calP(kappa); in the uncoupled column that
condition degenerates to the dry poles kappa=(beta_n*L)^-4.

  x = kappa (log), y = |q_n| (log)
  color = phase, period 360 deg: phase_t(theta) = (1-cosd(theta))/2 * 0.92
      (endpoint compressed slightly so 0 deg and +-180 deg are distinguishable)
  line thickness = mode number (thicker = higher mode)
"""

using CairoMakie, LaTeXStrings, LinearAlgebra, JLD2

const WC = joinpath(homedir(), "Documents", "Github", "waves_code", "Julia")
include(joinpath(WC, "scripts", "plot_dimensionless_diagnostics_LH.jl"))
using .PaperPlotTheme

art = Surferbot.Sweep.load_sweep(joinpath(WC, "output", "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))
params = art.base_params
ctx = theoretical_modal_context_LH(params; output_dir=joinpath(WC, "output"))
EI_scale = Float64(params.rho_raft) * Float64(params.L_raft)^4 * Float64(params.omega)^2
xM = -0.12

function solve_uncoupled(EI, xM_norm, c)
    p = c.params; F_c = c.derived.F_c; L_c = c.derived.L_c
    loads = (c.F0 / F_c) .* Surferbot.gaussian_load(Float64(xM_norm), p.forcing_width, c.x_raft ./ L_c) .* (F_c / L_c)
    F_psi = c.Psi' * (loads .* c.weights)
    D = ComplexF64.(EI .* c.beta .^ 4 .- p.rho_raft * p.omega^2)
    return -(Diagonal(D) \ ComplexF64.(F_psi))
end

kappa_grid = 10 .^ range(log10(3e-6), log10(1.0); length=2500)
nmodes = length(ctx.mode_numbers)

function sweep(solver)
    mag = zeros(Float64, nmodes, length(kappa_grid))
    phase = zeros(Float64, nmodes, length(kappa_grid))
    for (ik, k) in enumerate(kappa_grid)
        q = solver(k * EI_scale, xM, ctx)
        # solve_theoretical_modal_response/solve_uncoupled return a DIMENSIONAL
        # q (loads are converted back to dimensional units before the solve,
        # see loads_dim in solve_theoretical_modal_response). Divide by L_raft
        # to get the non-dimensional bar-q_n plotted here, matching the same
        # convention used for the production Fig 5/6 modal amplitudes
        # (plot_kappa_snapshot.jl: mode_energy = abs.(modal.q) ./ L_raft).
        mag[:, ik] = abs.(q) ./ Float64(params.L_raft)
        phase[:, ik] = rad2deg.(angle.(q))
    end
    return mag, phase
end

mag_c, phase_c = sweep((EI, xm, c) -> solve_theoretical_modal_response(EI, xm, c))
mag_u, phase_u = sweep(solve_uncoupled)

kpole_dry_all = [(m, Float64(params.rho_raft) * Float64(params.omega)^2 / (EI_scale * b^4))
                 for (m, b) in zip(ctx.mode_numbers, ctx.beta) if b > 0]
kpole_dry_even = [k for (m, k) in kpole_dry_all if iseven(m)]
kpole_dry_odd  = [k for (m, k) in kpole_dry_all if isodd(m)]

# The vertical lines mark G(kappa) = 0, the zero-thrust condition of the main
# text: at these stiffnesses Re(S A*) = 0 for every forcing profile.
#
# G = cos(delta_e - delta_o) * b_e b_o', so G vanishes where the even and odd
# transfer phases are in quadrature.  Evaluating that condition through the
# delta_p directly is not robust: b_p is real but its entries change sign, and
# each sign change injects a spurious pi jump (it yields 11 crossings here, of
# which only 6 are genuine).  The paper's own scalar
#
#     calP(kappa) = Re[ e^{i(theta_e - theta_o)} P_e(kappa)* P_o(kappa) ]
#
# (eq:modal_kappa_polynomial) carries no such branch problem, is real, and
# changes sign at each root, so it is bisected instead.  theta_p is the common
# phase of the radiation vector a_p (eq:modal_radiation_direction, verified
# here to 5e-7 degrees across each block).
#
# The roots are JOINT: calP mixes both parity blocks, so there is one set of
# six, not three per parity.  Each root is placed in the panel whose block
# supplies it, determined from the response peak it accompanies -- the pairing
# is 1:1 and strictly parity-alternating, because a block's phase swings by
# ~pi across its own resonance (measured 124-147 deg here) and that swing is
# what drives delta_e - delta_o through pi/2.  In the uncoupled column the same
# condition degenerates to the dry poles kappa=(beta_n*L)^-4 already drawn
# there, which is the Lambda -> 0 statement made in the main text.
function response_peaks(mag, idx; minprom=0.30)
    env = vec(maximum(mag[idx, :], dims=1))
    peaks = Float64[]
    for i in 2:length(env)-1
        (env[i] > env[i-1] && env[i] >= env[i+1]) || continue
        l = i; while l > 1 && env[l-1] < env[l]; l -= 1; end
        r = i; while r < length(env) && env[r+1] < env[r]; r += 1; end
        min(log10(env[i]/env[l]), log10(env[i]/env[r])) >= minprom || continue
        push!(peaks, kappa_grid[i])
    end
    return peaks
end

function modal_block(kappa, idx)
    D = ComplexF64.(kappa * EI_scale .* ctx.beta .^ 4
                    .- Float64(params.rho_raft) * Float64(params.omega)^2
                    .+ ctx.c_hydro)
    return (Diagonal(D) .- ctx.Z_psi .+ ctx.C_sigma)[idx, idx]
end

# roots of calP(kappa) = 0, i.e. G(kappa) = 0
function zero_thrust_roots(even_idx, odd_idx; klo=3e-6, khi=1.0, ngrid=40000, nbisect=80)
    th(idx) = angle(ComplexF64(ctx.a_vec[idx][argmax(abs.(ctx.a_vec[idx]))]))
    dth = th(even_idx) - th(odd_idx)
    calP(k) = real(cis(dth) * conj(det(modal_block(k, even_idx))) * det(modal_block(k, odd_idx)))
    kg = 10 .^ range(log10(klo), log10(khi); length=ngrid)
    v = calP.(kg)
    roots = Float64[]
    for i in 1:length(kg)-1
        (isfinite(v[i]) && isfinite(v[i+1]) && sign(v[i]) != sign(v[i+1])) || continue
        a, b, sa = kg[i], kg[i+1], sign(v[i])
        for _ in 1:nbisect
            m = sqrt(a * b)
            sign(calP(m)) == sa ? (a = m) : (b = m)
        end
        push!(roots, sqrt(a * b))
    end
    return roots
end

const EVEN_IDX = [i for (i, m) in enumerate(ctx.mode_numbers) if iseven(m)]
const ODD_IDX  = [i for (i, m) in enumerate(ctx.mode_numbers) if isodd(m)]

# Assign each joint G=0 root to the parity block whose resonance it accompanies.
let pk_e = response_peaks(mag_c, EVEN_IDX), pk_o = response_peaks(mag_c, ODD_IDX)
    roots = zero_thrust_roots(EVEN_IDX, ODD_IDX)
    global validated_even = Float64[]
    global validated_odd  = Float64[]
    for r in roots
        de = isempty(pk_e) ? Inf : minimum(abs.(log10.(pk_e ./ r)))
        do_ = isempty(pk_o) ? Inf : minimum(abs.(log10.(pk_o ./ r)))
        push!(de <= do_ ? validated_even : validated_odd, r)
    end
    @info "G(kappa)=0 roots" all=roots even=validated_even odd=validated_odd
    for (nm, rs, pk) in (("even", validated_even, pk_e), ("odd", validated_odd, pk_o))
        for r in rs
            j = argmin(abs.(log10.(pk ./ r)))
            @info "  $nm G-root vs accompanying resonance" root=r resonance=pk[j] reldiff=abs(pk[j]-r)/r
        end
    end
end

# Phase -> ONE shared colorbar, raw phase degrees, period 360 deg. t=0 (phase=0)
# and t=1 (phase=+-180) are the two endpoints of :balance, both nearly-black
# at full saturation; compress the mapped range to [0, 0.92] so phase=0 stays
# at the full dark-navy end but phase=+-180 lands on a lighter, more clearly
# red/orange shade instead of near-black maroon.
const PHASE_CMAP = cgrad(:balance)
const PHASE_T_MAX = 0.92
phase_t(theta_deg) = (1 .- cosd.(theta_deg)) ./ 2 .* PHASE_T_MAX
phase_color(phase_deg::AbstractVector) = [PHASE_CMAP[t] for t in phase_t(phase_deg)]

function draw_panel!(ax, mag, phase, kpoles, parity_modes)
    for (j, m) in enumerate(ctx.mode_numbers)
        m in parity_modes || continue
        cols = phase_color(phase[j, :])
        lw = 1.0 + 1.1 * j
        lines!(ax, kappa_grid, mag[j, :]; color=cols, linewidth=lw)
    end
    vlines!(ax, kpoles; color=:gray50, linestyle=:dashdot, linewidth=1.0)
    xlims!(ax, extrema(kappa_grid))
end

const EVEN_MODES = [m for m in ctx.mode_numbers if iseven(m)]
const ODD_MODES  = [m for m in ctx.mode_numbers if isodd(m)]

fig = PaperPlotTheme.with_theme() do
    fig = Figure(size=(880, 780), backgroundcolor=:white, fontsize=18, figure_padding=(6, 8, 4, 4))

    # rows = even/odd, columns = uncoupled/coupled
    # Styling matches plot_fig10_modal_maps_3x3.jl: fontsize=18 for headers and
    # axis labels, 16 for tick labels and the colorbar ticks, no bold headers,
    # minor x-ticks/x-grid and y-grid disabled.
    header_kw = (fontsize=18, tellwidth=false, tellheight=true)
    Label(fig[1,1], "Uncoupled (Λ=0)"; header_kw...)
    Label(fig[1,2], "Coupled"; header_kw...)
    Label(fig[2,0], "Even modes"; rotation=pi/2, fontsize=18, tellwidth=true, tellheight=false)
    Label(fig[3,0], "Odd modes"; rotation=pi/2, fontsize=18, tellwidth=true, tellheight=false)

    axis_kw = (xlabelsize=18, ylabelsize=18, xticklabelsize=16, yticklabelsize=16,
        xminorticksvisible=false, xminorgridvisible=false, yminorgridvisible=false)
    ax_ue = Axis(fig[2,1]; xscale=log10, yscale=log10, ylabel=L"|\bar{q}_n|", axis_kw...)
    ax_ce = Axis(fig[2,2]; xscale=log10, yscale=log10, axis_kw...)
    ax_uo = Axis(fig[3,1]; xscale=log10, yscale=log10, xlabel=L"\kappa", ylabel=L"|\bar{q}_n|", axis_kw...)
    ax_co = Axis(fig[3,2]; xscale=log10, yscale=log10, xlabel=L"\kappa", axis_kw...)

    draw_panel!(ax_ue, mag_u, phase_u, kpole_dry_even, EVEN_MODES)
    draw_panel!(ax_ce, mag_c, phase_c, validated_even, EVEN_MODES)
    draw_panel!(ax_uo, mag_u, phase_u, kpole_dry_odd,  ODD_MODES)
    draw_panel!(ax_co, mag_c, phase_c, validated_odd,  ODD_MODES)

    linkxaxes!(ax_ue, ax_uo, ax_ce, ax_co)
    linkyaxes!(ax_ue, ax_uo, ax_ce, ax_co)
    ax_ue.xticklabelsvisible = false
    ax_ce.xticklabelsvisible = false
    ax_ce.yticklabelsvisible = false
    ax_co.yticklabelsvisible = false

    phase_ticks_deg = -180:90:180
    Colorbar(fig[2:3, 3];
        colormap = cgrad([PHASE_CMAP[t] for t in phase_t(collect(-180.0:1.0:180.0))]),
        limits = (-180, 180), ticks = collect(phase_ticks_deg),
        label = "phase (deg)", labelsize=18, ticklabelsize=16)

    legend_elems = Any[]; legend_labels = String[]
    for (j, m) in enumerate(ctx.mode_numbers)
        lw = 1.0 + 1.1 * j
        push!(legend_elems, LineElement(color=:gray20, linewidth=lw))
        push!(legend_labels, "mode $m")
    end
    Legend(fig[4, 1:2], legend_elems, legend_labels; orientation=:horizontal,
        framevisible=false, labelsize=16, patchsize=(24, 10), tellheight=true, nbanks=1)

    colgap!(fig.layout, 6)
    rowgap!(fig.layout, 4)
    resize_to_layout!(fig)

    # Saved as PNG, not PDF: CairoMakie keeps per-vertex-colored lines() as
    # vector polygons in PDF output, so px_per_unit never rasterizes them and
    # the antialiasing haze reappears whenever a PDF viewer re-rasterizes at
    # its own zoom level. PNG forces Makie to rasterize everything (including
    # the lines) once, at this resolution, avoiding that entirely.
    out = joinpath(WC, "output", "figures", "plot_bare_q_modal_grid.png")
    save(out, fig; px_per_unit=6)
    println("Saved $out")
    fig
end
