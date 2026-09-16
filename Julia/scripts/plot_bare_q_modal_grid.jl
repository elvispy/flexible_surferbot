"""
plot_bare_q_modal_grid.jl

Bare modal response |q_n| vs kappa, 2x2 grid:
    rows    = even modes, odd modes
    columns = uncoupled (Lambda=0), coupled

Companion to the Uncoupled limit Lambda=0 appendix section: shows the free-free
beam's own resonant modal geometry (dry poles kappa=beta_n^-4) alongside the
fully coupled case (S-perp-A roots), split by parity.

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

# Coupled (S-perp-A) resonance roots, split by parity. The 4 roots originally
# found (via root search on the coupled modal system) missed the lowest-kappa
# (most-flexible) resonance for both parities; refined by local log-grid
# maximization of |q| (200000 points) around the observed peaks in mag_c,
# giving 3.7638e-5 (even) and 1.6602e-5 (odd) in addition to the 3 already
# found per parity (odd/even alternate with increasing kappa).
validated_even = sort([3.7637590044590734e-5, 3.5877272926336350e-4, 1.9567552784652463e-2])
validated_odd  = sort([1.6601568914776633e-5, 1.0274283400507156e-4, 1.8552764122338056e-3])

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
