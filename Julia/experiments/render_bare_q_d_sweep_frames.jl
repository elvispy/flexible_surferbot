"""
render_bare_q_d_sweep_frames.jl

Supplementary-material video source: sweeps the raft width d (equivalently
Lambda = d/L) from 0 up to 3x the SurferBot reference value, and renders one
Figure-10-style 2x2 frame per d. The left (uncoupled) column is fixed
throughout (it's the Lambda=0 reference and doesn't depend on d at all); only
the right (coupled) column evolves per frame. Axis and colorbar limits are
fixed across all frames (computed from the strongest-coupling frame) so the
viewer can see amplitudes actually change rather than axes silently rescaling.

Resonance vlines are found automatically per frame (max-over-modes envelope,
local-maxima detection with a minimum prominence), since hand-tuned exact
root locations aren't practical across 25 different d values.

Usage: julia --project=. experiments/render_bare_q_d_sweep_frames.jl
Then:  ffmpeg -framerate 3 -i frame_%03d.png -vf "fps=3,format=yuv420p" out.mp4
"""

using CairoMakie, LaTeXStrings, LinearAlgebra, Printf

const WC = joinpath(homedir(), "Documents", "Github", "waves_code", "Julia")
include(joinpath(WC, "scripts", "plot_dimensionless_diagnostics_LH.jl"))
using .PaperPlotTheme

const FRAME_DIR = joinpath(WC, "output", "figures", "d_sweep_frames")
mkpath(FRAME_DIR)

art = Surferbot.Sweep.load_sweep(joinpath(WC, "output", "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))
p0 = art.base_params
xM = -0.12
const NFRAMES = 25
const D_REF = Float64(p0.d)
const L_RAFT = Float64(p0.L_raft)
const D_MAX = 3 * D_REF
const D_VALUES = collect(range(0.0, D_MAX; length=NFRAMES))

function with_d(p0, dnew)
    return Surferbot.FlexibleParams(; sigma=p0.sigma, rho=p0.rho, omega=p0.omega, nu=p0.nu, g=p0.g,
        L_raft=p0.L_raft, motor_position=p0.motor_position, d=dnew, EI=p0.EI, rho_raft=p0.rho_raft,
        L_domain=p0.L_domain, domain_depth=p0.domain_depth, n=p0.n, Nz=p0.M, ooa=p0.ooa,
        motor_inertia=p0.motor_inertia, motor_force=p0.motor_force, forcing_width=p0.forcing_width, bc=p0.bc)
end

function solve_uncoupled(EI, xM_norm, c)
    p = c.params; F_c = c.derived.F_c; L_c = c.derived.L_c
    loads = (c.F0 / F_c) .* Surferbot.gaussian_load(Float64(xM_norm), p.forcing_width, c.x_raft ./ L_c) .* (F_c / L_c)
    F_psi = c.Psi' * (loads .* c.weights)
    D = ComplexF64.(EI .* c.beta .^ 4 .- p.rho_raft * p.omega^2)
    return -(Diagonal(D) \ ComplexF64.(F_psi))
end

kappa_grid = 10 .^ range(log10(3e-6), log10(1.0); length=2500)

function sweep(solver, ctx, EI_scale)
    nmodes = length(ctx.mode_numbers)
    mag = zeros(Float64, nmodes, length(kappa_grid))
    phase = zeros(Float64, nmodes, length(kappa_grid))
    for (ik, k) in enumerate(kappa_grid)
        q = solver(k * EI_scale, xM, ctx)
        mag[:, ik] = abs.(q) ./ L_RAFT
        phase[:, ik] = rad2deg.(angle.(q))
    end
    return mag, phase
end

# --- fixed uncoupled (Lambda=0) reference, computed once ---
ctx0 = theoretical_modal_context_LH(p0; output_dir=joinpath(WC, "output"))
EI_scale = Float64(p0.rho_raft) * Float64(p0.L_raft)^4 * Float64(p0.omega)^2
mag_u, phase_u = sweep(solve_uncoupled, ctx0, EI_scale)

kpole_dry_all = [(m, Float64(p0.rho_raft) * Float64(p0.omega)^2 / (EI_scale * b^4))
                 for (m, b) in zip(ctx0.mode_numbers, ctx0.beta) if b > 0]
kpole_dry_even = [k for (m, k) in kpole_dry_all if iseven(m)]
kpole_dry_odd  = [k for (m, k) in kpole_dry_all if isodd(m)]

const EVEN_MODES = [m for m in ctx0.mode_numbers if iseven(m)]
const ODD_MODES  = [m for m in ctx0.mode_numbers if isodd(m)]

const PHASE_CMAP = cgrad(:balance)
const PHASE_T_MAX = 0.92
phase_t(theta_deg) = (1 .- cosd.(theta_deg)) ./ 2 .* PHASE_T_MAX
phase_color(phase_deg::AbstractVector) = [PHASE_CMAP[t] for t in phase_t(phase_deg)]

"""Automatic resonance-line detector: local maxima of the max-over-modes
envelope with a minimum prominence, merging near-duplicate kappa's."""
function auto_resonance_kappas(mag, modes_idx; min_prominence_db=6.0, merge_log_tol=0.05)
    envelope = vec(maximum(mag[modes_idx, :]; dims=1))
    logenv = log10.(max.(envelope, eps()))
    peaks = Float64[]
    for i in 2:length(logenv)-1
        if logenv[i] > logenv[i-1] && logenv[i] >= logenv[i+1]
            lo = max(1, i-30); hi = min(length(logenv), i+30)
            background = minimum(logenv[lo:hi])
            if (logenv[i] - background) * 20 >= min_prominence_db  # crude dB-like prominence
                push!(peaks, kappa_grid[i])
            end
        end
    end
    merged = Float64[]
    for k in sort(peaks)
        if isempty(merged) || abs(log10(k) - log10(merged[end])) > merge_log_tol
            push!(merged, k)
        end
    end
    return merged
end

function draw_panel!(ax, mag, phase, kpoles, parity_modes, ctx_modes)
    for (j, m) in enumerate(ctx_modes)
        m in parity_modes || continue
        cols = phase_color(phase[j, :])
        lw = 1.0 + 1.1 * j
        lines!(ax, kappa_grid, mag[j, :]; color=cols, linewidth=lw)
    end
    vlines!(ax, kpoles; color=:gray50, linestyle=:dashdot, linewidth=1.0)
    xlims!(ax, extrema(kappa_grid))
end

# --- pass 1: find the strongest-coupling frame's data to fix axis limits ---
p_max = with_d(p0, D_MAX)
ctx_max = theoretical_modal_context_LH(p_max; output_dir=joinpath(WC, "output"))
mag_c_max, _ = sweep((EI, xm, c) -> solve_theoretical_modal_response(EI, xm, c), ctx_max, EI_scale)
YMIN = 10.0^floor(log10(minimum(filter(x -> x > 0, vcat(vec(mag_u), vec(mag_c_max))))))
YMAX = 10.0^ceil(log10(maximum(vcat(vec(mag_u), vec(mag_c_max)))))
println("Fixed y-limits: ", (YMIN, YMAX))

# --- pass 2: render one frame per d ---
for (fi, d) in enumerate(D_VALUES)
    Lambda = d / L_RAFT
    ctx_d = d == D_MAX ? ctx_max : theoretical_modal_context_LH(with_d(p0, d); output_dir=joinpath(WC, "output"))
    mag_c, phase_c = d == D_MAX ? (mag_c_max, sweep((EI, xm, c) -> solve_theoretical_modal_response(EI, xm, c), ctx_max, EI_scale)[2]) :
                                   sweep((EI, xm, c) -> solve_theoretical_modal_response(EI, xm, c), ctx_d, EI_scale)

    validated_even = auto_resonance_kappas(mag_c, [j for (j,m) in enumerate(ctx_d.mode_numbers) if iseven(m)])
    validated_odd  = auto_resonance_kappas(mag_c, [j for (j,m) in enumerate(ctx_d.mode_numbers) if isodd(m)])

    fig = PaperPlotTheme.with_theme() do
        fig = Figure(size=(880, 820), backgroundcolor=:white, fontsize=18, figure_padding=(6, 8, 4, 4))

        header_kw = (fontsize=18, tellwidth=false, tellheight=true)
        Label(fig[1,1], "Uncoupled (Λ=0)"; header_kw...)
        Label(fig[1,2], "Coupled"; header_kw...)
        Label(fig[2,0], "Even modes"; rotation=pi/2, fontsize=18, tellwidth=true, tellheight=false)
        Label(fig[3,0], "Odd modes"; rotation=pi/2, fontsize=18, tellwidth=true, tellheight=false)
        Label(fig[0,1:2], @sprintf("d = %.4f m   (Λ = %.3f)", d, Lambda); fontsize=20, font=:bold, tellwidth=false)

        axis_kw = (xlabelsize=18, ylabelsize=18, xticklabelsize=16, yticklabelsize=16,
            xminorticksvisible=false, xminorgridvisible=false, yminorgridvisible=false)
        ax_ue = Axis(fig[2,1]; xscale=log10, yscale=log10, ylabel=L"|\bar{q}_n|", axis_kw...)
        ax_ce = Axis(fig[2,2]; xscale=log10, yscale=log10, axis_kw...)
        ax_uo = Axis(fig[3,1]; xscale=log10, yscale=log10, xlabel=L"\kappa", ylabel=L"|\bar{q}_n|", axis_kw...)
        ax_co = Axis(fig[3,2]; xscale=log10, yscale=log10, xlabel=L"\kappa", axis_kw...)

        draw_panel!(ax_ue, mag_u, phase_u, kpole_dry_even, EVEN_MODES, ctx0.mode_numbers)
        draw_panel!(ax_ce, mag_c, phase_c, validated_even, EVEN_MODES, ctx_d.mode_numbers)
        draw_panel!(ax_uo, mag_u, phase_u, kpole_dry_odd,  ODD_MODES, ctx0.mode_numbers)
        draw_panel!(ax_co, mag_c, phase_c, validated_odd,  ODD_MODES, ctx_d.mode_numbers)

        # Set limits on every axis BEFORE linking: linkyaxes! autolimits any
        # axis whose limits aren't already fixed, and autolimits on a frame
        # with any non-finite plotted value crashes ("reducing over an empty
        # collection") rather than just ignoring it.
        for ax in (ax_ue, ax_ce, ax_uo, ax_co)
            xlims!(ax, extrema(kappa_grid))
            ylims!(ax, YMIN, YMAX)
        end
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
        for (j, m) in enumerate(ctx0.mode_numbers)
            lw = 1.0 + 1.1 * j
            push!(legend_elems, LineElement(color=:gray20, linewidth=lw))
            push!(legend_labels, "mode $m")
        end
        Legend(fig[4, 1:2], legend_elems, legend_labels; orientation=:horizontal,
            framevisible=false, labelsize=16, patchsize=(24, 10), tellheight=true, nbanks=1)

        colgap!(fig.layout, 6)
        rowgap!(fig.layout, 4)
        resize_to_layout!(fig)
        fig
    end

    out = joinpath(FRAME_DIR, @sprintf("frame_%03d.png", fi))
    save(out, fig; px_per_unit=4)
    println("Saved $out  (d=$d, Lambda=$Lambda, $fi/$NFRAMES)")
end
