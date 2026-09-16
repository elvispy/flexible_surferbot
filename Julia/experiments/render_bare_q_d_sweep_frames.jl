"""
render_bare_q_d_sweep_frames.jl

Supplementary-material video source: sweeps the raft width d (equivalently
Lambda = d/L) from 0 up to 3x the SurferBot reference value, and renders the
COUPLED response only, split by parity (2 panels: even, odd), overlaying the
fixed Lambda=0 (uncoupled) reference as a thin gray underlay in the SAME two
panels rather than devoting two more (unchanging) panels to it.

Two-stage design:
  1. PHYSICS: solve at K log-spaced d keyframes (log-spacing because the
     response changes fast just above d=0 and slowly at large d; a linear d
     grid wastes frames on the slow part and starves the fast part). Costs
     ~47s/keyframe (the expensive part is building each d's modal pressure
     map, not the kappa sweep itself).
  2. RENDER: interpolate M frames between each pair of keyframes by linearly
     interpolating the COMPLEX q(kappa) arrays (not magnitude and phase
     separately -- interpolating angle directly produces artifacts at the
     +-180 deg wrap; interpolating the underlying real/imaginary parts does
     not), so playback is smooth without paying for extra physics solves.

Axis/colorbar limits are fixed across all frames (from the strongest-
coupling keyframe) so amplitude changes are real, not axis rescaling.
Resonance vlines are recomputed per rendered frame (interpolated or not) via
automatic peak detection on that frame's own magnitude envelope.

Usage: julia --project=. experiments/render_bare_q_d_sweep_frames.jl
Then:  ffmpeg -framerate 12 -i frame_%03d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=12,format=yuv420p" -c:v libx264 out.mp4
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
const D_REF = Float64(p0.d)
const L_RAFT = Float64(p0.L_raft)
const D_MAX = 3 * D_REF
const D_MIN_LOG = D_MAX * 1e-3   # smallest nonzero d in the log-spaced part
const N_KEYFRAMES = 15           # physics solves: d=0, then N_KEYFRAMES-1 log-spaced up to D_MAX
const N_INTERP = 4               # interpolated frames inserted between each pair of keyframes
const D_KEYFRAMES = vcat([0.0], 10 .^ range(log10(D_MIN_LOG), log10(D_MAX); length=N_KEYFRAMES - 1))

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
EI_scale = Float64(p0.rho_raft) * Float64(p0.L_raft)^4 * Float64(p0.omega)^2

function sweep_complex(solver, ctx)
    nmodes = length(ctx.mode_numbers)
    Q = zeros(ComplexF64, nmodes, length(kappa_grid))
    for (ik, k) in enumerate(kappa_grid)
        Q[:, ik] = solver(k * EI_scale, xM, ctx) ./ L_RAFT
    end
    return Q
end

# --- fixed uncoupled (Lambda=0) reference, computed once ---
ctx0 = theoretical_modal_context_LH(p0; output_dir=joinpath(WC, "output"))
const MODE_NUMBERS = ctx0.mode_numbers
Q_u = sweep_complex(solve_uncoupled, ctx0)
mag_u, phase_u = abs.(Q_u), rad2deg.(angle.(Q_u))

kpole_dry_all = [(m, Float64(p0.rho_raft) * Float64(p0.omega)^2 / (EI_scale * b^4))
                 for (m, b) in zip(ctx0.mode_numbers, ctx0.beta) if b > 0]
kpole_dry_even = [k for (m, k) in kpole_dry_all if iseven(m)]
kpole_dry_odd  = [k for (m, k) in kpole_dry_all if isodd(m)]

const EVEN_MODES = [m for m in MODE_NUMBERS if iseven(m)]
const ODD_MODES  = [m for m in MODE_NUMBERS if isodd(m)]
const EVEN_IDX = [j for (j, m) in enumerate(MODE_NUMBERS) if iseven(m)]
const ODD_IDX  = [j for (j, m) in enumerate(MODE_NUMBERS) if isodd(m)]

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
            if (logenv[i] - background) * 20 >= min_prominence_db
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

function draw_panel!(ax, mag_ref, phase_ref, mag, phase, kpoles, parity_idx)
    # thin gray uncoupled reference underlay (no phase color -- it's fixed context)
    for j in parity_idx
        lines!(ax, kappa_grid, mag_ref[j, :]; color=(:gray70, 0.9), linewidth=1.0)
    end
    for j in parity_idx
        cols = phase_color(phase[j, :])
        lw = 1.0 + 1.1 * j
        lines!(ax, kappa_grid, mag[j, :]; color=cols, linewidth=lw)
    end
    vlines!(ax, kpoles; color=:gray30, linestyle=:dashdot, linewidth=1.2)
    xlims!(ax, extrema(kappa_grid))
end

# --- phase 1: physics at log-spaced keyframes ---
println("Phase 1: solving $(N_KEYFRAMES) keyframes (d=0 + $(N_KEYFRAMES-1) log-spaced up to $D_MAX)...")
Q_keyframes = Vector{Matrix{ComplexF64}}(undef, N_KEYFRAMES)
for (ki, d) in enumerate(D_KEYFRAMES)
    ctx_d = iszero(d) ? ctx0 : theoretical_modal_context_LH(with_d(p0, d); output_dir=joinpath(WC, "output"))
    Q_keyframes[ki] = iszero(d) ? Q_u : sweep_complex((EI, xm, c) -> solve_theoretical_modal_response(EI, xm, c), ctx_d)
    println("  keyframe $ki/$N_KEYFRAMES done (d=$d, Lambda=$(d/L_RAFT))")
end

all_mags = vcat(vec(mag_u), (vec(abs.(Q)) for Q in Q_keyframes)...)
YMIN = 10.0^floor(log10(minimum(filter(x -> x > 0, all_mags))))
YMAX = 10.0^ceil(log10(maximum(all_mags)))
println("Fixed y-limits: ", (YMIN, YMAX))

# --- phase 2: interpolate + render ---
function render_frame(fi, d, Q)
    mag = abs.(Q); phase = rad2deg.(angle.(Q))
    Lambda = d / L_RAFT
    validated_even = auto_resonance_kappas(mag, EVEN_IDX)
    validated_odd  = auto_resonance_kappas(mag, ODD_IDX)

    fig = PaperPlotTheme.with_theme() do
        fig = Figure(size=(620, 820), backgroundcolor=:white, fontsize=18, figure_padding=(6, 8, 4, 4))
        Label(fig[0,1], @sprintf("d = %.5f m   (Λ = %.3f)", d, Lambda); fontsize=20, font=:bold, tellwidth=false)

        axis_kw = (xlabelsize=18, ylabelsize=18, xticklabelsize=16, yticklabelsize=16,
            xminorticksvisible=false, xminorgridvisible=false, yminorgridvisible=false)
        ax_e = Axis(fig[1,1]; xscale=log10, yscale=log10, ylabel=L"|\bar{q}_n|", title="Even modes", axis_kw...)
        ax_o = Axis(fig[2,1]; xscale=log10, yscale=log10, xlabel=L"\kappa", ylabel=L"|\bar{q}_n|", title="Odd modes", axis_kw...)

        draw_panel!(ax_e, mag_u, phase_u, mag, phase, kpole_dry_even, EVEN_IDX)
        draw_panel!(ax_o, mag_u, phase_u, mag, phase, kpole_dry_odd,  ODD_IDX)
        for ax in (ax_e, ax_o)
            xlims!(ax, extrema(kappa_grid))
            ylims!(ax, YMIN, YMAX)
        end
        linkxaxes!(ax_e, ax_o)
        ax_e.xticklabelsvisible = false

        phase_ticks_deg = -180:90:180
        Colorbar(fig[1:2, 2];
            colormap = cgrad([PHASE_CMAP[t] for t in phase_t(collect(-180.0:1.0:180.0))]),
            limits = (-180, 180), ticks = collect(phase_ticks_deg),
            label = "phase (deg)", labelsize=18, ticklabelsize=16)

        legend_elems = Any[]; legend_labels = String[]
        for (j, m) in enumerate(MODE_NUMBERS)
            lw = 1.0 + 1.1 * j
            push!(legend_elems, LineElement(color=:gray20, linewidth=lw))
            push!(legend_labels, "mode $m")
        end
        Legend(fig[3, 1:2], legend_elems, legend_labels; orientation=:horizontal,
            framevisible=false, labelsize=16, patchsize=(24, 10), tellheight=true, nbanks=2)

        colgap!(fig.layout, 6)
        rowgap!(fig.layout, 4)
        resize_to_layout!(fig)
        fig
    end

    out = joinpath(FRAME_DIR, @sprintf("frame_%03d.png", fi))
    save(out, fig; px_per_unit=6)
    return out
end

println("Phase 2: rendering $(N_KEYFRAMES + (N_KEYFRAMES-1)*N_INTERP) frames (keyframes + interpolated)...")
fi = 0
for ki in 1:N_KEYFRAMES
    global fi += 1
    out = render_frame(fi, D_KEYFRAMES[ki], Q_keyframes[ki])
    println("  Saved $out  (keyframe $ki/$N_KEYFRAMES)")
    ki == N_KEYFRAMES && break
    d0, d1 = D_KEYFRAMES[ki], D_KEYFRAMES[ki+1]
    Q0, Q1 = Q_keyframes[ki], Q_keyframes[ki+1]
    for s in 1:N_INTERP
        t = s / (N_INTERP + 1)
        d_t = (1 - t) * d0 + t * d1
        Q_t = (1 - t) .* Q0 .+ t .* Q1
        global fi += 1
        out = render_frame(fi, d_t, Q_t)
        println("  Saved $out  (interp $s/$N_INTERP between keyframes $ki,$(ki+1))")
    end
end
println("Done: $fi frames in $FRAME_DIR")
