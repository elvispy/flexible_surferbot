using Surferbot, JLD2, Printf, Statistics

const SCRIPT_DIR  = @__DIR__
const output_dir  = joinpath(SCRIPT_DIR, "..", "output")

include(joinpath(SCRIPT_DIR, "..", "scripts", "plot_dimensionless_diagnostics_LH.jl"))

art_cpl  = Surferbot.Sweep.load_sweep(joinpath(output_dir, "jld2",
               "sweep_motor_position_EI_coupled_from_matlab.jld2"))
art_ucpl = Surferbot.Sweep.load_sweep(joinpath(output_dir, "jld2",
               "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))

EI_scale_cpl  = Float64(art_cpl.base_params.rho_raft) *
                Float64(art_cpl.base_params.L_raft)^4  *
                Float64(art_cpl.base_params.omega)^2
EI_scale_ucpl = Float64(art_ucpl.base_params.rho_raft) *
                Float64(art_ucpl.base_params.L_raft)^4  *
                Float64(art_ucpl.base_params.omega)^2

ctx_LH_cpl    = theoretical_modal_context_LH(art_cpl.base_params;  output_dir=output_dir)
ctx_beam_cpl  = theoretical_modal_context(art_cpl.base_params;     output_dir=output_dir)
ctx_beam_ucpl = theoretical_modal_context(art_ucpl.base_params;    output_dir=output_dir)

const XM_GRID = collect(range(0.0, 0.5; length=601))
const N_GRID  = length(XM_GRID)

# ── Branch tracker ────────────────────────────────────────────────────────────
const BRANCH_TOL  = 0.10
const TERM_MIN_XM = 0.35
const TERM_TOL    = 0.08

function track_branches(pts_logEI, pts_xM, logEI_axis)
    slices = [Float64[] for _ in logEI_axis]
    for (lk, xM) in zip(pts_logEI, pts_xM)
        idx = argmin(abs.(logEI_axis .- lk))
        push!(slices[idx], xM)
    end
    for s in slices; sort!(s); end

    next_id = Ref(0)
    active  = Dict{Int, Vector{Tuple{Float64,Float64}}}()
    closed  = Vector{Vector{Tuple{Float64,Float64}}}()

    for (i, lk) in enumerate(logEI_axis)
        roots = slices[i]; isempty(roots) && continue
        root_to_id = fill(-1, length(roots)); used = Set{Int}()
        for (ri, xM) in enumerate(roots)
            bd = Inf; bi = -1
            for (id, br) in active
                id in used && continue
                d = abs(xM - br[end][2])
                if d < BRANCH_TOL && d < bd; bd = d; bi = id; end
            end
            if bi > 0; root_to_id[ri] = bi; push!(used, bi); end
        end
        for id in collect(keys(active))
            id in used || (push!(closed, active[id]); delete!(active, id))
        end
        for (ri, xM) in enumerate(roots)
            if root_to_id[ri] > 0
                push!(active[root_to_id[ri]], (lk, xM))
            else
                next_id[] += 1; active[next_id[]] = [(lk, xM)]
            end
        end
    end
    append!(closed, values(active))
    return closed
end

function branch_slope(branch)
    length(branch) < 2 && return 0.0
    xs = [p[1] for p in branch]; ys = [p[2] for p in branch]
    mx = mean(xs); my = mean(ys)
    sum((xs .- mx) .* (ys .- my)) / max(sum((xs .- mx).^2), eps())
end

# ── Compute roots without resonance stripes ───────────────────────────────────
# Uses the same interior + terminus logic as the actual script but skips resonance
# detection, so the branch tracker sees clean data.

function diag_LH(q, ctx)
    d = theoretical_endpoint_diagnostics_LH(q, ctx)
    return abs(d.S), abs(d.A), abs(d.eta_LH_1), abs(d.eta_LH_end)
end

function diag_beam(q, ctx)
    d = theoretical_endpoint_diagnostics_beam(q, ctx)
    return abs(d.S), abs(d.A), abs(d.eta_beam_1), abs(d.eta_beam_end)
end

function compute_roots_clean(EI_list, cname, ctx, diag_fn)
    logEI_axis = log10.(EI_list)
    pts_lEI    = Float64[]
    pts_xM     = Float64[]
    prev_hi    = Float64[]

    for (iei, EI) in enumerate(EI_list)
        absS=Float64[]; absA=Float64[]; ae1=Float64[]; aee=Float64[]
        for xM in XM_GRID
            s, a, e1, ee = diag_fn(solve_theoretical_modal_response(EI, xM, ctx), ctx)
            push!(absS,s); push!(absA,a); push!(ae1,e1); push!(aee,ee)
        end
        roots = roots_for_condition(cname, XM_GRID, absS, absA, ae1, aee)

        # Terminus detection
        if !isempty(prev_hi)
            vn, rn = _cond_val_ratio(cname, N_GRID,   absS, absA, ae1, aee)
            vn1, _ = _cond_val_ratio(cname, N_GRID-1, absS, absA, ae1, aee)
            if vn <= vn1 && rn < RATIO_CUTOFF
                for pxM in prev_hi
                    if !any(r -> abs(r - pxM) < TERM_TOL, roots)
                        push!(pts_lEI, logEI_axis[iei])
                        push!(pts_xM,  0.5)
                        break
                    end
                end
            end
        end
        prev_hi = filter(r -> r > TERM_MIN_XM, roots)

        for r in roots
            push!(pts_lEI, logEI_axis[iei])
            push!(pts_xM,  r)
        end
    end
    return pts_lEI, pts_xM, log10.(EI_list)
end

# ── Run analysis for one figure ───────────────────────────────────────────────
function run_analysis(label, EI_list, EI_scale, ctx, diag_fn)
    logEI_axis = log10.(EI_list)
    logK_axis  = logEI_axis .- log10(EI_scale)

    println("\n" * "="^65)
    println(label)
    println("="^65)

    grand_up = 0; grand_05 = 0

    for cname in ["S", "A", "eta_1", "eta_end"]
        pts_lEI, pts_xM, _ = compute_roots_clean(EI_list, cname, ctx, diag_fn)
        branches = track_branches(pts_lEI, pts_xM, logEI_axis)

        up = filter(b -> branch_slope(b) > 0 &&
                         (length(b) >= 2 || b[end][2] >= 0.499), branches)
        sort!(up; by = b -> b[end][2])

        reaches = count(b -> b[end][2] >= 0.499, up)
        early   = length(up) - reaches
        grand_up += length(up); grand_05 += reaches

        println("\n  |$cname|:  $(length(up)) upward branches")
        println("    #   slope    logκ_start  logκ_end   xM_start  xM_end   terminal")
        for (bi, br) in enumerate(up)
            sl  = branch_slope(br)
            lk0 = br[1][1]   - log10(EI_scale)
            lk1 = br[end][1] - log10(EI_scale)
            x0  = br[1][2];   x1 = br[end][2]
            at  = x1 >= 0.499
            @printf "    %2d  %+.3f   %+.2f        %+.2f       %.3f      %.3f    %s\n" bi sl lk0 lk1 x0 x1 (at ? "→0.5 ✓" : "STOPS EARLY")
        end
        println("    SUMMARY: $reaches reach xM=0.5,  $early stop early")
    end
    println("\n  Grand total: $grand_up upward → $grand_05 reach 0.5, $(grand_up-grand_05) stop early")
end

EI_list_cpl  = collect(Float64.(art_cpl.parameter_axes.EI))
EI_list_ucpl = collect(Float64.(art_ucpl.parameter_axes.EI))

run_analysis("LH (domain-end) COUPLED",
             EI_list_cpl, EI_scale_cpl, ctx_LH_cpl, diag_LH)

run_analysis("BEAM-END COUPLED",
             EI_list_cpl, EI_scale_cpl, ctx_beam_cpl, diag_beam)

run_analysis("BEAM-END UNCOUPLED",
             EI_list_ucpl, EI_scale_ucpl, ctx_beam_ucpl, diag_beam)
