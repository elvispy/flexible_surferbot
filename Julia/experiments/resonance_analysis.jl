using Surferbot, CSV, DataFrames, Statistics, LinearAlgebra, Printf
include(joinpath(@__DIR__, "../scripts/plot_dimensionless_diagnostics_LH.jl"))
output_dir = joinpath(@__DIR__, "../output")
shift = log10(0.052 * 0.05^4 * (2π*80)^2)
xM_grid = collect(range(0.0, 0.49; length=401))

art_cpl  = load_sweep(joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2"))
ctx_LH   = theoretical_modal_context_LH(art_cpl.base_params; output_dir=output_dir)
ctx_beam = theoretical_modal_context(art_cpl.base_params; output_dir=output_dir)

# --- CSV-based resonance indicator -------------------------------------------
# For each logK column in the full-solve CSV, compute fraction of xM rows where
# |alpha_LH| < 0.1.  This is the ground truth: if the fraction is high, the
# full solver sees a near-resonance there.

df = CSV.read(joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv"), DataFrame)
all_logEI = sort(unique(df.log10_EI))

println("=== CSV ground truth: fraction of xM with |alpha_LH| < 0.1 at each logK ===")
println("(only columns with frac > 0.5 shown)")
for le in all_logEI
    rows = df[df.log10_EI .== le, :]
    alpha = @. -(rows.eta_1_domain_re^2 + rows.eta_1_domain_im^2 -
                 rows.eta_end_domain_re^2 - rows.eta_end_domain_im^2) /
               (rows.eta_1_domain_re^2 + rows.eta_1_domain_im^2 +
                rows.eta_end_domain_re^2 + rows.eta_end_domain_im^2)
    frac = mean(abs.(alpha) .< 0.1)
    frac > 0.5 || continue
    @printf("  logK=%7.3f  frac_small_alpha=%.3f\n", le - shift, frac)
end

# --- Theoretical detection at various thresholds vs CSV ground truth ----------
println("\n=== Theory vs CSV: LH resonances (quantile of |alpha_theory| < 0.04) ===")
scatter_logK = collect(range(-4.1, 0.586; length=57))
EI_list = 10 .^ (scatter_logK .+ shift)

println("  logK    q10_LH  q10_beam  q15_LH  q15_beam  q20_beam  oddLH  oddbeam")
for EI in EI_list
    absS_LH=Float64[]; absA_LH=Float64[]; abs1_LH=Float64[]; abse_LH=Float64[]
    absS_bm=Float64[]; absA_bm=Float64[]; abs1_bm=Float64[]; abse_bm=Float64[]
    for xM in xM_grid
        qL = solve_theoretical_modal_response(EI, xM, ctx_LH)
        dL = theoretical_endpoint_diagnostics_LH(qL, ctx_LH)
        push!(absS_LH,abs(dL.S)); push!(absA_LH,abs(dL.A))
        push!(abs1_LH,abs(dL.eta_LH_1)); push!(abse_LH,abs(dL.eta_LH_end))

        qB = solve_theoretical_modal_response(EI, xM, ctx_beam)
        dB = theoretical_endpoint_diagnostics_beam(qB, ctx_beam)
        push!(absS_bm,abs(dB.S)); push!(absA_bm,abs(dB.A))
        push!(abs1_bm,abs(dB.eta_beam_1)); push!(abse_bm,abs(dB.eta_beam_end))
    end
    aL = abs.(@. -(abs1_LH^2 - abse_LH^2)/(abs1_LH^2 + abse_LH^2 + eps()))
    aB = abs.(@. -(abs1_bm^2 - abse_bm^2)/(abs1_bm^2 + abse_bm^2 + eps()))
    q10L = quantile(aL,0.10); q15L = quantile(aL,0.15)
    q10B = quantile(aB,0.10); q15B = quantile(aB,0.15); q20B = quantile(aB,0.20)
    logK = log10(EI) - shift
    (q10L < 0.04 || q10B < 0.04) || continue  # only show firing rows
    logK < -4.0 && continue  # only visible range
    oddL = mean(absS_LH) < mean(absA_LH)
    oddB = mean(absS_bm) < mean(absA_bm)
    fireL10 = q10L<0.04; fireL15 = q15L<0.04
    fireB10 = q10B<0.04; fireB15 = q15B<0.04; fireB20 = q20B<0.04
    yL10 = fireL10 ? "Y" : "n"; yL15 = fireL15 ? "Y" : "n"
    yB10 = fireB10 ? "Y" : "n"; yB15 = fireB15 ? "Y" : "n"; yB20 = fireB20 ? "Y" : "n"
    yoL = oddL ? "Y" : "n"; yoB = oddB ? "Y" : "n"
    @printf("  %6.3f  LH:%s→%s  bm:%s→%s→%s  oddL=%s oddB=%s\n",
        logK, yL10, yL15, yB10, yB15, yB20, yoL, yoB)
end
