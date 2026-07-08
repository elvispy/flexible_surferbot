using Test
using Surferbot

@testset "root tracing utilities" begin
    @testset "find_filtered_minima: sub-grid interpolation" begin
        # |x - x0|^2 = (x - x0)^2 is exactly parabolic, so the interpolated root
        # should recover x0 to floating-point precision even though x0 does not
        # sit on the grid — this is the regression check for the staircase bug
        # where the old implementation just snapped to the nearest grid point.
        xgrid = collect(range(0.0, 1.0; length=21))  # spacing 0.05
        x0 = 0.5327
        values = abs.(xgrid .- x0)
        ratio = zeros(length(xgrid))  # always passes ratio_cutoff
        roots = Surferbot.find_filtered_minima(xgrid, values, ratio; ratio_cutoff=1.0)
        @test length(roots) == 1
        @test roots[1] ≈ x0 atol = 1e-10

        # A root that DOES sit exactly on a grid point should still be recovered
        # exactly (denom==0 fallback path).
        xgrid2 = collect(range(0.0, 1.0; length=11))
        x0_exact = xgrid2[6]
        values2 = abs.(xgrid2 .- x0_exact)
        roots2 = Surferbot.find_filtered_minima(xgrid2, values2, zeros(length(xgrid2)); ratio_cutoff=1.0)
        @test length(roots2) == 1
        @test roots2[1] ≈ x0_exact atol = 1e-10

        # Regression check: on a DENSE grid (small spacing), the interpolated
        # root for two nearby-but-distinct minima must not collapse onto the
        # same grid-quantized value (the original bug's symptom — a "staircase"
        # of repeated values across consecutive samples of a slowly-moving root).
        xgrid3 = collect(range(0.0, 1.0; length=2001))  # spacing 5e-4
        for x0_test in (0.30001, 0.30003, 0.30005)
            v = abs.(xgrid3 .- x0_test)
            r = Surferbot.find_filtered_minima(xgrid3, v, zeros(length(xgrid3)); ratio_cutoff=1.0)
            @test only(r) ≈ x0_test atol = 1e-9
        end

        # ratio_cutoff filtering still works: a minimum that fails the ratio
        # test must not be reported.
        ratio_fail = fill(1.0, length(xgrid))
        roots_none = Surferbot.find_filtered_minima(xgrid, values, ratio_fail; ratio_cutoff=0.5)
        @test isempty(roots_none)
    end

    @testset "dedup_resonance_runs" begin
        @test isempty(Surferbot.dedup_resonance_runs(Tuple{Int,Float64}[]))

        # Single candidate: returned unchanged.
        single = Surferbot.dedup_resonance_runs([(5, 0.3)])
        @test single == [(5, 0.3)]

        # One run of consecutive indices (a single physical resonance sampled by
        # several adjacent grid columns) collapses to the minimum-score index —
        # this is the regression check for the duplicate-vertical-stripe bug.
        run_candidates = [(10, 0.05), (11, 0.01), (12, 0.04)]
        deduped = Surferbot.dedup_resonance_runs(run_candidates)
        @test length(deduped) == 1
        @test deduped[1] == (11, 0.01)

        # Two separate runs (indices with a gap > 1 between them) must each
        # produce their own representative, not merge into one.
        two_runs = [(1, 0.02), (2, 0.03), (50, 0.09), (51, 0.01), (52, 0.08)]
        deduped2 = Surferbot.dedup_resonance_runs(two_runs)
        @test length(deduped2) == 2
        @test (1, 0.02) in deduped2
        @test (51, 0.01) in deduped2

        # Unsorted input is handled the same as sorted input.
        shuffled = [(52, 0.08), (1, 0.02), (51, 0.01), (2, 0.03), (50, 0.09)]
        @test Surferbot.dedup_resonance_runs(shuffled) == deduped2
    end

    @testset "cluster_branches" begin
        # A genuine crossing root that happens to land at the SAME logK as an
        # injected resonance stripe must survive as part of the smooth branch,
        # not be swallowed by the stripe-exclusion filter. Build a smooth branch
        # 0.10, 0.12, ..., 0.20 across 6 logK columns, with a resonance stripe
        # (20 points, the default resonance_n_pts) ALSO injected at the middle
        # logK value (mirroring how a resonance column doubles as both a
        # detected resonance and an ordinary fine-grid root in production).
        lks = collect(0.0:1.0:5.0)
        xms = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        stripe_lk = lks[3]  # coincides with the genuine root at xms[3] = 0.14
        stripe_xm = collect(range(0.0, 0.5; length=20))
        logK_pts = vcat(lks, fill(stripe_lk, 20))
        xM_pts   = vcat(xms, stripe_xm)

        lk_out, xm_out, res_lks = Surferbot.cluster_branches(logK_pts, xM_pts)
        @test res_lks == [stripe_lk]
        # the genuine point at stripe_lk must still be present, in order, with
        # its real value — not dropped, and not replaced by NaN.
        idx = findfirst(==(stripe_lk), lk_out)
        @test idx !== nothing
        @test xm_out[idx] ≈ 0.14
        # all 6 points connect into one smooth branch (consecutive xm's are
        # within the 0.25 connection threshold), so there's no internal NaN gap
        # — only the single trailing separator.
        @test count(isnan, xm_out) == 1
        @test xm_out[1:end-1] ≈ xms
        @test isnan(xm_out[end])

        # Global-distance greedy matching: three branches at a starting column
        # (0.0006, 0.211, 0.395) and two targets at the next column (0.207,
        # 0.394). Order-dependent greedy (ascending xm) would let 0.0006 grab
        # 0.207 (distance 0.2064) before 0.211 gets a chance, cascading into
        # a wrong 0.211->0.394 match. The correct pairing is 0.211->0.207
        # (distance 0.004) and 0.395->0.394 (distance 0.001), leaving 0.0006
        # unmatched (isolated, dropped as a single-point branch).
        logK_pts2 = [0.0, 0.0, 0.0, 1.0, 1.0]
        xM_pts2   = [0.0006, 0.211, 0.395, 0.207, 0.394]
        lk_out2, xm_out2, res_lks2 = Surferbot.cluster_branches(logK_pts2, xM_pts2)
        @test isempty(res_lks2)
        # Reconstruct branches by splitting on NaN separators.
        branches = Vector{Float64}[]
        cur = Float64[]
        for x in xm_out2
            if isnan(x)
                isempty(cur) || push!(branches, cur)
                cur = Float64[]
            else
                push!(cur, x)
            end
        end
        # The unmatched 0.0006 branch survives as a length-1 segment (a lone
        # point flanked by NaNs draws no visible line — that's what "dropped"
        # means here), while the other two are correctly, tightly paired.
        @test length(branches) == 3
        singleton = only(filter(b -> length(b) == 1, branches))
        @test singleton ≈ [0.0006] atol=1e-6
        paired = sort(filter(b -> length(b) == 2, branches); by = first)
        @test length(paired) == 2
        @test paired[1] ≈ [0.211, 0.207] atol=1e-8
        @test paired[2] ≈ [0.395, 0.394] atol=1e-8
    end
end
