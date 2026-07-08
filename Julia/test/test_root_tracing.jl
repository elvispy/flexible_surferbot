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
end
