using Test
using Surferbot

# The geometric vertical grid must reproduce the uniform grid at equal depth. Same x grid,
# same surface spacing; only the vertical node distribution differs.

const _GZ_EI0 = 3.0e9 * 3e-2 * 9e-4^3 / 12

@testset "Graded z grid reproduces the uniform grid" begin
    for (label, eif) in (("rigid", Inf), ("flexible EIx1", 1.0))
        p0 = FlexibleParams(motor_position = -0.012, EI = _GZ_EI0 * eif, nu = 0.0)
        dd = Surferbot.derive_params(p0)
        n = round(Int, dd.n * 0.5); n += iseven(n) ? 1 : 0
        Nz1 = round(Int, dd.Nz * 0.5)
        for df in (2.0, 8.0)
            H = df * dd.domain_depth
            Nzu = round(Int, Nz1 * df)
            ru = solve_cached(FlexibleParams(motor_position = -0.012, EI = _GZ_EI0 * eif, nu = 0.0,
                                             n = n, Nz = Nzu, domain_depth = H))
            rg = solve_cached(FlexibleParams(motor_position = -0.012, EI = _GZ_EI0 * eif, nu = 0.0,
                                             n = n, domain_depth = H,
                                             dz_surface = dd.domain_depth / (Nz1 - 1), z_growth = 1.05))
            err = abs(rg.thrust - ru.thrust) / abs(ru.thrust)
            @info "graded vs uniform [$label, depth x$df]: Nz $(rg.metadata.args.Nz) vs $Nzu, rel thrust diff $err"
            @test err < 1e-3
            GC.gc()
        end
    end
end
