using Surferbot

# Heavy physics tests share solves: a case defined by identical parameters is solved once
# per test process and reused by every file that asks for it.
const _SOLVE_CACHE = Dict{Any,Any}()
solve_cached(p) = get!(() -> flexible_solver(p), _SOLVE_CACHE, p)
