# Wave-driven propulsion of a flexible raft

Numerical companion to the paper *"Wave-driven propulsion of a flexible raft"*
(the "Surferbot" problem: a small vibrating raft propelled by the waves it
generates). This repo builds the free-surface solver, runs the parameter
sweeps, and produces every figure in the manuscript.

The active codebase is **`Julia/`**. Everything else is reference material:

| Path | What it is |
|---|---|
| `Julia/` | The solver, sweeps, and figure scripts. Start here. |
| `docs/surferbot_paper_draft.tex` | Symlink to the paper's `main.tex`. This is the physics ground truth — if Julia code and paper disagree, the paper wins unless a mistake in the paper is found and confirmed with a human. |
| `MATLAB/old_code` | The original MATLAB implementation this repo achieves numerical parity with, following Benham et al. (2024, JFM), *"On-wave driven propulsion"*. |
| `python/` | An earlier JAX-based prototype (DtN operators, rigid-raft solver). Superseded by `Julia/`; kept for reference, not under active development. |

## Setup

Requires Julia 1.10+.

```bash
cd Julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Plotting scripts (`scripts/plot_*.jl`) use `CairoMakie`, which lives in a
**separate** environment so the test suite doesn't pay for its precompile
time:

```bash
cd Julia/scripts
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## The solver

`Julia/src/Surferbot.jl` assembles and solves the coupled flexible-beam /
free-surface system. A minimal run looks like:

```julia
using Surferbot

params = FlexibleParams(
    L_raft = 0.1, motor_position = 0.024, EI = 3e-5,
    omega = 2π * 10, rho_raft = 0.05,
)
system   = assemble_flexible_system(params)
solution = solve_tensor_system(system.A, system.b)
```

`FlexibleParams` holds every physical input (raft length, motor position,
flexural rigidity, forcing frequency, surface tension, domain size, grid
resolution, ...); see the docstring in `Surferbot.jl` for the full field
list. `assemble_flexible_system` builds the linear system from those
parameters; `solve_tensor_system` solves it (and transparently handles
`ForwardDiff.Dual`-valued sparse systems for AD-based sensitivity work).

## Pipeline: sweep → postprocess → plot

Three kinds of scripts live in `Julia/`, and the naming tells you which is
which:

- **`scripts/sweep_*.jl`** — runs the solver over a parameter grid and
  writes raw output to `output/csv/` or `output/jld2/`. These are the
  expensive step; some are meant to run on a cluster (see the matching
  `.slurm` file).
- **`scripts/postprocess_*.jl`** — derives summary quantities (thrust,
  asymmetry, modal decomposition, ...) from sweep output.
- **`scripts/plot_*.jl`** — reads pre-computed CSV/JLD2 files and produces a
  figure. Plot scripts never run their own sweeps — if a figure is missing
  its input data, run the corresponding `sweep_*`/`postprocess_*` script
  first. A plot script's output figure shares its base name (e.g.
  `plot_kappa_snapshot.jl` → `output/figures/plot_kappa_snapshot_*.pdf`).

Run any of them the same way, e.g.:

```bash
julia --project=scripts scripts/plot_kappa_snapshot.jl --paper-snapshots
```

`Julia/experiments/` is the sandbox: debugging scripts and one-off checks
that aren't expected to keep working or producing lasting output.

## Tests

```bash
cd Julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

CI (`.github/workflows/julia-tests.yml`) runs the same command on every push
to `main`. Tests cover the solver assembly, modal decomposition, root
tracing for the resonance diagnostics, MATLAB-parity checks, and a handful
of physics invariants (symmetric forcing → zero net thrust, edge boundary
conditions, etc.).

## Working on the physics

See `CLAUDE.md` for the ground-truth rules this repo follows: the paper
draft is canonical for the Julia implementation, Benham et al. (2024) is
canonical for the MATLAB reference, and MATLAB/Julia parity is judged by the
L2 norm of the free-surface elevation η — not by whether a scalar diagnostic
like thrust happens to match.
