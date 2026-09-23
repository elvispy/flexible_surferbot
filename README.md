# Wave-driven propulsion of a flexible raft

Numerical companion to *Wave-driven propulsion of a flexible raft*. A raft floating at a
water surface is driven by an onboard motor and propelled by the gravity–capillary waves
it radiates. This repository holds the free-surface solver, the parameter sweeps, and the
scripts that draw every figure in the paper.

![Surferbot simulation](assets/surferbot_demo.gif)

*A raft of two materials joined at its midpoint, the stiffer half in black and the
compliant half in grey, driven at `x_M/L = 0.26`. The waves leaving to the right are
larger than those leaving to the left, and that imbalance is the thrust. Produced by
`flexible_solver` and `render_surferbot_run`.*

| Path | Contents |
|---|---|
| `Julia/` | Solver, sweeps, figure scripts. The active code. |
| `MATLAB/old_code/` | Original implementation, following Benham, Devauchelle & Thomson (2024, *JFM* **987**, A44), *On wave-driven propulsion*. `Julia/` targets numerical parity with it. |
| `python/` | Earlier JAX prototype: DtN operators and a rigid-raft solver. Superseded by `Julia/`. |
| `docs/surferbot_paper_draft.tex` | Symlink to the paper's `main.tex`, the physics ground truth for `Julia/`. |

## Setup

Julia 1.10 or later. Two environments: the package itself, and a separate one for the
CairoMakie plotting scripts, kept apart so the test suite does not pay Makie's
precompilation cost.

```bash
cd Julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
cd scripts && julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## A first run

```julia
using Surferbot

L = 0.1
base   = FlexibleParams(L_raft = L, motor_position = -0.3L, EI = 3e-5,
                        omega = 2π * 10, rho_raft = 0.05, forcing_width = 0.01/3)
nb     = Surferbot.derive_params(base).nb_contact           # raft nodes, set by the wavelength
EI     = vcat(fill(3e-5, nb ÷ 2), fill(3e-4, nb - nb ÷ 2))  # two materials, joined at the midpoint

params = FlexibleParams(L_raft = L, motor_position = -0.3L, EI = EI,
                        omega = 2π * 10, rho_raft = 0.05, forcing_width = 0.01/3)
result = flexible_solver(params)

render_surferbot_run(result; outdir = "output", basename = "run", duration_periods = 2)
```

`EI` takes a scalar or, as above, a per-node vector for a graded or multi-material beam.
`flexible_solver` assembles and solves the coupled beam and free-surface system and
returns a `FlexibleResult`. `render_surferbot_run` turns that into an MP4 with a JSON
provenance sidecar, which is how the animation above was made.

## Sweeps and figures

Three script prefixes, one pipeline:

- `scripts/sweep_*.jl` runs the solver over a parameter grid and writes raw results to
  `output/csv/` or `output/jld2/`. Some are meant for a cluster; see the matching
  `.slurm` file.
- `scripts/postprocess_*.jl` derives thrust, wake asymmetry and modal amplitudes from
  sweep output.
- `scripts/plot_*.jl` reads those files and draws one figure. It never runs its own
  sweep, so a missing figure means the matching sweep has not been run yet. Output
  shares the script's base name: `plot_kappa_snapshot.jl` writes
  `output/figures/plot_kappa_snapshot_*.pdf`.

```bash
julia --project=scripts scripts/plot_kappa_snapshot.jl --paper-snapshots
```

`Julia/experiments/` holds debugging scripts with no guarantee of lasting output.

## Tests

```bash
cd Julia && julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the same command on every push to `main` (`.github/workflows/julia-tests.yml`).
The suite covers system assembly, modal decomposition, resonance root tracing, parity
with the MATLAB code, and physics invariants such as zero net thrust under symmetric
forcing.

`test/test_paper_derivations.jl` differs in kind from the rest: it re-derives the paper's
analytical results in a computer algebra system and compares them with what the paper
prints, each structural claim paired with a negative control that has to fail. It reads
as a document, and `output/pdf/paper_derivations.pdf` is that document rendered;
`scripts/render_paper_derivations.jl` rebuilds it.

## API

| Function | Signature | Use |
|---|---|---|
| `FlexibleParams` | `FlexibleParams(; kwargs...)` | Define a simulation: raft length, motor position, flexural rigidity (scalar or per-node vector), forcing frequency, fluid properties, domain size, grid resolution. |
| `flexible_solver` | `flexible_solver(params) -> FlexibleResult` | Solve the coupled beam and free-surface system for one parameter set. |
| `derive_params` | `derive_params(params) -> NamedTuple` | Non-dimensional groups and the grid choices the solver makes from `params`. |
| `render_surferbot_run` | `render_surferbot_run(result; outdir, basename, fps, duration_periods)` | Render a result as an MP4 with a JSON provenance sidecar. |
| `beam_edge_metrics`, `beam_asymmetry` | `beam_edge_metrics(result)`, `beam_asymmetry(eta_left, eta_right)` | Surface elevation at each raft and domain edge, and the left–right asymmetry built from it. |
| `sweep_parameters` | `sweep_parameters(base, grid; solver, beam_metrics_fn, save_path) -> SweepArtifact` | Run `solver` over every combination of a NamedTuple grid. |
| `save_sweep`, `load_sweep` | `save_sweep(path, artifact)`, `load_sweep(path)` | Persist or reload a sweep as JLD2. |

A `FlexibleResult` carries `U`, `power` and `thrust`, the grid `x` and `z`, the complex
fields `phi`, `phi_z`, `eta` and `pressure`, and `max_curvature` and `wave_steepness`.
The paper holds the raft against horizontal translation, so `thrust` is the quantity
reported there; `U` is the drift speed that thrust would produce under the drag law in
`postprocess.jl`.
