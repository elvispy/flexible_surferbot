# Surferbot.jl

Finite-difference solver for a flexible raft at a free surface, driven at a single
frequency. Solves the coupled Euler–Bernoulli beam and weakly viscous quasi-potential
fluid system, and returns the wave field, the raft shape and the thrust.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'          # the package
cd scripts && julia --project=. -e 'using Pkg; Pkg.instantiate()'   # the plotting scripts
```

| Path | Contents |
|---|---|
| `src/` | The package. `Surferbot.jl` holds the types and the solver; `fd.jl` the finite-difference operators; `modal.jl` the modal reduction; `postprocess.jl` thrust and radiation stress; `video.jl` the renderer. |
| `scripts/` | Sweeps, post-processing and figure scripts. Separate environment, because they pull in CairoMakie. |
| `test/` | The suite CI runs. |
| `experiments/` | Scratch work. No guarantee that anything here keeps working. |
| `output/` | Generated figures, sweep data, and the rendered derivations document. |

## One solve

```julia
using Surferbot

params = FlexibleParams(L_raft = 0.05, d = 0.03, rho_raft = 0.052,
                        motor_position = -0.006, omega = 2π * 80,
                        EI = 1e-4, forcing_width = 0.0025)
result = flexible_solver(params)

result.thrust        # mean horizontal thrust on the raft, N
result.eta           # complex surface elevation on the grid
beam_asymmetry(beam_edge_metrics(result)...)   # left-right radiation imbalance
```

`FlexibleParams` carries the physical inputs and the grid controls; `derive_params`
reports what the solver makes of them, including the non-dimensional groups and the
grid size it chose from the wavelength.

```julia
d = derive_params(params)
d.nd_groups.kappa, d.nd_groups.Lambda, d.Nx, d.Nz
```

## A graded or multi-material raft

`EI` accepts a vector with one entry per raft node, so stiffness can vary along the
body. `Inf` marks a rigid section.

```julia
nb = derive_params(params).nb_contact
EI = vcat(fill(1e-6, nb ÷ 5), fill(Inf, nb - nb ÷ 5))   # compliant nose, rigid tail
result = flexible_solver(FlexibleParams(params; EI = EI))
```

## A sweep

```julia
grid = (motor_position = range(-0.025, 0.0; length = 40),
        EI             = 10 .^ range(-7, -3; length = 40))
artifact = sweep_parameters(params, grid; save_path = "output/jld2/sweep.jld2")
```

`sweep_parameters` runs the solver over every combination and returns a
`SweepArtifact`; `save_sweep` and `load_sweep` persist it as JLD2. The `scripts/plot_*`
scripts read those files and never run their own sweep.

## The modal reduction

Instead of one solve per parameter, prescribe each free–free mode in turn, build the
hydrodynamic impedance and capillary endpoint matrices, and afterwards recover any
stiffness or forcing from an `N × N` solve.

```julia
basis = build_raw_freefree_basis(nb, L_raft)
modal = decompose_raft_freefree_modes(result, basis)
modal.q              # modal amplitudes
```

`freefree_betaL_roots` and `freefree_mode_shape` give the eigenvalues and shapes on
their own. `capillary_endpoint_map` assembles the edge-force matrix.

## Rendering

```julia
render_surferbot_run(result; outdir = "output", basename = "run",
                     duration_periods = 2, nondim = true)
```

Writes an MP4 and a JSON sidecar recording the parameters, the git commit and the
script that produced it. `bare = true` drops the axes, labels and margins, and
`figsize`, `figdpi`, `background` and `depth` control the canvas, for figure or
cover artwork. `depth` extends the fluid below the surface so a frame can be made
taller without changing the vertical exaggeration.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/test_paper_derivations.jl` is the odd one out: it re-derives the paper's
analytical results in a computer algebra system rather than exercising the solver.
`scripts/render_paper_derivations.jl` renders it as `output/pdf/paper_derivations.pdf`.
