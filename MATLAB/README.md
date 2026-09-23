# MATLAB

The original implementation, following Benham, Devauchelle & Thomson (2024, *J. Fluid
Mech.* **987**, A44), *On wave-driven propulsion*. It solves the rigid-raft problem.

`Julia/` supersedes it and is where new work belongs. This directory is kept because
the Julia solver is tested against it: `Julia/test/test_matlab_*.jl` compare the
finite-difference weights, the assembled system and the solution node by node, and
`surferbot_results/` holds the reference output those tests read.

| Path | Contents |
|---|---|
| `src/` | The solver. |
| `old_code/` | The version the parity tests target. |
| `utils/` | Finite-difference and quadrature helpers. |
| `test/` | MATLAB-side checks. |
| `surferbot_results/` | Reference output consumed by the Julia parity tests. |

Changing anything here will break those tests unless the reference output is
regenerated alongside it.
