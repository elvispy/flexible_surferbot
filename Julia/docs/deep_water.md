# Deep-water criterion for the raft problem

Linear potential flow in a layer `-H < z < 0` with a rigid bottom has the free-surface
Dirichlet-to-Neumann symbol

    phi_z = N_H(q) phi,    N_H(q) = q tanh(qH) = |q| (1 - 2 / (exp(2|q|H) + 1)).

Depth enters the problem only through `dN = N_H - |q| = -2|q| / (exp(2|q|H) + 1)`.

## Raft scale (impedance)

At the raft's scale (`q ~ 1/L`) the free surface responds with `phi_z / phi ~ omega^2/g`, far above
`q`, so outside the raft `phi ~ 0` (pressure release). For a heaving strip of half-width `a = L/2`
the deep-water solution is `phi_s = sqrt(a^2 - x^2)`: it satisfies `(-d_xx)^(1/2) phi_s = 1` on the
strip (using `int_0^inf J_1 = 1`) and has added mass `int phi_s dx = pi a^2 / 2`. The operator is
symmetric, so the first-order change of the added mass is

    eps_Z(H) = (1/m_inf) int dq/(2 pi) |phi_hat|^2 * 2|q| / (exp(2|q|H) + 1),
    phi_hat(q) = pi a J_1(q a) / q,

and for `H >> a`

    eps_Z = (pi^2 / 48) (a/H)^2          (int_0^inf u/(e^u+1) du = pi^2/12).

The same law with `H` replaced by the half-width `X = L_domain/2` describes the truncation by the
lateral boundaries (observed, not derived). Solving `eps_Z = eps`:

    H >= pi L / sqrt(192 eps),    L_domain >= pi L / sqrt(48 eps).

Measured on 20 points of the (EI, x_M/L) map, the thrust error from a finite width is 0.1 (median)
to about 1 (maximum) times `eps_Z`; `eps` is therefore an approximate upper bound on the thrust error.

## Wave scale (radiation stress)

With `B = sigma k^2 / (rho g)` the finite-depth part of the radiation stress relative to the deep
part is

    S_fd / S_main = 4 kH (1 + B) / ((1 + 3B) sinh 2kH).

Setting it equal to `eps` gives `kH = -W_{-1}(-eps (1+3B) / (2 (1+B))) / 2`, with `W_{-1}` the lower
real branch of the Lambert W function (`kH >= 3.0..3.6` for `eps = 1e-2`, `4.3..5.0` for `1e-3`).
The condition `tanh(kH) >= 0.99` of the legacy grid corresponds to `kH = 2.65`, i.e. 3.6 %.

`FlexibleParams(grid = :deep, deep_tol = eps)` applies all three conditions together with a
geometric vertical grid and raft edges on grid nodes.

## What `deep_tol` is, and is not

`eps` bounds the leading-order impedance error; it is an envelope, not an error estimate.
Measured on 20 points of the (EI, x_M/L) map (aligned edges, graded z, half horizontal resolution):

* Depth only (width 12 L, reference H = 24 L): the thrust error divided by `(pi^2/192)(L/H)^2` has a
  maximum of 0.97-1.00 for H/L = 2..4 (relative to |T|) and a median of 0.04-0.09. Relative to the
  radiated flux `Sigma = T/alpha` the maximum is 0.19-0.20 and the per-point prefactor spans 0.003-0.2.
  The scaling is `H^-p` with a per-point exponent `p` between 1.2 and 2.0 (median 1.66) at H/L = 2..4.
* Width only (depth 8 L, reference L_domain = 48 L): median 0.10, maximum 0.45-1.4 times the law.
* The depth error makes the load too large and the width error too small, with the same coefficient,
  so `L_domain = 2H` (the `:deep` rule) cancels the leading-order error: end-to-end thrust errors are
  25-250 times below `eps`.
* Extrapolating `T(H, X)`, `T(2H, X)`, `T(H, 2X)` with exponent 2 in both directions gives the correct
  sign of the error at 19/19 points and an estimate 1.48 times the true error (16-84 %: 1.45-1.66)
  at base (H, L_domain) = (1 L, 2 L). The factor is not explained by the depth exponent (using 1.66
  leaves it unchanged); errors below about 1e-4 are not resolved by the reference used.
