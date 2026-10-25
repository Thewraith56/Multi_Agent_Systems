# Drone JED / DTCBF Navigation

Discrete-time, stochastic-safety obstacle avoidance for a simulated drone,
implementing Theorem 6 (JED controller) + Theorem 5 (finite-horizon Pu
bound) from Cosner, Culbertson, Taylor, Ames, *"Robust Safety under
Stochastic Uncertainty with Discrete-Time Control Barrier Functions"*
(arXiv:2302.07469v2).

Run: `python main.py`

## Architecture — three independent, stacked layers

1. **Global planner (A\*)** — `solver/planner.py`. Plans once over inflated
   obstacle geometry (`r_robot` + buffer + a velocity-dependent braking
   term, using a representative speed since A\* has no velocity state),
   then simplifies the path with line-of-sight shortcutting.
2. **Waypoint tracker (LQR)** — `solver/tracker.py`. Infinite-horizon
   discrete LQR (`scipy.linalg.solve_discrete_are`) regulating straight to
   `x_ref = [g_i, 0, 0]` for the current waypoint. Has **no** knowledge of
   obstacles beyond what A* already encoded in the waypoints, and its only
   interface to the safety filter is the scalar `u_nom` it emits.
3. **JED safety filter** — `solver/jed_solver.py` + `solver/barriers.py`.
   Every step, rebuilds a reactive multi-hyperplane polytope DTCBF from
   whichever obstacles are within sensing range, then projects `u_nom` onto
   the nearest input satisfying the log-sum-exp relaxation of the
   expected-barrier condition (Appendix C of the paper, extended here to
   biased, anisotropic Gaussian noise). A slack variable absorbs
   infeasibility in narrow gaps; any solve with `slack > 0` is logged as
   breaking the strict Theorem 5 guarantee.

`solver/pu_bound.py` implements Theorem 5 (Eq. 13/14) and Theorem 6's
`delta`, and `solver/sim_loop.run_monte_carlo` validates the bound against
empirical exit rates over many rollouts (paper's Fig. 2/4 style plot).

**No CLF anywhere** — goal-reaching is entirely the A*/LQR layers' job;
safety is entirely JED's job. They only ever interact through `u_nom`.

## What changed in the uploaded files

- `base.py` / `drone.py` (now under `solver/vehicles/`): removed
  `V_clf`/`V_clf_grad` (no CLF), removed `h_obstacle`/`h_obstacle_grad`
  (barrier is now geometry-driven in `barriers.py`, not per-vehicle) and
  `goal_target` (no MPC/ADMM running cost). `dynamics_step`/`dynamics_jac`
  collapsed into one linear `step()` using `solver/dynamics.build_AB()`.
- `app.py` / `param_panel.py`: rewired for the new pipeline — "Plan + Run
  JED" replaces the old MPC solve, plus new "Run Monte Carlo Validation"
  and "Tuning Dashboard" buttons/panel sections; the old CLF/probabilistic-
  CBF-method fields are gone.
- `scene_editor.py`: interaction logic unchanged (still shared/drone-
  compatible); `draw_result` now labels the dashed line "A* waypoints"
  instead of "Reference".

## New files

`solver/`: `config.py` (every tunable constant), `dynamics.py` (A,B),
`noise.py`, `barriers.py`, `jed_solver.py`, `planner.py`, `tracker.py`,
`sim_loop.py`, `pu_bound.py`.
`ui/`: `analysis.py` (post-run plots), `tuning_dashboard.py` (offline,
live-slider Theorem 5 explorer).

## Known caveats

- The LSE relaxation's outer variable `t` is swept on a small log-spaced
  grid per step rather than jointly optimized with `u` inside a single
  cvxpy DPP problem — see `barriers.lse_expected_barrier_constraint`'s
  docstring for why (the joint (u,t) expression is not affine-DCP; for
  fixed `t` it is, which is the standard Chernoff/MGF pattern).
- `M` (Theorem 5's upper bound on h) is saturated at `R_sense` when no
  obstacle is active, which makes the bound quite conservative far from
  obstacles — consistent with the paper's own observation (Fig. 4) that
  the theoretical bound is much looser than Monte Carlo in practice.
