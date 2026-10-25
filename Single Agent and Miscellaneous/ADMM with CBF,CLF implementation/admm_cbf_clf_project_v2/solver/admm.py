"""
solver/admm.py — Vehicle-agnostic ADMM outer loop.

Architectures (only two — 'iii' has been dropped, see below)
--------------------------------------------------------------
  'i'   — CBF soft (slack on CBF) + terminal scale, no CLF
             z = [x_flat | u_flat | δ_cbf(M·N)]

  'ii'  — CBF hard + CLF soft (slack on CLF), no terminal scale
             z = [x_flat | u_flat | δ_clf(N)]


Three ADMM steps — r-update replaced with a fixed A* route
--------------------------------------------------------------
  r-update  : REPLACED. r used to be solved every iteration via
              unconstrained L-BFGS-B with a soft `obstacle_penalty`
              term (see constraints.py). That's a Lagrangian trade-off,
              not a guarantee — whenever the ADMM consensus pull or the
              goal-tracking pull was strong enough, it was cheaper for
              the optimiser to eat some penalty than to detour, so the
              reference (and, chasing it, the trajectory) would cut
              through obstacles.

              r is now computed ONCE, before the ADMM loop, as a
              collision-free route from x0 to goal produced by grid A*
              (see planner_astar.py) and held fixed for every iteration.
              A* cannot expand into an occupied cell, so r is safe by
              construction — there's no penalty weight to tune and
              nothing to trade off. The xu-update still pulls x toward
              this fixed r via the ordinary ADMM consensus term
              (ρ/2)‖x−r+v‖², and the v-update still runs every
              iteration as usual; only the r half of ADMM is gone.
  xu-update : SLSQP, equality + inequality constraints (unchanged)
  v-update  : closed form (unchanged)

CBF is now HOCBF (see cbf.py for the full derivation) — h_obstacle has
relative degree 2 w.r.t. the controls for this vehicle, so a plain
one-step CBF condition has no real safety-invariance guarantee. gamma_cbf
still works as a single knob (splits into gamma1 == gamma2), or you can
pass gamma1_cbf / gamma2_cbf separately to tune the two stages of the
HOCBF chain independently.

p_slack is AUTOMATICALLY SEPARATED by architecture
-----------------------------------------------------
  arch 'i' (slack on CBF):
      Use params['slack_penalty']  — default 1000.
      Large value is correct: we want to heavily penalise safety violations.
      CBF slack at a feasible warm start is near 0 (robot starts outside
      obstacles), so the gradient is small and SLSQP is not stressed.

  arch 'ii' (slack on CLF):
      Use params['slack_penalty_clf']  — default 10.
      Small value is essential. The CLF slack at cold start is λ·V(x₀)
      which can be >> 1. With p_slack=1000 the gradient at the warm start
      is 2·1000·9.3 = 18,600 per slack variable — SLSQP returns status 8
      every iteration (positive directional derivative for linesearch)
      and the robot never moves. With p_slack=10 this gradient is 186
      and SLSQP converges reliably.


"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .costs          import L, build_weight_vector
from .cbf            import build_cbf_constraints
from .clf            import build_clf_constraints
from .planner_astar  import plan_astar_path

_ARCHITECTURES = ('i', 'ii')


# ── Dynamics equality constraint ──────────────────────────────────────────

def _dynamics_constraint(vehicle, x0_state, N, dt, size_x, size_u, total_z):
    n = vehicle.n
    m = vehicle.m

    def _fun(z):
        x_flat = z[:size_x]
        u_flat = z[size_x : size_x + size_u]
        res = np.zeros(size_x)
        res[:n] = x_flat[:n] - x0_state
        for t in range(N):
            xt = x_flat[t*n:(t+1)*n]
            ut = u_flat[t*m:(t+1)*m]
            res[(t+1)*n:(t+2)*n] = (
                x_flat[(t+1)*n:(t+2)*n] - vehicle.dynamics_step(xt, ut, dt)
            )
        return res

    def _jac(z):
        x_flat = z[:size_x]
        u_flat = z[size_x : size_x + size_u]
        J = np.zeros((size_x, total_z))
        J[:n, :n] = np.eye(n)
        for t in range(N):
            xt = x_flat[t*n:(t+1)*n]
            ut = u_flat[t*m:(t+1)*m]
            Jx, Ju = vehicle.dynamics_jac(xt, ut, dt)
            row = slice((t+1)*n, (t+2)*n)
            J[row, (t+1)*n:(t+2)*n]             =  np.eye(n)
            J[row,  t*n   :(t+1)*n]             = -Jx
            J[row, size_x+t*m:size_x+(t+1)*m]  = -Ju
        return J

    return {'type': 'eq', 'fun': _fun, 'jac': _jac}


# ── Fixed A*-derived reference (replaces the old r-update) ─────────────────

def _build_fixed_reference(vehicle, x0_state, goal, obstacles, N, astar_kwargs):
    """
    Build the flat reference trajectory r used for the whole ADMM run.

    Position columns (state[0], state[1]) come from plan_astar_path — a
    collision-free route from x0 to goal, resampled to N+1 waypoints, so
    r is safe by construction (see module docstring). Every other state
    column (heading, velocity, angular rate, ...) is filled in from
    vehicle.goal_target, i.e. the vehicle's own "stopped, at goal"
    convention. Those columns carry zero stage weight for every vehicle
    in this codebase (see stage_weights on each Vehicle), so the value
    they hold en route doesn't affect the cost — only the terminal row
    does, and that terminal row is exactly vehicle.goal_target's
    terminal row regardless of what we do here.

    Parameters
    ----------
    vehicle      : BaseVehicle — needs .n and .goal_target
    x0_state     : initial vehicle state, shape (n,)
    goal         : (x, y) target, shape (2,)
    obstacles    : list of obstacle dicts
    N            : planning horizon
    astar_kwargs : dict, optional overrides forwarded to plan_astar_path
                   (resolution, inflate, padding) — sensible defaults are
                   used for any key that's absent.

    Returns
    -------
    r_fixed : np.ndarray, shape (n*(N+1),)
    """
    n = vehicle.n
    path_xy = plan_astar_path(x0_state[:2], goal, obstacles, N, **astar_kwargs)

    r_fixed = vehicle.goal_target(goal, N).copy()
    for t in range(N + 1):
        r_fixed[t * n : t * n + 2] = path_xy[t]
    return r_fixed


# ── Feasible slack initialiser ────────────────────────────────────────────

def _compute_min_slack(
    x_flat, vehicle, goal, obstacles,
    gamma1, gamma2, lam, N, architecture,
    n_slack_cbf, n_slack_clf,
    cbf_slack_offset, clf_slack_offset,
    total_z, margin=1e-3,
):
    """
    Given a state trajectory x_flat, compute the minimum δ ≥ 0 that makes
    every slack constraint feasible, plus a small margin so z0 is strictly
    (not just marginally) feasible when passed to SLSQP.

    This is called:
      (a) Once before the ADMM loop (cold start).
      (b) After any SLSQP failure to restore feasibility for the next iter.

    Without this, arch ii's CLF slack starts at 0 but needs λ·V(x₀) ≈ 9.3,
    giving SLSQP a gradient of 2·p_slack·9.3 pointing toward reducing slack
    while the constraint says "keep slack ≥ 9.3" — status 8 every iteration.

    CBF (arch 'i') rows mirror cbf.py's HOCBF chain exactly — row 0 is the
    ψ_1(x_0) boundary condition, rows 1..N-1 are the running ψ_2 condition:

        row 0        : δ ≥ (1−g1)·h(x_0) − h(x_1)
        row t=1..N-1 : δ ≥ [(1−g1)+(1−g2)]·h(x_t) − (1−g1)(1−g2)·h(x_{t-1}) − h(x_{t+1})
    """
    n       = vehicle.n
    n_slack = n_slack_cbf + n_slack_clf
    slack   = np.zeros(n_slack)
    c1, c2  = 1.0 - gamma1, 1.0 - gamma2

    # CBF slack (arch 'i')
    if n_slack_cbf > 0:
        for oi, obs in enumerate(obstacles):
            h = np.array([
                vehicle.h_obstacle(x_flat[t*n:(t+1)*n], obs) for t in range(N + 1)
            ])
            base = oi * N
            # row 0 — boundary condition ψ_1(x_0) ≥ 0
            slack[base] = max(0.0, c1 * h[0] - h[1]) + margin
            # rows 1..N-1 — running condition ψ_2(x_{t-1}) ≥ 0
            for t in range(1, N):
                needed = (c1 + c2) * h[t] - c1 * c2 * h[t - 1] - h[t + 1]
                slack[base + t] = max(0.0, needed) + margin

    # CLF slack (arch 'ii')
    if n_slack_clf > 0:
        offset = clf_slack_offset - (total_z - n_slack)
        for t in range(N):
            xt  = x_flat[ t      * n : (t + 1) * n]
            xt1 = x_flat[(t + 1) * n : (t + 2) * n]
            Vt  = vehicle.V_clf(xt,  goal)
            Vt1 = vehicle.V_clf(xt1, goal)
            needed = Vt1 - (1.0 - lam) * Vt    # constraint: (1-λ)Vt - Vt1 + δ ≥ 0
            slack[offset + t] = max(0.0, needed) + margin

    return slack


# ── Violation diagnostic ──────────────────────────────────────────────────

def _print_violations(z, xu_constraints, label=""):
    eq_max, ineq_max = [], []
    for c in xu_constraints:
        val = np.atleast_1d(c['fun'](z))
        if c['type'] == 'eq':
            eq_max.append(float(np.max(np.abs(val))))
        else:
            ineq_max.append(float(np.max(np.maximum(0.0, -val))))
    dyn = eq_max[0] if eq_max else 0.0
    cbf = max(ineq_max[:-1]) if len(ineq_max) > 1 else (ineq_max[0] if ineq_max else 0.0)
    clf = ineq_max[-1] if len(ineq_max) > 1 else 0.0
    print(f"  {label}violations — dynamics={dyn:.2e}  CBF={cbf:.2e}  CLF={clf:.2e}")


# ── Main solver ───────────────────────────────────────────────────────────

def solver(
    x0           : np.ndarray,
    goal         : np.ndarray,
    obstacles    : list,
    params       : dict,
    vehicle,
    architecture : str = 'i',
) -> dict:
    """
    Run ADMM to find a trajectory from x0 to goal avoiding obstacles.

    Parameters
    ----------
    x0           : initial vehicle state, shape (n,)
    goal         : (x, y) target, shape (2,)
    obstacles    : list of obstacle dicts from scene_editor
    params       : dict — required keys:
                     N, rho, dt, tol, max_iter, weight_ratio,
                     lambda_clf, terminal_scale
                   optional keys (with auto-defaults):
                     gamma_cbf          — HOCBF decay rate, used for both
                                          stages unless overridden, default 0.5
                     gamma1_cbf         — ψ_1 (velocity-level) decay rate,
                                          default = gamma_cbf
                     gamma2_cbf         — ψ_2 (accel-level) decay rate,
                                          default = gamma1_cbf
                     slack_penalty      — CBF slack penalty (arch i), default 1000
                     slack_penalty_clf  — CLF slack penalty (arch ii),  default 10
                     astar_resolution   — grid cell size for the fixed
                                          reference route, default:
                                          scene extent / 60
                     astar_inflate      — obstacle inflation margin used
                                          when planning that route,
                                          default: generous auto value
                                          (see planner_astar.py)
                     astar_padding      — extra bounding-box margin so
                                          A* can route around obstacles
                                          near the scene edge, default:
                                          auto value (see planner_astar.py)
    vehicle      : BaseVehicle instance
    architecture : 'i' or 'ii'

    Returns
    -------
    dict — r_hist, x_hist, u_hist, v_hist, residuals, costs,
           dt, N, weight_ratio, architecture, vehicle_name, slsqp_failures
    """
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"architecture must be one of {_ARCHITECTURES}; got '{architecture}'")

    n = vehicle.n
    m = vehicle.m

    N              = int(params["N"])
    P              = float(params["rho"])
    tol            = float(params["tol"])
    max_iter       = int(params["max_iter"])
    weight_ratio   = float(params["weight_ratio"])
    dt             = float(params["dt"])
    gamma1         = float(params.get("gamma1_cbf", params.get("gamma_cbf", 0.5)))
    gamma2         = float(params.get("gamma2_cbf", gamma1))
    lam            = float(params.get("lambda_clf",     0.1))
    terminal_scale = float(params.get("terminal_scale", 80.0))

    # Architecture-specific p_slack (see module docstring for rationale)
    if architecture == 'ii':
        p_slack = float(params.get("slack_penalty_clf", 10.0))
    else:
        p_slack = float(params.get("slack_penalty", 1000.0))

    alpha  = weight_ratio / (weight_ratio + 1.0)
    size_x = n * (N + 1)
    size_u = m * N
    M      = len(obstacles)

    # SLSQP options — tunable via params. The old default (max(300,15*N))
    # let a single stuck call burn ~2.5s doing nothing (a FAILED SLSQP
    # call's result is discarded — see below — so iterations spent past
    # the point of no return are pure waste). 150 is enough for the vast
    # majority of calls to converge cleanly; calls that are genuinely
    # stuck now fail ~2.5x faster and hand control back to the next ADMM
    # round (where the growing dual variable v usually unsticks them).
    slsqp_ftol    = float(params.get("slsqp_ftol",    max(tol * 0.1, 1e-6)))
    slsqp_maxiter = int(params.get("slsqp_maxiter",   max(300, 15 * N)))

    tag = f"[{type(vehicle).__name__} | arch {architecture}]"

    # ── z-vector layout ────────────────────────────────────────────────
    if architecture == 'i':
        n_slack_cbf      = M * N
        n_slack_clf      = 0
        cbf_slack_offset = size_x + size_u
        clf_slack_offset = size_x + size_u   # unused
    else:  # architecture == 'ii'
        n_slack_cbf      = 0
        n_slack_clf      = N
        cbf_slack_offset = size_x + size_u   # unused
        clf_slack_offset = size_x + size_u

    n_slack = n_slack_cbf + n_slack_clf
    total_z = size_x + size_u + n_slack

    # ── cost setup ────────────────────────────────────────────────────
    weight_vec = build_weight_vector(
        n, N, vehicle.stage_weights, vehicle.terminal_weights, terminal_scale,
    )

    # r is no longer solved for — it's a fixed, collision-free route from
    # A*, computed once here and held constant for the whole ADMM run
    # (see module docstring and _build_fixed_reference above).
    astar_kwargs = {
        k: params[pk] for k, pk in (
            ("resolution", "astar_resolution"),
            ("inflate",    "astar_inflate"),
            ("padding",    "astar_padding"),
            ("max_speed",  "astar_max_speed"),
            ("ramp_time",  "astar_ramp_time"),
        ) if pk in params
    }
    astar_kwargs.setdefault("dt", dt)
    r_fixed  = _build_fixed_reference(vehicle, x0, goal, obstacles, N, astar_kwargs)
    r_target = r_fixed  # kept for cost bookkeeping/logging only — see L()

    # Diagnostic: a FIXED reference that demands more speed than the
    # vehicle can plausibly deliver is the #1 cause of the xu-update
    # thrashing/oscillating every ADMM round (see planner_astar.py's
    # _resample_path_eased docstring). Neither this module nor
    # BaseVehicle know the vehicle's real speed limit, so this can't be
    # silently fixed — surface it instead. If 'astar_max_speed' wasn't
    # set in params, the reference is UNCAPPED and will demand exactly
    # whatever speed the route/horizon combination implies.
    _ref_pos = r_fixed.reshape(N + 1, n)[:, :2]
    _ref_speeds = np.linalg.norm(np.diff(_ref_pos, axis=0), axis=1) / dt
    if "astar_max_speed" not in params and _ref_speeds.max() > 0:
        print(
            f"{tag} reference speed: peak={_ref_speeds.max():.2f}, "
            f"avg={_ref_speeds.mean():.2f} (units/s) over horizon "
            f"N*dt={N * dt:.2f}s. Uncapped (set params['astar_max_speed'] "
            f"to cap it) — if this peak looks unrealistically high for "
            f"the vehicle, that's almost certainly why the xu-update is "
            f"struggling; increase N/dt or set astar_max_speed."
        )

    bounds = (
        [(None, None)] * (size_x + size_u) +
        [(0.0,  None)] * n_slack
    )

    # ── constraints (built once, reused every iteration) ──────────────
    dyn_con  = _dynamics_constraint(vehicle, x0, N, dt, size_x, size_u, total_z)
    cbf_cons = build_cbf_constraints(
        vehicle, obstacles, gamma1, gamma2, N, architecture,
        size_x, size_u, total_z, cbf_slack_offset,
    )
    clf_cons = build_clf_constraints(
        vehicle, goal, lam, N, architecture,
        size_x, size_u, total_z, clf_slack_offset,
    )
    xu_constraints = [dyn_con] + cbf_cons + clf_cons

    # ── initialise ADMM ────────────────────────────────────────────────
    k              = 0
    slsqp_failures = 0
    x = np.tile(x0, N + 1)
    u = np.zeros(size_u)
    v = np.zeros(size_x)
    r = r_fixed.copy()   # fixed for the whole run — see above

    # Compute feasible slack for cold start (avoids status 8 at iter 0)
    prev_slack = _compute_min_slack(
        x, vehicle, goal, obstacles, gamma1, gamma2, lam, N, architecture,
        n_slack_cbf, n_slack_clf, cbf_slack_offset, clf_slack_offset, total_z,
    )

    r_hist, x_hist, u_hist, v_hist = [], [], [], []
    residuals, costs = [], []

    # ── ADMM loop ──────────────────────────────────────────────────────
    while True:
        x_ = x.copy()
        u_ = u.copy()
        v_ = v.copy()
        r_ = r  # fixed — no r-update to solve, see module docstring

        # ── xu-update ─────────────────────────────────────────────── #
        def Lxu(z):
            xz    = z[:size_x]
            uz    = z[size_x : size_x + size_u]
            slack = z[size_x + size_u:]
            return (
                L(xz, uz, r_, v_, r_target, P, alpha, weight_vec)
                + p_slack * float(np.sum(slack ** 2))
            )

        def Lxu_grad(z):
            xz    = z[:size_x]
            uz    = z[size_x : size_x + size_u]
            slack = z[size_x + size_u:]
            return np.concatenate([
                P * (xz - r_ + v_),
                2.0 * (1.0 - alpha) * uz,
                2.0 * p_slack * slack,
            ])

        # Warm start: previous trajectory + feasible slack
        z0 = np.zeros(total_z)
        z0[:size_x]                   = x_
        z0[size_x : size_x + size_u] = u_
        z0[size_x + size_u:]          = prev_slack

        res_xu = minimize(
            Lxu,
            x0          = z0,
            jac         = Lxu_grad,
            constraints = xu_constraints,
            bounds      = bounds,
            method      = "SLSQP",
            options     = {"ftol": slsqp_ftol, "maxiter": slsqp_maxiter},
        )

        # Commit only if SLSQP converged
        if res_xu.status == 0:
            x_         = res_xu.x[:size_x]
            u_         = res_xu.x[size_x : size_x + size_u]
            prev_slack = res_xu.x[size_x + size_u:].copy()
        else:
            slsqp_failures += 1
            _print_violations(res_xu.x, xu_constraints, label=f"iter {k} ")
            print(
                f"  {tag} SLSQP status {res_xu.status}: {res_xu.message}"
                f" — keeping previous x_, u_"
            )
            # Recompute feasible slack from current (unchanged) x_
            prev_slack = _compute_min_slack(
                x_, vehicle, goal, obstacles, gamma1, gamma2, lam, N, architecture,
                n_slack_cbf, n_slack_clf, cbf_slack_offset, clf_slack_offset, total_z,
            )

        # ── v-update ──────────────────────────────────────────────── #
        v_ += x_ - r_
        x, r, u, v = x_, r_, u_, v_

        # ── record ────────────────────────────────────────────────── #
        r_hist.append(r_.reshape(N + 1, n).copy())
        x_hist.append(x_.reshape(N + 1, n).copy())
        u_hist.append(u_.reshape(N,     m).copy())
        v_hist.append(v_.reshape(N + 1, n).copy())

        residual = float(np.linalg.norm(r_ - x_) ** 2)
        cost_val = float(L(x_, u_, r_, v_, r_target, P, alpha, weight_vec))
        residuals.append(residual)
        costs.append(cost_val)

        # ── stopping criteria ─────────────────────────────────────── #
        if residual < tol:
            print(
                f"{tag} Converged in {k} iterations.  "
                f"residual={residual:.3e}  "
                f"SLSQP failures={slsqp_failures}/{k+1}"
            )
            break

        if k >= max_iter:
            print(
                f"{tag} Max iterations ({max_iter}) reached.  "
                f"residual={residual:.3e}  "
                f"SLSQP failures={slsqp_failures}/{k+1}"
            )
            break

        k += 1

    return {
        "r_hist"        : r_hist,
        "x_hist"        : x_hist,
        "u_hist"        : u_hist,
        "v_hist"        : v_hist,
        "residuals"     : residuals,
        "costs"         : costs,
        "dt"            : dt,
        "N"             : N,
        "weight_ratio"  : weight_ratio,
        "architecture"  : architecture,
        "vehicle_name"  : type(vehicle).__name__,
        "slsqp_failures": slsqp_failures,
    }
