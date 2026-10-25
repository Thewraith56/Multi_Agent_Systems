"""
solver/jed_solver.py — JED safety filter.

Projects a nominal control u_nom (from tracker.WaypointTracker, treated as a
black box) onto the nearest input satisfying the stochastic DTCBF
constraint, via the log-sum-exp relaxation of Appendix C (barriers.py),
extended to the reactive multi-hyperplane polytope of barriers.py.

Per Section IV / Theorem 6 of the paper: because this LSE bound already
upper-bounds -E[h(x_{k+1})] directly (it is not the "plug in the mean, then
apply Lemma 1" construction), no separate Jensen-gap margin c_J is needed —
c_J = 0 for this barrier. See pu_bound.compute_delta for how this feeds
Theorem 5's delta.

A slack variable s >= 0 absorbs infeasibility in narrow gaps (heavily
penalized in the objective); any solve with s > 0 breaks the strict
Theorem 5 guarantee and is flagged by the caller (sim_loop.py logs this).
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from . import config
from .barriers import lse_expected_barrier_constraint, h_polytope


class JEDResult:
    __slots__ = ("u", "slack", "t_used", "h_x", "feasible")

    def __init__(self, u, slack, t_used, h_x, feasible):
        self.u = u
        self.slack = slack
        self.t_used = t_used
        self.h_x = h_x
        self.feasible = feasible


def _solve_for_fixed_t(x_k, u_nom, A, B, mu, Sigma, C, w, alpha, h_x, t,
                        u_max=config.U_MAX, slack_penalty=config.SLACK_PENALTY):
    """One convex QP solve for a fixed scalar t. Returns (u, slack, cost)."""
    u = cp.Variable(2)
    s = cp.Variable(nonneg=True)

    lse_bound = lse_expected_barrier_constraint(x_k, u, A, B, mu, Sigma, C, w, t)
    constraints = [
        lse_bound <= -alpha * h_x + s,
        cp.abs(u) <= u_max,
    ]
    objective = cp.Minimize(cp.sum_squares(u - u_nom) + slack_penalty * cp.square(s))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception:
            return None, None, np.inf

    if u.value is None:
        return None, None, np.inf
    return u.value, float(s.value), float(prob.value)


def jed_qp_solve(
    x_k: np.ndarray,
    u_nom: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    C: np.ndarray,
    w: np.ndarray,
    alpha: float = config.ALPHA,
    M_saturate: float | None = None,
    t_grid: np.ndarray | None = None,
) -> JEDResult:
    """
    Full JED safety filter for one timestep. Scans a small log-spaced grid
    of t values (approximating the paper's joint (u,t) optimization; see
    lse_expected_barrier_constraint's docstring), solves the convex QP for
    each, and keeps the best (lowest-cost, i.e. least control deviation +
    slack) result.
    """
    h_x = h_polytope(x_k, C, w, M_saturate=M_saturate)

    if C.shape[0] == 0:
        # No active obstacles -> unconstrained, just clip to actuator limits.
        u_clipped = np.clip(u_nom, -config.U_MAX, config.U_MAX)
        return JEDResult(u_clipped, 0.0, None, h_x, True)

    if t_grid is None:
        t_grid = np.geomspace(config.T_MIN, config.T_MAX, 12)

    best = None
    for t in t_grid:
        u_val, s_val, cost = _solve_for_fixed_t(x_k, u_nom, A, B, mu, Sigma, C, w, alpha, h_x, float(t))
        if u_val is None:
            continue
        if best is None or cost < best[2]:
            best = (u_val, s_val, cost, float(t))

    if best is None:
        # Total solver failure -- fall back to zero control (hold) rather
        # than propagate a crash into the sim loop.
        return JEDResult(np.zeros(2), np.inf, None, h_x, False)

    u_val, s_val, _, t_used = best
    return JEDResult(u_val, s_val, t_used, h_x, True)
