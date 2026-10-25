"""
solver/barriers.py — Reactive multi-hyperplane polytope DTCBF.

Rebuilt every timestep from the drone's current position. Generalizes the
single "separating hyperplane" CBF to multiple simultaneous obstacles:

    h(x) = -max_i( (C_k x)_i - (w_k)_i )      (min of affines -> concave,
                                                 lambda_max = 0 exactly)

Safe set: {x : C_k x <= w_k}  <=>  {x : h(x) >= 0}.

Each row (n_hat_i, b_i) is a supporting hyperplane {x : n_hat_i . x >= b_i}
of one active (nearby) obstacle; converting to the C x <= w convention used
by h means c_i = -n_hat_i, w_i = -b_i.
"""

from __future__ import annotations

import numpy as np

from . import config


# ═══════════════════════════════════════════════════════════════════
#  Per-obstacle separating hyperplane
# ═══════════════════════════════════════════════════════════════════

def get_separating_hyperplane(
    x: np.ndarray,
    obstacle: dict,
    r_robot: float = config.R_ROBOT,
    v: np.ndarray | None = None,
    a_max: float = config.A_MAX,
) -> tuple[np.ndarray, float] | None:
    """
    Nearest point / outward normal supporting hyperplane for one obstacle,
    inflated by r_robot and (if v is given) a velocity-dependent braking
    buffer:

        b += max(0, v_toward_obstacle)^2 / (2*a_max)
        v_toward_obstacle = -v . n_hat     (closing speed along the normal)

    Parameters
    ----------
    x        : drone state, shape (n,) — only x[:2] (position) is used.
    obstacle : {"type": "circle", "params": (cx, cy, R)}
               or {"type": "rectangle", "params": (x_min, y_min, x_max, y_max)}
    r_robot  : robot radius to inflate the obstacle by.
    v        : drone velocity, shape (2,), or None to skip the braking term.
    a_max    : max commanded acceleration, used to size the braking buffer.

    Returns
    -------
    (n_hat, b) such that the safe half-space is {x : n_hat . x >= b}, or
    None if the state is exactly at the obstacle's reference point (degenerate,
    normal undefined) — caller should treat this as "already unsafe".
    """
    pos = np.asarray(x[:2], dtype=float)

    if obstacle["type"] == "circle":
        cx, cy, R = obstacle["params"]
        center = np.array([cx, cy])
        diff = pos - center
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return None
        n_hat = diff / dist
        b = float(n_hat @ center) + R + r_robot

    elif obstacle["type"] == "rectangle":
        x_min, y_min, x_max, y_max = obstacle["params"]
        lo = np.array([x_min, y_min])
        hi = np.array([x_max, y_max])
        p_star = np.clip(pos, lo, hi)
        diff = pos - p_star
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            # Inside the rectangle (shouldn't happen if already safe) — push
            # out along the axis of least penetration.
            penetration = np.minimum(pos - lo, hi - pos)
            axis = int(np.argmin(penetration))
            n_hat = np.zeros(2)
            n_hat[axis] = 1.0 if pos[axis] - lo[axis] < hi[axis] - pos[axis] else -1.0
            p_star = pos.copy()
        else:
            n_hat = diff / dist
        b = float(n_hat @ p_star) + r_robot

    else:
        raise ValueError(f"Unknown obstacle type: {obstacle['type']!r}")

    if v is not None:
        v_toward = float(-np.asarray(v, dtype=float) @ n_hat)   # closing speed
        if v_toward > 0.0:
            b += (v_toward ** 2) / (2.0 * a_max)

    return n_hat, b


# ═══════════════════════════════════════════════════════════════════
#  Active-set polytope
# ═══════════════════════════════════════════════════════════════════

def build_active_polytope(
    x: np.ndarray,
    obstacles: list[dict],
    R_sense: float,
    r_robot: float = config.R_ROBOT,
    v: np.ndarray | None = None,
    a_max: float = config.A_MAX,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Build C_k, w_k from every obstacle within R_sense of the current
    position. Only "active" (nearby) obstacles are included, to keep the
    per-step QP small and avoid over-conservative/infeasible constraints
    from far-away obstacles.

    Returns
    -------
    C : shape (m_active, n_pos=2)   [rows are -n_hat]
    w : shape (m_active,)           [entries are -b]
    active_idx : list of obstacle indices that were included
    """
    pos = np.asarray(x[:2], dtype=float)
    rows_C, rows_w, active_idx = [], [], []

    for i, obs in enumerate(obstacles):
        hp = get_separating_hyperplane(x, obs, r_robot, v, a_max)
        if hp is None:
            # Degenerate / already-at-center case: treat as maximally active.
            rows_C.append(np.zeros(2))
            rows_w.append(-1e6)     # forces h very negative -> flagged unsafe
            active_idx.append(i)
            continue
        n_hat, b = hp

        # distance from current position to the obstacle's reference point,
        # used only to decide whether it's "close enough" to matter.
        if obs["type"] == "circle":
            cx, cy, R = obs["params"]
            dist_to_surface = np.linalg.norm(pos - np.array([cx, cy])) - R
        else:
            x_min, y_min, x_max, y_max = obs["params"]
            lo, hi = np.array([x_min, y_min]), np.array([x_max, y_max])
            p_star = np.clip(pos, lo, hi)
            dist_to_surface = np.linalg.norm(pos - p_star)

        if dist_to_surface <= R_sense:
            rows_C.append(-n_hat)
            rows_w.append(-b)
            active_idx.append(i)

    if not rows_C:
        return np.zeros((0, 2)), np.zeros((0,)), []

    return np.array(rows_C), np.array(rows_w), active_idx


def sense_radius(v: np.ndarray | None, dt: float = config.DT,
                  sigma_d: float = config.SIGMA_D_REPRESENTATIVE) -> float:
    """R_sense = ||v||*dt + 3*sigma_d."""
    speed = float(np.linalg.norm(v)) if v is not None else 0.0
    return speed * dt + config.SENSE_SIGMA_MULT * sigma_d


# ═══════════════════════════════════════════════════════════════════
#  Barrier value
# ═══════════════════════════════════════════════════════════════════

def h_polytope(x: np.ndarray, C: np.ndarray, w: np.ndarray,
               M_saturate: float | None = None) -> float:
    """
    h(x) = -max_i( (C x)_i - w_i ).

    If the active set is empty (no rows), h is not naturally bounded — the
    caller passes M_saturate (config.compute_M(...)) to report a saturated
    "as safe as we can claim" value instead of +inf.
    """
    if C.shape[0] == 0:
        return float(M_saturate) if M_saturate is not None else float("inf")
    pos = np.asarray(x[:2], dtype=float)
    return float(-np.max(C @ pos - w))


# ═══════════════════════════════════════════════════════════════════
#  Expected-barrier LSE relaxation (extends paper Appendix C to biased,
#  anisotropic Gaussian noise over the FULL state; only position rows of
#  A, B, mu, Sigma matter since C, w act on position only)
# ═══════════════════════════════════════════════════════════════════

def lse_expected_barrier_constraint(x_k, u, A, B, mu, Sigma, C_k, w_k, t):
    """
    Build the cvxpy expression for an upper bound on -E[h(x_{k+1})], via the
    log-sum-exp relaxation of the pointwise max in h:

        -E[h(x_{k+1})] <= (1/t) * log( sum_i exp(t*mu_i(u) + (t^2/2)*sigma_i^2) )

    with, per active face i (row of C_k, w_k, acting on position only):
        mu_i(u)   = c_i . (A_pos x_k + B_pos u) - w_i + c_i . mu_pos
        sigma_i^2 = c_i . Sigma_pos . c_i

    Design note: for FIXED scalar t > 0, each mu_i(u) is affine in the
    decision variable u, so t*mu_i(u) + const is affine in u and this whole
    expression is a straightforward convex log-sum-exp of affine functions
    (DCP-compliant). The paper additionally optimizes jointly over t; we
    instead treat t as an outer scalar swept by the caller (jed_solver.py)
    and solve the resulting convex QP/NLP for u at each candidate t — this
    is the standard MGF/Chernoff-bound pattern and avoids relying on a
    bilinear (u,t) term inside cvxpy's DCP ruleset.

    Parameters
    ----------
    x_k    : current state, shape (n,)  (numpy)
    u      : cvxpy Variable, shape (m,) — the decision variable
    A, B   : full dynamics matrices, shape (n,n), (n,m)
    mu     : noise mean, shape (n,)
    Sigma  : noise covariance, shape (n,n)
    C_k, w_k : active polytope (acts on position, shape (m_active,2), (m_active,))
    t      : scalar float > 0 (fixed for this call)

    Returns
    -------
    cvxpy Expression — the RHS-independent LSE bound; caller imposes
        expr <= -alpha * h(x_k) + slack
    """
    import cvxpy as cp

    if C_k.shape[0] == 0:
        # No active obstacles this step -> vacuously satisfied; return -inf-like.
        return cp.Constant(-1e6)

    A_pos = A[:2, :]
    B_pos = B[:2, :]
    mu_pos = mu[:2]
    Sigma_pos = Sigma[:2, :2]

    x_next_nom = A_pos @ x_k          # position rows only, control term added below
    terms = []
    for i in range(C_k.shape[0]):
        c_i = C_k[i]
        w_i = w_k[i]
        mu_i = c_i @ (x_next_nom + B_pos @ u) - w_i + float(c_i @ mu_pos)
        sigma2_i = float(c_i @ Sigma_pos @ c_i)
        terms.append(t * mu_i + 0.5 * (t ** 2) * sigma2_i)

    lse = cp.log_sum_exp(cp.hstack(terms))
    return lse / t
