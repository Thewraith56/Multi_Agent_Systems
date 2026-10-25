"""
solver/admm.py — ADMM outer loop (layered control architecture).

This is your solver() function with the minimal changes discussed:
  - Accepts (x0, goal, obstacles, params) instead of reading globals.
  - Cx / Cu / L receive their weights via the closures below.
  - dynamics(z) is replaced by build_dynamics_constraint().
  - A history dict is returned for the analysis plots.

Everything else — the three-step ADMM structure, both local closures
(Lr and Lxu), the stopping criteria, the loop variable names — is
unchanged from your original notebook code.
"""

import numpy as np
from scipy.optimize import minimize

from .costs       import L, Cx_grad
from .dynamics    import build_dynamics_constraint
from .constraints import obstacle_penalty, obstacle_penalty_grad


# Fixed for unicycle — do not change
# Upgraded to Acceleration-Controlled Dynamic Model
_N_STATES   = 5   # [x1, x2, theta, v, omega]
_N_CONTROLS = 2   # [a,  alpha]



def solver(
    x0        : np.ndarray,
    goal      : np.ndarray,
    obstacles : list,
    params    : dict,
) -> dict:
    """
    Run ADMM to find a trajectory from x0 to goal avoiding obstacles.

    Parameters
    ----------
    x0        : initial state, shape (3,)  — [x1, x2, theta]
    goal      : target (x, y),  shape (2,)
    obstacles : list of obstacle dicts from scene_editor
    params    : dict with keys:
                  N, rho, dt, tol, max_iter, weight_ratio

    Returns
    -------
    dict with keys:
        r_hist    : list of np.ndarray (N+1, 3) — reference per iteration
        x_hist    : list of np.ndarray (N+1, 3) — state    per iteration
        u_hist    : list of np.ndarray (N,   2) — controls per iteration
        v_hist    : list of np.ndarray (N+1, 3) — dual var per iteration
        residuals : list of float  — ‖x − r‖²  per iteration
        costs     : list of float  — L_ρ value  per iteration
        dt        : float          — timestep (passed through for plots)
        N         : int            — horizon   (passed through for plots)
    """
    # ── unpack params ──────────────────────────────────────────────────
    n        = _N_STATES
    m        = _N_CONTROLS
    N        = int(params["N"])
    P        = float(params["rho"])
    tol      = float(params["tol"])
    max_iter = int(params["max_iter"])
    weight_ratio = float(params["weight_ratio"])
    dt       = float(params["dt"])
    alpha = weight_ratio / (weight_ratio + 1.0)
    size_x = n * (N + 1)
    size_u = m * N

    # r_target: goal (x, y) replicated for every timestep; theta target = 0
    # Target: [Goal X, Goal Y, target theta=0, target v=0, target omega=0]
    r_target = np.tile(np.array([goal[0], goal[1], 0.0, 0.0, 0.0]), N + 1)
    # ── initialise — your original variable names ──────────────────────
    k = 0
    x = np.tile(x0, N + 1)          # start the state trajectory at x0
    u = np.zeros(size_u)
    v = np.zeros(size_x)
    r = x.copy()

    # ── history ────────────────────────────────────────────────────────
    r_hist, x_hist, u_hist, v_hist = [], [], [], []
    residuals, costs = [], []

    # ── dynamics constraint (x0 fixed for all ADMM iterations) ────────
    cons = build_dynamics_constraint(x0, n, m, N, dt)

    # ── ADMM loop ──────────────────────────────────────────────────────
    while True:
        x_ = x.copy()
        u_ = u.copy()
        v_ = v.copy()
        r_ = r.copy()

        # ── r-update : trajectory generation layer (eq. 6a) ───────── #
        def Lr(r):
            return (
                L(x_, u_, r, v_, r_target, P, alpha)
                + obstacle_penalty(r, obstacles, n, N)
            )

        def Lr_grad(r):
            err = x_ - r + v_
            return (
                Cx_grad(r, r_target, alpha)
                - P * err
                + obstacle_penalty_grad(r, obstacles, n, N)
            )

        res = minimize(Lr, x0=r_.copy(), jac=Lr_grad, method="L-BFGS-B")
        r_  = res.x

        # ── (x, u)-update : feedback control layer (eq. 6b) ────────── #
        def Lxu(z):
            xz = z[:size_x]
            uz = z[size_x:]
            return L(xz, uz, r_, v_, r_target, P, alpha)

        def Lxu_grad(z):
            xz = z[:size_x]
            uz = z[size_x:]
            g_x = P * (xz - r_ + v_)
            g_u = 2.0 * (1.0 - alpha) * uz
            return np.concatenate([g_x, g_u])

        z0  = np.concatenate([x_, u_])
        res = minimize(Lxu, x0=z0, jac=Lxu_grad, constraints=cons, method="SLSQP")
        x_  = res.x[:size_x]
        u_  = res.x[size_x:]

        # ── v-update : dual variable (eq. 6c) ───────────────────────── #
        v_ += x_ - r_

        (v, x, r, u) = (v_, x_, r_, u_)

        # ── record ──────────────────────────────────────────────────── #
        r_hist.append(r_.reshape(N + 1, n).copy())
        x_hist.append(x_.reshape(N + 1, n).copy())
        u_hist.append(u_.reshape(N,     m).copy())
        v_hist.append(v_.reshape(N + 1, n).copy())

        residual = float(np.linalg.norm(r_ - x_) ** 2)
        cost_val = float(L(x_, u_, r_, v_, r_target, P, alpha))
        residuals.append(residual)
        costs.append(cost_val)

        # ── stopping criteria — your original logic ──────────────────── #
        if residual < tol:
            print(f"Converged in {k} iterations!")
            break

        if k >= max_iter:
            print("Max iterations reached.")
            break

        k += 1

    return {
        "r_hist"    : r_hist,
        "x_hist"    : x_hist,
        "u_hist"    : u_hist,
        "v_hist"    : v_hist,
        "residuals" : residuals,
        "costs"     : costs,
        "dt"        : dt,
        "N"         : N,
        "weight_ratio" : weight_ratio,
    }
