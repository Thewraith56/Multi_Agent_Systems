

import numpy as np
from scipy.optimize import minimize

from .costs       import L
from .dynamics    import build_dynamics_constraint
from .constraints import obstacle_penalty

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

        res = minimize(Lr, x0=r_.copy())
        r_  = res.x

        # ── (x, u)-update : feedback control layer (eq. 6b) ────────── #
        def Lxu(z):
            xz = z[:size_x]
            uz = z[size_x:]
            return L(xz, uz, r_, v_, r_target, P, alpha)

        z0  = np.concatenate([x_, u_])
        res = minimize(Lxu, x0=z0, constraints=cons, method="SLSQP")
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
