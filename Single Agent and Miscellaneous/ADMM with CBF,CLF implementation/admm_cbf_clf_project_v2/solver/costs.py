"""
solver/costs.py — Vehicle-agnostic stage and terminal costs.

All weight vectors are supplied by the caller (built from the vehicle object
+ terminal_scale from params), so this module contains zero vehicle-specific
constants.


Public API
----------
build_weight_vector(n, N, stage_weights, terminal_weights, terminal_scale)
    -> np.ndarray shape (n*(N+1),)

Cx      (r, r_target, alpha, weight_vec)          -> float
Cx_grad (r, r_target, alpha, weight_vec)          -> np.ndarray
Cu      (u, alpha)                                -> float
L       (x, u, r, v, r_target, P, alpha, weight_vec) -> float
"""

import numpy as np


# ── Weight vector construction ────────────────────────────────────────────

def build_weight_vector(
    n                : int,
    N                : int,
    stage_weights    : np.ndarray,
    terminal_weights : np.ndarray,
    terminal_scale   : float,
) -> np.ndarray:
    """
    Build the flat cost-weight vector covering the full horizon.

    Timesteps t = 0 … N-1  →  stage_weights   (shape n each)
    Timestep  t = N         →  terminal_weights * terminal_scale

    Set terminal_scale = 0.0 for architectures ii and iii, where the CLF
    constraint drives convergence and the terminal cost is redundant.

    Parameters
    ----------
    n                : state dimension
    N                : planning horizon (number of steps)
    stage_weights    : shape (n,)
    terminal_weights : shape (n,)
    terminal_scale   : scalar multiplier applied to terminal_weights

    Returns
    -------
    w : np.ndarray, shape (n*(N+1),)
    """
    w = np.tile(stage_weights.astype(float), N + 1)
    w[N * n :] = terminal_weights.astype(float) * terminal_scale
    return w


# ── Individual cost terms ─────────────────────────────────────────────────

def Cx(
    r          : np.ndarray,
    r_target   : np.ndarray,
    alpha      : float,
    weight_vec : np.ndarray,
) -> float:
    """
    Weighted squared distance of the reference r from the target trajectory.

        Cx = α · Σ_t  w_t ⊙ (r_t − r*_t)²
    """
    error = r - r_target
    return float(alpha * np.sum(weight_vec * error ** 2))


def Cx_grad(
    r          : np.ndarray,
    r_target   : np.ndarray,
    alpha      : float,
    weight_vec : np.ndarray,
) -> np.ndarray:
    """
    Analytic gradient of Cx w.r.t. r.

        ∂Cx/∂r = 2α · w ⊙ (r − r*)
    """
    return 2.0 * alpha * weight_vec * (r - r_target)


def Cu(u: np.ndarray, alpha: float) -> float:
    """
    Quadratic control-effort cost.

        Cu = (1 − α) · ‖u‖²
    """
    return float((1.0 - alpha) * np.sum(u ** 2))


# ── Full augmented Lagrangian (without slack penalty) ─────────────────────

def L(
    x          : np.ndarray,
    u          : np.ndarray,
    r          : np.ndarray,
    v          : np.ndarray,
    r_target   : np.ndarray,
    P          : float,
    alpha      : float,
    weight_vec : np.ndarray,
) -> float:
    """
    Augmented Lagrangian evaluated at (x, u, r) with dual variable v.

        L = Cx(r) + Cu(u) + (ρ/2)·(‖x − r + v‖² − ‖v‖²)

    The slack penalty  p·‖δ‖²  is added separately in the xu-update
    objective (Lxu inside admm.py) so that L itself stays architecture-neutral.

    Parameters
    ----------
    x, u, r, v  : flat trajectory vectors
    r_target     : flat desired trajectory, same shape as r
    P            : ADMM penalty coefficient ρ
    alpha        : weight-ratio scalar  α = γ/(γ+1)
    weight_vec   : shape (n*(N+1),)  — from build_weight_vector()
    """
    return (
        Cx(r, r_target, alpha, weight_vec)
        + Cu(u, alpha)
        + (P / 2.0) * (
            float(np.linalg.norm(x - r + v) ** 2)
            - float(np.linalg.norm(v)        ** 2)
        )
    )
