"""
solver/costs.py — Cost functions for the ADMM optimal control problem.

These are your Cx, Cu, L functions exactly as written — the only change
is that weights and r_target are passed as arguments instead of globals,
so the functions work correctly when imported across modules.
"""



import numpy as np

def Cx(r: np.ndarray, r_target: np.ndarray, alpha: float) -> float:
    """State cost: weighted squared distance to target trajectory."""
    error = r - r_target
    return alpha * np.sum(error ** 2)


def Cu(u: np.ndarray, alpha: float) -> float:
    """Control cost: weighted squared control effort."""
    return (1.0 - alpha) * np.sum(u ** 2)


def L(
    x        : np.ndarray,
    u        : np.ndarray,
    r        : np.ndarray,
    v        : np.ndarray,
    r_target : np.ndarray,
    P        : float,
    alpha    : float,
) -> float:
    """
    Scaled-form augmented Lagrangian (eq. 4 in the paper).

        L_ρ(r, x, u, v) = Cx(r) + Cu(u)
                          + (ρ/2) * ( ‖x − r + v‖² − ‖v‖² )
    """
    return (
        Cx(r, r_target, alpha)
        + Cu(u, alpha)
        + (P / 2) * (
            np.linalg.norm(x - r + v) ** 2
            - np.linalg.norm(v)       ** 2
        )
    )