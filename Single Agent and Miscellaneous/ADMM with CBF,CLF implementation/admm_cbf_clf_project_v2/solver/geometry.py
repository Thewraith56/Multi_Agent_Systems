"""
solver/vehicles/geometry.py — Shared position-only obstacle barrier math.

Every vehicle planned for this codebase (Drone, Unicycle, Bicycle) defines
its CBF barrier h(x) to depend only on position (x[0], x[1]) — see the
convention documented in base.py. That means the circle / smoothed-
rectangle math is IDENTICAL across vehicles; only the surrounding state
vector layout differs. This module holds that shared math once, so adding
a new vehicle means calling into here from h_obstacle / h_obstacle_grad,
not re-deriving (and risking re-deriving incorrectly) the same formulas.

See unicycle.py's original docstring / cbf.py's module docstring for the
full derivation and rationale of the smoothed rectangle barrier; this
module only holds the (vehicle-independent) implementation.
"""

from __future__ import annotations

import numpy as np


def smooth_max4(terms: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """
    Conservative, smooth approximation of max(terms) for exactly 4 terms.

        smoothmax_β(t) = LSE_β(t) − ln(4)/β
        LSE_β(t)        = m + (1/β)·ln( Σ exp(β·(t_i − m)) ),  m = max(t)

    Properties (exact, not approximate):
        max(t) − ln(4)/β  ≤  smoothmax_β(t)  ≤  max(t)
    so it NEVER overestimates the true max — the resulting barrier is
    always at least as conservative as the exact one. As β → ∞, this
    converges to max(t) (and the softmax weights below → one-hot on the
    argmax, recovering exact hard-max gradients).

    Returns
    -------
    value   : float — smoothmax_β(terms)
    weights : np.ndarray shape (4,) — softmax weights, sum to 1, used to
              blend the per-term gradients (see rectangle_barrier below).
    """
    m       = np.max(terms)
    shifted = beta * (terms - m)
    exps    = np.exp(shifted)
    denom   = np.sum(exps)
    weights = exps / denom
    value   = m + np.log(denom) / beta - np.log(4.0) / beta
    return float(value), weights


def circle_barrier(x1: float, x2: float, params: tuple) -> tuple[float, np.ndarray]:
    """
    h(x) = (x1−cx)² + (x2−cy)² − R²   — positive outside, negative inside.

    Returns
    -------
    value : float
    grad2 : np.ndarray shape (2,) — [∂h/∂x1, ∂h/∂x2]
    """
    cx, cy, radius = params
    value = (x1 - cx) ** 2 + (x2 - cy) ** 2 - radius ** 2
    grad2 = np.array([2.0 * (x1 - cx), 2.0 * (x2 - cy)])
    return value, grad2


def rectangle_barrier(
    x1: float, x2: float, params: tuple, beta: float,
) -> tuple[float, np.ndarray]:
    """
    h(x) = smoothmax_β(x1−x_max, x_min−x1, x2−y_max, y_min−x2)

    A conservative, C∞ approximation of the exact L∞ signed distance to
    the rectangle boundary — see smooth_max4() for why this particular
    form was chosen over a hard max().

    Returns
    -------
    value : float
    grad2 : np.ndarray shape (2,) — [∂h/∂x1, ∂h/∂x2]
    """
    x_min, y_min, x_max, y_max = params
    terms = np.array([
        x1 - x_max,    # positive ↔ right of right wall
        x_min - x1,   # positive ↔ left  of left  wall
        x2 - y_max,   # positive ↔ above top    wall
        y_min - x2,   # positive ↔ below bottom wall
    ])
    value, w = smooth_max4(terms, beta)
    grad2 = np.array([w[0] - w[1], w[2] - w[3]])
    return value, grad2


def position_barrier(
    state: np.ndarray, obs: dict, rect_beta: float,
) -> tuple[float, np.ndarray]:
    """
    Dispatch to circle_barrier / rectangle_barrier based on obs["type"],
    for any vehicle whose barrier depends only on state[0], state[1].

    Parameters
    ----------
    state     : shape (n,) — only state[0], state[1] are read
    obs       : dict with 'type' ('circle'|'rectangle') and 'params'
    rect_beta : smoothing sharpness passed to rectangle_barrier

    Returns
    -------
    value : float
    grad2 : np.ndarray shape (2,) — gradient w.r.t. (x1, x2) only.
            Caller is responsible for embedding this into the full
            n-dimensional state gradient (see h_obstacle_grad in each
            vehicle — grad[2:] = 0 for every planned vehicle here).
    """
    x1, x2 = float(state[0]), float(state[1])

    if obs["type"] == "circle":
        return circle_barrier(x1, x2, obs["params"])
    elif obs["type"] == "rectangle":
        return rectangle_barrier(x1, x2, obs["params"], rect_beta)

    # Unknown obstacle type — treat as no obstacle (safe), zero gradient
    return 1.0, np.zeros(2)
