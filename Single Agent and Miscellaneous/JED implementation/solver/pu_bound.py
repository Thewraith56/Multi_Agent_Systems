"""
solver/pu_bound.py — Theorem 5 (Eq. 13/14) finite-horizon exit-probability
bound, and Theorem 6's delta for the JED / LSE-relaxation barrier.

Reference: Cosner, Culbertson, Taylor, Ames (arXiv:2302.07469v2).
"""

from __future__ import annotations

import numpy as np

from . import config


def compute_M(R_sense: float, arena_bounds: tuple = config.WORLD_BOUNDS) -> float:
    """Re-exported for convenience; see config.compute_M's docstring."""
    return config.compute_M(R_sense, arena_bounds)


def compute_delta(lambda_max: float = 0.0, cov_trace: float = 0.0, c_J: float = 0.0) -> float:
    """
    Theorem 6:  delta = c_J - (lambda_max / 2) * tr(cov(d)).

    For the reactive multi-hyperplane polytope barrier, h(x) = -max_i(affine)
    is a minimum of affine functions -> concave with lambda_max = 0 EXACTLY
    (zero curvature everywhere it's differentiable). Because the LSE
    relaxation already directly bounds -E[h(x_{k+1})] (see
    barriers.lse_expected_barrier_constraint), no separate Jensen-gap
    correction is layered on top, so c_J = 0 as well.

    This function still takes lambda_max / cov_trace / c_J as explicit
    arguments (rather than hard-coding delta = 0) so the computation is
    auditable and so the tuning dashboard can show the arithmetic, per the
    "compute delta explicitly rather than assuming" requirement.
    """
    return float(c_J - 0.5 * lambda_max * cov_trace)


def pu_bound(alpha: float, delta: float, M: float, gamma: float, h0: float, K: int) -> float:
    """
    Theorem 5, Eq. (13) [Case 1: delta < -gamma*(1-alpha)] or
    Eq. (14) [Case 2: delta >= -gamma*(1-alpha)].

    Returns the certified upper bound on Pu(K, gamma, x0) = P(min_k h(xk) < -gamma).
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    if M <= 0:
        raise ValueError("M must be > 0")

    switch = -gamma * (1.0 - alpha)

    if delta < switch:
        # Case 1, Eq. (13)
        term1 = ((M - h0) / (M + gamma)) * (alpha ** K)
        if abs(alpha - 1.0) < 1e-12:
            geom_sum = float(K)
        else:
            geom_sum = sum(alpha ** i for i in range(K))   # sum_{i=1}^{K} alpha^{i-1}, i=1..K
        term2 = ((M * (1.0 - alpha) - delta) / (M + gamma)) * geom_sum
        bound = term1 + term2
    else:
        # Case 2, Eq. (14)
        ratio = (M * alpha + gamma + delta) / (M + gamma)
        bound = 1.0 - ((h0 + gamma) / (M + gamma)) * (ratio ** K)

    return float(np.clip(bound, 0.0, 1.0))


def pu_bound_sweep(alpha: float, delta: float, M: float, gammas, h0: float, K: int) -> np.ndarray:
    """Vectorized convenience wrapper for a sweep of gamma values (dashboard plot)."""
    return np.array([pu_bound(alpha, delta, M, float(g), h0, K) for g in gammas])


def which_case(delta: float, gamma: float, alpha: float) -> str:
    switch = -gamma * (1.0 - alpha)
    return "Case 1 (Eq. 13)" if delta < switch else "Case 2 (Eq. 14)"
