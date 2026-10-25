"""
solver/clf.py — Vehicle-agnostic CLF constraint builder for SLSQP.

Only architectures 'i' and 'ii' are supported (arch 'iii' — hard CLF — was
dropped: it's infeasible at cold start for any vehicle starting from rest,
see the note that used to live in admm.py).

For each horizon timestep t (t = 0 … N-1), enforces the CLF decay condition:

    Arch 'ii'  (slack on CLF  — CBF is hard):
        V(x_{t+1}) ≤ (1 − λ)·V(x_t) + δ_t
        As SLSQP ineq (≥ 0):
            (1 − λ)·V(x_t) − V(x_{t+1}) + δ_t  ≥  0             (*)

    Arch 'i'   (no CLF at all):
        Returns an empty list — caller gets no CLF constraints.

Unlike h_obstacle (see cbf.py), V_clf includes the velocity states (v, ω)
directly, so V̇ depends on the controls [a, α] with no extra differentiation
needed — CLF is relative degree 1 here and doesn't need an HOCBF-style
chain the way the CBF does.

Design notes
------------
* The CLF constraint is returned as a single dict whose 'fun' returns an
  array of shape (N,) covering all timesteps — one SLSQP constraint object
  total, regardless of horizon length.
* The Jacobian is w.r.t. the FULL z vector [x_flat | u_flat | slack],
  matching the layout expected by cbf.py and admm.py.
* u-columns and CBF-slack columns are always 0 in this Jacobian.

Public API
----------
build_clf_constraints(
    vehicle, goal, lam, N, architecture,
    size_x, size_u, total_z, clf_slack_offset=0
) -> list[dict]   (length 0 for arch 'i', length 1 otherwise)
"""

import numpy as np


def build_clf_constraints(
    vehicle          ,
    goal             : np.ndarray,
    lam              : float,
    N                : int,
    architecture     : str,
    size_x           : int,
    size_u           : int,
    total_z          : int,
    clf_slack_offset : int = 0,
) -> list:
    """
    Build one SLSQP inequality-constraint dict covering all N timesteps.

    Parameters
    ----------
    vehicle          : BaseVehicle — must implement V_clf, V_clf_grad, n
    goal             : (x, y) target, shape (2,)
    lam              : CLF decay rate λ ∈ (0, 1)
                       larger → faster required convergence (harder constraint)
    N                : planning horizon
    architecture     : 'i'  → no CLF (returns [])
                       'ii' → CLF soft (slack δ_t on CLF side)
    size_x           : n * (N+1)
    size_u           : m * N
    total_z          : total length of z
    clf_slack_offset : first index in z of the CLF slack block
                       = size_x + size_u   for arch 'ii'
                       ignored              for arch 'i'

    Returns
    -------
    list of length 0 (arch 'i') or 1 (arch 'ii')
    """
    if architecture not in ('i', 'ii'):
        raise ValueError(f"architecture must be 'i' or 'ii'; got '{architecture}'")

    if architecture == 'i':
        return []

    # only arch 'ii' reaches here, and it always carries CLF slack
    n = vehicle.n

    def _fun(z: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (N,).
        Entry t = (1−λ)·V(x_t) − V(x_{t+1}) [+ δ_t]
        SLSQP requires every entry ≥ 0.
        """
        x_flat = z[:size_x]
        vals   = np.empty(N)

        for t in range(N):
            xt  = x_flat[ t      * n : (t + 1) * n]
            xt1 = x_flat[(t + 1) * n : (t + 2) * n]

            Vt  = vehicle.V_clf(xt,  goal)
            Vt1 = vehicle.V_clf(xt1, goal)

            # δ_t lives at clf_slack_offset + t
            vals[t] = (1.0 - lam) * Vt - Vt1 + z[clf_slack_offset + t]

        return vals

    def _jac(z: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (N, total_z).
        Row t is ∂(constraint_t)/∂z.

        Non-zero blocks per row t:
          columns  t*n    … (t+1)*n-1   : +(1−λ)·∇V(x_t)
          columns (t+1)*n … (t+2)*n-1   : −∇V(x_{t+1})
          column  clf_slack_offset + t  : +1
        u-columns and CBF-slack columns remain 0.
        """
        x_flat = z[:size_x]
        J = np.zeros((N, total_z))

        for t in range(N):
            xt  = x_flat[ t      * n : (t + 1) * n]
            xt1 = x_flat[(t + 1) * n : (t + 2) * n]

            gVt  = vehicle.V_clf_grad(xt,  goal)   # shape (n,)
            gVt1 = vehicle.V_clf_grad(xt1, goal)   # shape (n,)

            # ∂/∂x_t:    +(1−λ)·∇V(x_t)
            J[t,  t      * n : (t + 1) * n] += (1.0 - lam) * gVt
            # ∂/∂x_{t+1}: −∇V(x_{t+1})
            J[t, (t + 1) * n : (t + 2) * n] -= gVt1
            # ∂/∂δ_t: +1
            J[t, clf_slack_offset + t] = 1.0

        return J

    return [{'type': 'ineq', 'fun': _fun, 'jac': _jac}]
