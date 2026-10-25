"""
solver/cbf.py — Vehicle-agnostic HOCBF constraint builder for SLSQP.

Why HOCBF (and not the plain first-order CBF)
----------------------------------------------
h_i depends only on position (x1, x2). For an acceleration-controlled
vehicle (unicycle: state [x1,x2,θ,v,ω], control [a,α]) the control never
appears in ḣ — it only shows up after a *second* differentiation, through
v̇ = a. That means h has relative degree 2 w.r.t. the control, and a plain
one-step condition "h(x_{t+1}) − (1−γ)h(x_t) ≥ 0" has no real safety
invariance guarantee: it penalises the raw change in h but the control
can't actually shape that change directly (v, ω are states, not commands).
This shows up in practice as weak/late gradients right at the obstacle
boundary — exactly the failure mode noted in the project's limitations
write-up.

HOCBF (Xiao & Belta, discretised here) fixes this by building a short
chain of barrier functions instead of one:

    ψ_0(x)     = h(x)
    ψ_1(x_t)   = h(x_{t+1})   − (1−g1)·h(x_t)          (velocity-level CBF)
    ψ_2(x_t)   = ψ_1(x_{t+1}) − (1−g2)·ψ_1(x_t)          (accel-level CBF —
                                                            control finally
                                                            appears here)

Enforcing ψ_2(x_t) ≥ 0 for every t, together with the boundary condition
ψ_1(x_0) ≥ 0, guarantees h(x_t) ≥ 0 for the whole horizon (given the start
state is itself safe, h(x_0) ≥ 0). Expanded in terms of h alone:

    ψ_2(x_t) = h(x_{t+2}) − [(1−g1)+(1−g2)]·h(x_{t+1}) + (1−g1)(1−g2)·h(x_t)

g1, g2 ∈ (0,1) are two independent decay rates ("velocity-level" and
"accel-level"). Passing a single gamma (g1 == g2) recovers a symmetric
HOCBF and is a reasonable default; splitting them lets you tune how
aggressively each stage of the chain is allowed to decay.

Row layout (unchanged slack bookkeeping)
-----------------------------------------
Each obstacle still contributes exactly N rows — same as the old
first-order builder — so cbf_slack_offset / n_slack_cbf in admm.py don't
need to change:

    row 0        : ψ_1(x_0) = h(x_1) − (1−g1)·h(x_0)  ≥ 0   (boundary term)
    row t=1..N-1 : ψ_2(x_{t-1})  using the triple (x_{t-1}, x_t, x_{t+1})

The Jacobian is always w.r.t. the FULL z vector
    z = [x_flat | u_flat | slack]
so SLSQP can couple it correctly with the dynamics equality constraint.

Public API
----------
build_cbf_constraints(
    vehicle, obstacles, gamma1, gamma2, N, architecture,
    size_x, size_u, total_z, cbf_slack_offset=0
) -> list[dict]

Each dict in the returned list is one SLSQP-compatible constraint covering
all N rows for one obstacle (vectorised):
    { 'type': 'ineq', 'fun': f, 'jac': J }
where f(z) has shape (N,) and J(z) has shape (N, total_z).
"""

import numpy as np


def build_cbf_constraints(
    vehicle          ,
    obstacles        : list,
    gamma1           : float,
    gamma2           : float,
    N                : int,
    architecture     : str,
    size_x           : int,
    size_u           : int,
    total_z          : int,
    cbf_slack_offset : int = 0,
) -> list:
    """
    Build one SLSQP inequality-constraint dict per obstacle, implementing
    the discrete second-order HOCBF chain described in the module
    docstring.

    Parameters
    ----------
    vehicle          : BaseVehicle — must implement h_obstacle, h_obstacle_grad, n
    obstacles        : list of obstacle dicts from scene_editor
                       each has 'type' ('circle'|'rectangle') and 'params'
    gamma1           : ψ_1 decay rate g1 ∈ (0, 1)  — "velocity-level" CBF
    gamma2           : ψ_2 decay rate g2 ∈ (0, 1)  — "accel-level" CBF
                       smaller → more conservative (slower decay allowed)
    N                : planning horizon
    architecture     : 'i'  → slack on CBF (soft safety)
                       'ii' → CBF is hard   (no slack in this builder)
    size_x           : n * (N+1)
    size_u           : m * N
    total_z          : total length of z = size_x + size_u + n_slack
    cbf_slack_offset : first index in z of the CBF slack block
                       = size_x + size_u   for arch 'i'
                       ignored              for arch 'ii'

    Returns
    -------
    list of dicts, length = len(obstacles)
    Empty list if obstacles is empty (no constraints needed).
    """
    n             = vehicle.n
    has_cbf_slack = (architecture == 'i')
    c1            = 1.0 - gamma1
    c2            = 1.0 - gamma2
    constraints   = []

    for obs_idx, obs in enumerate(obstacles):

        # ── close over obs_idx and obs so each lambda is independent ──
        def _make_fun(oi: int, obs_dict: dict):
            def fun(z: np.ndarray) -> np.ndarray:
                """
                Returns array of shape (N,).
                Row 0     = ψ_1(x_0) = h(x_1) − (1−g1)·h(x_0)
                Row t>=1  = ψ_2(x_{t-1})
                          = h(x_{t+1}) − [(1−g1)+(1−g2)]·h(x_t)
                                       + (1−g1)(1−g2)·h(x_{t-1})
                [+ δ_{t,i}] added on every row when architecture has CBF slack.
                SLSQP requires this to be ≥ 0 at every entry.
                """
                x_flat = z[:size_x]
                vals   = np.empty(N)

                h = np.empty(N + 1)
                for t in range(N + 1):
                    h[t] = vehicle.h_obstacle(x_flat[t * n:(t + 1) * n], obs_dict)

                # row 0 — boundary condition ψ_1(x_0)
                vals[0] = h[1] - c1 * h[0]

                # rows 1..N-1 — running condition ψ_2(x_{t-1})
                for t in range(1, N):
                    vals[t] = h[t + 1] - (c1 + c2) * h[t] + c1 * c2 * h[t - 1]

                if has_cbf_slack:
                    # δ_{t,i} lives at cbf_slack_offset + i*N + t
                    vals += z[cbf_slack_offset + oi * N: cbf_slack_offset + oi * N + N]

                return vals
            return fun

        def _make_jac(oi: int, obs_dict: dict):
            def jac(z: np.ndarray) -> np.ndarray:
                """
                Returns array of shape (N, total_z).

                Row 0 (boundary, ψ_1(x_0)):
                  columns  0     … n-1        : −(1−g1)·∇h(x_0)
                  columns  n     … 2n-1        : +∇h(x_1)

                Row t=1..N-1 (running, ψ_2(x_{t-1})):
                  columns (t-1)*n … t*n-1      : +(1−g1)(1−g2)·∇h(x_{t-1})
                  columns  t*n    … (t+1)*n-1  : −[(1−g1)+(1−g2)]·∇h(x_t)
                  columns (t+1)*n … (t+2)*n-1  : +∇h(x_{t+1})

                Every row also gets a +1 at column
                  cbf_slack_offset + i*N + t   (if has_cbf_slack)

                u-columns and other slack columns remain 0.
                """
                x_flat = z[:size_x]
                J = np.zeros((N, total_z))

                gh = [None] * (N + 1)
                for t in range(N + 1):
                    gh[t] = vehicle.h_obstacle_grad(x_flat[t * n:(t + 1) * n], obs_dict)

                # row 0 — ∂ψ_1(x_0)/∂x
                J[0, 0    : n    ] -= c1 * gh[0]
                J[0, n    : 2 * n] += gh[1]

                # rows 1..N-1 — ∂ψ_2(x_{t-1})/∂x
                for t in range(1, N):
                    J[t, (t - 1) * n: t       * n] += c1 * c2       * gh[t - 1]
                    J[t,  t      * n: (t + 1) * n] -= (c1 + c2)     * gh[t]
                    J[t, (t + 1) * n: (t + 2) * n] += gh[t + 1]

                if has_cbf_slack:
                    for t in range(N):
                        J[t, cbf_slack_offset + oi * N + t] = 1.0

                return J
            return jac

        constraints.append({
            'type' : 'ineq',
            'fun'  : _make_fun(obs_idx, obs),
            'jac'  : _make_jac(obs_idx, obs),
        })

    return constraints
