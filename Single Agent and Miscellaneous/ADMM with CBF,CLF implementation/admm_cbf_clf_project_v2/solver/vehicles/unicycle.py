"""
solver/vehicles/unicycle.py — Acceleration-controlled dynamic unicycle.

State   x = [x₁, x₂, θ, v, ω]     n = 5
Control u = [a,  α]                 m = 2

Euler dynamics
--------------
    x₁_{t+1} = x₁_t + v_t · cos(θ_t) · dt
    x₂_{t+1} = x₂_t + v_t · sin(θ_t) · dt
    θ_{t+1}  = θ_t  + ω_t             · dt
    v_{t+1}  = v_t  + a_t             · dt
    ω_{t+1}  = ω_t  + α_t             · dt

CBF barrier
-----------
Circle   (cx, cy, R):
    h(x) = (x₁−cx)² + (x₂−cy)² − R²
    Positive outside, negative inside, zero on boundary.

Rectangle (x_min, y_min, x_max, y_max):
    h(x) = smoothmax_β(x₁−x_max, x_min−x₁, x₂−y_max, y_min−x₂)
    where smoothmax_β is a conservative (≤ true max), C∞ log-sum-exp
    approximation of the exact L∞ signed distance — see _smooth_max4()
    below. Replaces the old hard max(): at the exact hard boundary and at
    corners, max() has zero or discontinuous gradient, which starves the
    HOCBF chain (cbf.py) of the signal it needs right when it matters
    most. The smooth version is always ≤ the true max, so the safe set
    {h ≥ 0} it defines is always a (slightly rounded-corner) subset of
    the true safe set — never larger, so no new unsafe states are
    admitted.

CLF Lyapunov function
---------------------
    V(x) = (x₁−gx)² + (x₂−gy)² + K_V·v² + K_ω·ω²

    V = 0  iff  at goal position AND stopped (v = 0, ω = 0).
    Gradient is analytic and nonzero everywhere except the goal.
    θ is not included — multiple headings are acceptable at goal.
"""

from __future__ import annotations

import numpy as np
from .base import BaseVehicle
from ..geometry import position_barrier


class UnicycleVehicle(BaseVehicle):
    """
    Concrete BaseVehicle implementation for the dynamic unicycle.

    All methods follow the interface documented in base.py exactly.
    This is one of the vehicle-specific files needed to run the full
    ADMM + CBF / CLF solver stack (see also drone.py) for arch 'i' and 'ii'.
    Circle / smoothed-rectangle barrier math lives in geometry.py, shared
    across every vehicle whose h_obstacle depends only on position.
    """

    # ── Dimensions ────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return 5

    @property
    def m(self) -> int:
        return 2

    # ── Cost weight vectors ───────────────────────────────────────────

    @property
    def stage_weights(self) -> np.ndarray:
        """
        Position-only cost at every stage timestep.
        Heading, speed, and angular rate are not penalised en-route;
        the dynamics constraint keeps them physically consistent.
        """
        return np.array([1.0, 1.0, 0.0, 0.0, 0.0])

    @property
    def terminal_weights(self) -> np.ndarray:
        """
        At the final timestep, also penalise residual speed and
        angular rate so the robot stops at the goal.

        Multiplied by params['terminal_scale'] (default 80.0 for arch 'i',
        auto-zeroed for arch 'ii' / 'iii' where CLF drives stopping).
        """
        return np.array([1.0, 1.0, 0.0, 0.3, 0.3])

    # ── CLF tuning constants ──────────────────────────────────────────
    #   Weight on v² and ω² in the Lyapunov function.
    #   Larger  →  stronger "stop at goal" requirement in the CLF.
    #   Smaller →  CLF mainly cares about position, looser on velocity.
    _K_V : float = 0.5
    _K_W : float = 0.5

    # ── Rectangle barrier smoothing ────────────────────────────────────
    #   Sharpness β of the smooth-max used for h_obstacle on rectangles.
    #   Larger  →  closer to the exact hard L∞ boundary, corners rounded
    #              by less (~ln(4)/β metres), but more nonlinear — harder
    #              on the gradient-based solver near corners.
    #   Smaller →  smoother, easier to optimise through, but rounds the
    #              rectangle corners off by more (conservatively, so
    #              still always safe — never lets the robot get closer
    #              than the true boundary).
    #   10.0 rounds corners by ln(4)/10 ≈ 0.14 m, a good default for a
    #   scene on the order of metres.
    _RECT_SMOOTH_BETA : float = 10.0

    # ═══════════════════════════════════════════════════════════════
    #  Dynamics
    # ═══════════════════════════════════════════════════════════════

    def dynamics_step(
        self,
        state : np.ndarray,
        ctrl  : np.ndarray,
        dt    : float,
    ) -> np.ndarray:
        """One forward Euler step for the dynamic unicycle."""
        x1, x2, theta, v, omega = state
        a, alpha_acc             = ctrl
        return np.array([
            x1    + v * np.cos(theta) * dt,
            x2    + v * np.sin(theta) * dt,
            theta + omega             * dt,
            v     + a                 * dt,
            omega + alpha_acc         * dt,
        ])

    def dynamics_jac(
        self,
        state : np.ndarray,
        ctrl  : np.ndarray,
        dt    : float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytic Jacobians of dynamics_step.

        Jx[i,j] = ∂(next_state[i]) / ∂(state[j])
        Ju[i,j] = ∂(next_state[i]) / ∂(ctrl[j])

        Nonzero entries:
          Jx — diagonal is 1 (identity part of Euler)
               plus the trig terms from x₁, x₂ row derivatives
          Ju — only v row gets dt from a, omega row gets dt from alpha
        """
        x1, x2, theta, v, omega = state

        Jx = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt,  np.cos(theta) * dt, 0.0],
            [0.0, 1.0,  v * np.cos(theta) * dt,  np.sin(theta) * dt, 0.0],
            [0.0, 0.0,  1.0,                      0.0,                dt ],
            [0.0, 0.0,  0.0,                      1.0,                0.0],
            [0.0, 0.0,  0.0,                      0.0,                1.0],
        ])

        Ju = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [dt,  0.0],
            [0.0, dt ],
        ])

        return Jx, Ju

    # ═══════════════════════════════════════════════════════════════
    #  CBF barrier function
    # ═══════════════════════════════════════════════════════════════

    def h_obstacle(self, state: np.ndarray, obs: dict) -> float:
        """
        Barrier function h(x):
          h > 0  →  safe (outside obstacle)
          h = 0  →  on boundary
          h < 0  →  unsafe (inside obstacle)

        Circle  : h = (x₁−cx)² + (x₂−cy)² − R²
        Rectangle: h = max(x₁−x_max, x_min−x₁, x₂−y_max, y_min−x₂)

        Rectangle notes
        ---------------
        The four terms are the signed distances to the right, left, top,
        and bottom faces respectively.  Their (smooth) max is > 0 iff the
        robot is outside at least one face (i.e., outside the rectangle).
        A hard max() would give the exact L∞ signed-distance, but is flat
        (zero gradient) away from the active face and kinked at corners —
        see geometry.smooth_max4() for the conservative smoothed version
        used here, which fixes both problems at the cost of slightly
        rounding the corners inward (never outward, so still a valid
        barrier).

        Circle / rectangle math itself lives in geometry.py — shared by
        every vehicle whose barrier depends only on position, see there
        for the full derivation.
        """
        value, _ = position_barrier(state, obs, self._RECT_SMOOTH_BETA)
        return value

    def h_obstacle_grad(self, state: np.ndarray, obs: dict) -> np.ndarray:
        """
        Gradient of h_obstacle w.r.t. state, shape (5,).

        Only grad[0] (∂h/∂x₁) and grad[1] (∂h/∂x₂) are ever nonzero;
        grad[2:] = 0 because h depends only on position. The (x1,x2)
        gradient itself is computed by geometry.position_barrier — see
        geometry.py for the circle / smoothed-rectangle derivation.
        """
        grad = np.zeros(self.n)
        _, grad[:2] = position_barrier(state, obs, self._RECT_SMOOTH_BETA)
        return grad

    # ═══════════════════════════════════════════════════════════════
    #  CLF Lyapunov function
    # ═══════════════════════════════════════════════════════════════

    def V_clf(self, state: np.ndarray, goal: np.ndarray) -> float:
        """
        V(x) = (x₁−gx)² + (x₂−gy)² + K_V·v² + K_ω·ω²

        V = 0  iff  position = goal  AND  v = 0  AND  ω = 0.

        θ is not included because the robot reaching the goal in any
        heading is acceptable (only position + stop matters).
        """
        x1, x2, _theta, v, omega = state
        gx, gy = goal
        return (
            (x1 - gx) ** 2
            + (x2 - gy) ** 2
            + self._K_V * v     ** 2
            + self._K_W * omega ** 2
        )

    def V_clf_grad(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        ∂V/∂x = [2(x₁−gx),  2(x₂−gy),  0,  2·K_V·v,  2·K_ω·ω]
        """
        x1, x2, _theta, v, omega = state
        gx, gy = goal
        return np.array([
            2.0 * (x1 - gx),
            2.0 * (x2 - gy),
            0.0,
            2.0 * self._K_V * v,
            2.0 * self._K_W * omega,
        ])

    # ═══════════════════════════════════════════════════════════════
    #  State / target construction
    # ═══════════════════════════════════════════════════════════════

    def initial_state(self, scene_start: np.ndarray) -> np.ndarray:
        """
        The scene editor already stores start as shape-(5,):
            [x, y, theta, v=0.0, omega=0.0]
        which is exactly the unicycle state vector — return it directly.
        """
        return scene_start.astype(float).copy()

    def goal_target(self, goal_xy: np.ndarray, N: int) -> np.ndarray:
        """
        Flat desired trajectory: robot at (gx, gy), stopped, for all N+1 steps.

        r_target = tile([gx, gy, 0, 0, 0], N+1)   shape (5*(N+1),)

        theta target = 0 (arbitrary — heading cost weight is 0 so it
        has no effect; only position and velocity columns matter).
        """
        target_state = np.array([
            float(goal_xy[0]),
            float(goal_xy[1]),
            0.0,    # theta target (cost weight = 0, no effect)
            0.0,    # v target  — robot must stop
            0.0,    # ω target  — robot must stop rotating
        ])
        return np.tile(target_state, N + 1)
