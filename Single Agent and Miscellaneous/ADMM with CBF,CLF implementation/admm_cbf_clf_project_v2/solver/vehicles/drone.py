"""
solver/vehicles/drone.py — 2-D double-integrator drone.

State   x = [x₁, x₂, vx, vy]     n = 4
Control u = [ax, ay]              m = 2

Euler dynamics
--------------
    x₁_{t+1} = x₁_t + vx_t · dt
    x₂_{t+1} = x₂_t + vy_t · dt
    vx_{t+1} = vx_t + ax_t · dt
    vy_{t+1} = vy_t + ay_t · dt



Relative degree note (see solver/cbf.py and base.py for the full HOCBF
derivation)
-----------------------------------------------------------------------
h depends only on position. Differentiating along the dynamics:

    ḣ  = ∂h/∂x₁·vx + ∂h/∂x₂·vy                  — depends on states (vx,vy),
                                                    not on control at all
    ḧ  = ∂h/∂x₁·ax + ∂h/∂x₂·ay + [state terms]   — control appears here

Both control channels (ax, ay) enter ḧ SYMMETRICALLY — unlike the unicycle
(where α only reaches position through a 3-integrator chain, one order
higher than a), the drone's translational controls both sit at exactly
relative degree 2. That means cbf.py's existing 2nd-order HOCBF chain
(ψ_0, ψ_1, ψ_2) is already the exact right order for BOTH controls here —
nothing about cbf.py needs to change or be extended for this vehicle.

CBF barrier
-----------
Identical math to unicycle.py, since h depends only on (x₁,x₂) for every
vehicle in this codebase — delegated to geometry.py, not re-derived here.

CLF Lyapunov function
----------------------
    V(x) = (x₁−gx)² + (x₂−gy)² + K_V·(vx² + vy²)

    V = 0  iff  at goal position AND stopped (vx = vy = 0).
    Unlike the unicycle, there's no heading to exclude — a drone has no
    orientation state at all in this simple model.
"""

from __future__ import annotations

import numpy as np
from .base import BaseVehicle
from ..geometry import position_barrier


class DroneVehicle(BaseVehicle):
    """
    Concrete BaseVehicle implementation for a 2-D double-integrator drone.

    All methods follow the interface documented in base.py exactly.
    Circle / smoothed-rectangle barrier math lives in geometry.py, shared
    with unicycle.py.
    """

    # ── Dimensions ────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return 4

    @property
    def m(self) -> int:
        return 2

    # ── Cost weight vectors ───────────────────────────────────────────

    @property
    def stage_weights(self) -> np.ndarray:
        """
        Position-only cost at every stage timestep — same rationale as
        the unicycle: velocity is left free en-route, the dynamics
        constraint keeps it physically consistent, and the CBF/CLF (not
        the running cost) are what actually shape the trajectory.
        """
        return np.array([1.0, 1.0, 0.0, 0.0])

    @property
    def terminal_weights(self) -> np.ndarray:
        """
        At the final timestep, also penalise residual velocity so the
        drone comes to rest at the goal (arch 'i' only — terminal_scale
        is auto-zeroed for arch 'ii', where the CLF drives stopping).
        """
        return np.array([1.0, 1.0, 0.3, 0.3])

    # ── CLF tuning constant ─────────────────────────────────────────────
    #   Weight on (vx²+vy²) in the Lyapunov function.
    #   Larger  →  stronger "stop at goal" requirement in the CLF.
    _K_V : float = 0.5

    # ── Rectangle barrier smoothing ────────────────────────────────────
    #   See unicycle.py's identical constant for the full trade-off note.
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
        """One forward Euler step for the double integrator."""
        x1, x2, vx, vy = state
        ax, ay          = ctrl
        return np.array([
            x1 + vx * dt,
            x2 + vy * dt,
            vx + ax * dt,
            vy + ay * dt,
        ])

    def dynamics_jac(
        self,
        state : np.ndarray,
        ctrl  : np.ndarray,
        dt    : float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytic Jacobians of dynamics_step — constant (state-independent)
        since the dynamics are linear. Simpler than the unicycle's, which
        has trig cross-terms because heading couples x1/x2 to v.

        Jx[i,j] = ∂(next_state[i]) / ∂(state[j])
        Ju[i,j] = ∂(next_state[i]) / ∂(ctrl[j])
        """
        Jx = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        Ju = np.array([
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
        Barrier function h(x) — identical math to unicycle.py's, since h
        depends only on (x1,x2) for every vehicle here. See geometry.py
        for the full circle / smoothed-rectangle derivation.
        """
        value, _ = position_barrier(state, obs, self._RECT_SMOOTH_BETA)
        return value

    def h_obstacle_grad(self, state: np.ndarray, obs: dict) -> np.ndarray:
        """
        Gradient of h_obstacle w.r.t. state, shape (4,).
        grad[0], grad[1] from geometry.position_barrier; grad[2:] = 0
        (h doesn't depend on vx, vy).
        """
        grad = np.zeros(self.n)
        _, grad[:2] = position_barrier(state, obs, self._RECT_SMOOTH_BETA)
        return grad

    # ═══════════════════════════════════════════════════════════════
    #  CLF Lyapunov function
    # ═══════════════════════════════════════════════════════════════

    def V_clf(self, state: np.ndarray, goal: np.ndarray) -> float:
        """
        V(x) = (x1−gx)² + (x2−gy)² + K_V·(vx² + vy²)

        V = 0  iff  position = goal  AND  vx = vy = 0.
        Relative degree 1 w.r.t. both controls (ax, ay appear directly in
        V̇ through the vx², vy² terms) — no HOCBF-style chain needed here,
        same as the unicycle's V_clf. See clf.py's module docstring for
        why CLF never needs the HOCBF treatment CBF does.
        """
        x1, x2, vx, vy = state
        gx, gy = goal
        return (x1 - gx) ** 2 + (x2 - gy) ** 2 + self._K_V * (vx ** 2 + vy ** 2)

    def V_clf_grad(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        ∂V/∂x = [2(x1−gx), 2(x2−gy), 2·K_V·vx, 2·K_V·vy]
        """
        x1, x2, vx, vy = state
        gx, gy = goal
        return np.array([
            2.0 * (x1 - gx),
            2.0 * (x2 - gy),
            2.0 * self._K_V * vx,
            2.0 * self._K_V * vy,
        ])

    # ═══════════════════════════════════════════════════════════════
    #  State / target construction
    # ═══════════════════════════════════════════════════════════════

    def initial_state(self, scene_start: np.ndarray) -> np.ndarray:
        """
        The scene editor always stores start as shape-(5,):
            [x, y, theta, v=0.0, omega=0.0]

        The drone has no heading or angular-rate state, so theta and
        omega are simply dropped; it always starts at rest
        (vx=0, vy=0 — the scene editor never sets a nonzero start speed
        anyway).
        """
        return np.array([
            float(scene_start[0]),
            float(scene_start[1]),
            0.0,   # vx = 0 at start
            0.0,   # vy = 0 at start
        ])

    def goal_target(self, goal_xy: np.ndarray, N: int) -> np.ndarray:
        """
        Flat desired trajectory: drone at (gx, gy), stopped, for all
        N+1 steps.

        r_target = tile([gx, gy, 0, 0], N+1)   shape (4*(N+1),)
        """
        target_state = np.array([
            float(goal_xy[0]),
            float(goal_xy[1]),
            0.0,    # vx target — drone must stop
            0.0,    # vy target — drone must stop
        ])
        return np.tile(target_state, N + 1)
