"""
solver/vehicles/drone.py — 2-D double-integrator drone.

State   x = [x1, x2, vx, vy]     n = 4
Control u = [ax, ay]              m = 2

CHANGED from the ADMM-era drone.py:
  - V_clf / V_clf_grad REMOVED. Goal-reaching is now the LQR waypoint
    tracker's job (solver/tracker.py), which regulates directly to
    x_ref = [gx, gy, 0, 0] via u = -K(x - x_ref) — no Lyapunov function
    needed or used anywhere in this pipeline.
  - h_obstacle / h_obstacle_grad REMOVED. Safety is now the reactive
    multi-hyperplane polytope DTCBF built fresh every timestep directly
    from obstacle geometry (solver/barriers.py) — it only reads
    state[:2], so there is nothing vehicle-specific left to implement here.
  - dynamics_step / dynamics_jac REPLACED by a single step() that uses the
    constant A, B from dynamics.build_AB() (the double integrator is
    linear, so the "Jacobian" is just A, B themselves — no separate method
    needed).
  - goal_target REMOVED — no MPC/ADMM reference trajectory to build.

Everything else (dimensions, initial_state) is unchanged in spirit.
"""

from __future__ import annotations

import numpy as np

from .base import BaseVehicle
from ..dynamics import build_AB
from .. import config


class DroneVehicle(BaseVehicle):
    """Concrete BaseVehicle for a 2-D double-integrator drone."""

    def __init__(self, dt: float = config.DT) -> None:
        self.dt = dt
        self.A, self.B = build_AB(dt)

    # ── Dimensions ────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return 4

    @property
    def m(self) -> int:
        return 2

    # ═══════════════════════════════════════════════════════════════
    #  Dynamics
    # ═══════════════════════════════════════════════════════════════

    def step(self, x: np.ndarray, u: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        """x_{k+1} = A x_k + B u_k (+ d_k)."""
        x_next = self.A @ x + self.B @ u
        if d is not None:
            x_next = x_next + d
        return x_next

    # ═══════════════════════════════════════════════════════════════
    #  State construction
    # ═══════════════════════════════════════════════════════════════

    def initial_state(self, scene_start: np.ndarray) -> np.ndarray:
        """
        scene_start = [x, y, theta, v=0.0, omega=0.0]. The drone has no
        heading or angular-rate state, so theta/omega are dropped; it
        always starts at rest (vx=0, vy=0).
        """
        return np.array([
            float(scene_start[0]),
            float(scene_start[1]),
            0.0,   # vx = 0 at start
            0.0,   # vy = 0 at start
        ])

    def position(self, x: np.ndarray) -> np.ndarray:
        return x[:2]

    def velocity(self, x: np.ndarray) -> np.ndarray:
        return x[2:4]
