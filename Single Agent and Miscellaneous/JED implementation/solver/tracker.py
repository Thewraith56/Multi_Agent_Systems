"""
solver/tracker.py — LQR waypoint tracker (layer 2 of the 3-layer architecture).

Pure state-feedback tracking toward the CURRENT waypoint from planner.py.
Has no knowledge of obstacles beyond what's implicit in the waypoints A*
already chose; its only interface to the JED safety filter is the resulting
u_nom vector (jed_solver.py treats it as a black box).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from . import config


def design_lqr_gain(A: np.ndarray, B: np.ndarray,
                     Q: np.ndarray = config.LQR_Q,
                     R: np.ndarray = config.LQR_R) -> np.ndarray:
    """Solve the discrete algebraic Riccati equation and return gain K
    such that u = -K @ (x - x_ref) is the infinite-horizon LQR regulator."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


class WaypointTracker:
    """
    Holds an ordered waypoint list + current index + LQR gain K.
    .get_u_nom(x_k) returns u_nom = -K (x_k - x_ref_i) and internally
    advances the waypoint index once within switch_radius of g_i.
    """

    def __init__(
        self,
        waypoints: list[np.ndarray],
        K: np.ndarray,
        switch_radius: float = config.SWITCH_RADIUS,
        goal_tolerance: float = config.GOAL_TOLERANCE,
        goal_speed_tol: float = config.GOAL_SPEED_TOL,
    ) -> None:
        if not waypoints:
            raise ValueError("WaypointTracker needs at least one waypoint")
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.K = K
        self.switch_radius = switch_radius
        self.goal_tolerance = goal_tolerance
        self.goal_speed_tol = goal_speed_tol
        self.idx = 0

    @property
    def current_waypoint(self) -> np.ndarray:
        return self.waypoints[self.idx]

    @property
    def at_final_waypoint(self) -> bool:
        return self.idx == len(self.waypoints) - 1

    def reached_goal(self, x_k: np.ndarray) -> bool:
        pos_err = np.linalg.norm(x_k[:2] - self.waypoints[-1])
        speed = np.linalg.norm(x_k[2:4])
        return self.at_final_waypoint and pos_err < self.goal_tolerance and speed < self.goal_speed_tol

    def get_u_nom(self, x_k: np.ndarray) -> np.ndarray:
        pos_err = np.linalg.norm(x_k[:2] - self.current_waypoint)
        if pos_err < self.switch_radius and not self.at_final_waypoint:
            self.idx += 1

        x_ref = np.array([self.current_waypoint[0], self.current_waypoint[1], 0.0, 0.0])
        u_nom = -self.K @ (x_k - x_ref)
        return u_nom
