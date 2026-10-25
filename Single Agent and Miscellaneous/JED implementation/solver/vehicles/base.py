"""
solver/vehicles/base.py — BaseVehicle ABC for the JED/DTCBF architecture.

CHANGED from the ADMM-era base.py:
  - No CLF (V_clf / V_clf_grad) — goal-reaching is now handled entirely by
    the A* + LQR layers (planner.py / tracker.py), which are independent of
    and unaware of the safety filter. There is no Lyapunov function anywhere
    in this pipeline.
  - No per-vehicle h_obstacle / h_obstacle_grad — the DTCBF barrier is now
    the reactive multi-hyperplane polytope built fresh every timestep from
    obstacle geometry directly (solver/barriers.py), not a smooth per-vehicle
    barrier evaluated via autodiff-style gradients. It only ever reads
    state[:2] (position), so it does not need a vehicle-specific method.
  - No goal_target — there is no MPC/ADMM running cost to build a reference
    trajectory for; the LQR tracker regulates directly to each waypoint.
  - dynamics_jac is dropped too: this codebase is drone-only and the double
    integrator is linear, so dynamics.build_AB() already gives the exact
    (constant) Jacobians directly — a separate per-state Jacobian method
    would just be A, B repeated.

What remains is exactly what the new pipeline actually needs: dimensions,
one noisy forward step, and how to turn the scene editor's start vector into
this vehicle's state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVehicle(ABC):

    # ═══════════════════════════════════════════════════════════════
    #  Dimensions
    # ═══════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def n(self) -> int:
        """State dimension."""

    @property
    @abstractmethod
    def m(self) -> int:
        """Control dimension."""

    # ═══════════════════════════════════════════════════════════════
    #  Dynamics
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        One discrete step:  x_{k+1} = A x_k + B u_k (+ d_k).

        Parameters
        ----------
        x : current state, shape (n,)
        u : control input,  shape (m,)
        d : optional additive disturbance, shape (n,)

        Returns
        -------
        next_state : shape (n,)
        """

    # ═══════════════════════════════════════════════════════════════
    #  State construction helpers
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def initial_state(self, scene_start: np.ndarray) -> np.ndarray:
        """
        Convert the scene editor's start vector to this vehicle's state.

        The scene editor always stores start as a shape-(5,) array:
            [x, y, theta, v=0.0, omega=0.0]
        (Shared scene editor across vehicle types; each vehicle takes what
        it needs and ignores the rest — the drone ignores theta/omega.)

        Parameters
        ----------
        scene_start : shape (5,) — [x, y, theta, v, omega]

        Returns
        -------
        x0 : shape (n,)
        """

    @abstractmethod
    def position(self, x: np.ndarray) -> np.ndarray:
        """Extract the (x, y) position from a full state vector, shape (2,)."""

    @abstractmethod
    def velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract the (vx, vy) velocity from a full state vector, shape (2,)."""
