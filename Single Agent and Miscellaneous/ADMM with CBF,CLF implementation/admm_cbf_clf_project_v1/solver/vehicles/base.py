
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
    #  Cost weight vectors
    #  (used by costs.build_weight_vector inside admm.solver)
    # ═══════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def stage_weights(self) -> np.ndarray:
        """
        Shape (n,).

        Per-state cost weights applied at every stage timestep t = 0 … N-1.

        Typical choice:
          position states   (x, y)         → 1.0
          heading / velocity / rate states  → 0.0  (don't penalise en-route)
        """

    @property
    @abstractmethod
    def terminal_weights(self) -> np.ndarray:
        """
        Shape (n,).

        Per-state cost weights applied at the terminal timestep t = N.
        Multiplied by params['terminal_scale'] in the ADMM loop.

        Typical choice:
          position + velocity states        → 1.0  (stop at goal)
          heading                           → 0.0 or small


        """

    # ═══════════════════════════════════════════════════════════════
    #  Dynamics
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def dynamics_step(
        self,
        state : np.ndarray,
        ctrl  : np.ndarray,
        dt    : float,
    ) -> np.ndarray:
        """
        One forward Euler integration step.

        Computes:
            next_state = state + f(state, ctrl) · dt

        Parameters
        ----------
        state : current state, shape (n,)
        ctrl  : control input,  shape (m,)
        dt    : timestep in seconds

        Returns
        -------
        next_state : shape (n,)
        """

    @abstractmethod
    def dynamics_jac(
        self,
        state : np.ndarray,
        ctrl  : np.ndarray,
        dt    : float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytic Jacobians of dynamics_step w.r.t. state and ctrl.

        Used by the dynamics equality constraint in admm._dynamics_constraint()
        to provide exact gradients to SLSQP — avoids finite-difference noise
        and is significantly faster for large N.

        Returns
        -------
        Jx : shape (n, n)   ∂(next_state)/∂state
        Ju : shape (n, m)   ∂(next_state)/∂ctrl
        """

    # ═══════════════════════════════════════════════════════════════
    #  CBF barrier function
    #  (called by cbf.build_cbf_constraints at every ADMM iteration)
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def h_obstacle(
        self,
        state : np.ndarray,
        obs   : dict,
    ) -> float:
        """
        Signed barrier function for one obstacle.

        The CBF safe set is  S_i = { x : h_i(x) ≥ 0 }.
            h_i(x) > 0  →  state is outside the obstacle  (safe)
            h_i(x) = 0  →  state is on the boundary
            h_i(x) < 0  →  state is inside the obstacle   (unsafe)

        h depends only on position components (state[0], state[1]).
        All other state components are ignored.


        Parameters
        ----------
        state : shape (n,)
        obs   : dict with keys
                  'type'   : 'circle' | 'rectangle'
                  'params' : (cx, cy, r)  or  (x_min, y_min, x_max, y_max)

        Returns
        -------
        float

        
        """

    @abstractmethod
    def h_obstacle_grad(
        self,
        state : np.ndarray,
        obs   : dict,
    ) -> np.ndarray:
        """
        Gradient of h_obstacle w.r.t. state.

        Because h depends only on (state[0], state[1]):
          grad[0] = ∂h/∂x
          grad[1] = ∂h/∂y
          grad[2:] = 0   (for all heading / velocity / angular-rate states)

        Returns
        -------
        grad : shape (n,)
        """

    # ═══════════════════════════════════════════════════════════════
    #  CLF Lyapunov function
    #  (called by clf.build_clf_constraints at every ADMM iteration)
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def V_clf(
        self,
        state : np.ndarray,
        goal  : np.ndarray,
    ) -> float:
        """
        Control Lyapunov Function.

        Required mathematical properties:
            V(x) ≥ 0    for all x
            V(x) = 0    if and only if x is at the goal
                        (goal position AND zero velocity)

        Typical construction for ground vehicles:
            V = (x−gx)² + (y−gy)² + k_v·v² + k_ω·ω²

        where k_v, k_ω > 0 are tuning scalars that weight the
        "stop at the goal" requirement.

        Parameters
        ----------
        state : shape (n,)
        goal  : shape (2,)  — (x, y) target from scene_editor

        Returns
        -------
        float ≥ 0
        """

    @abstractmethod
    def V_clf_grad(
        self,
        state : np.ndarray,
        goal  : np.ndarray,
    ) -> np.ndarray:
        """
        Gradient of V_clf w.r.t. state.

        Returns
        -------
        grad : shape (n,)

        All components must be correct — position, velocity, angular
        rate terms all appear in a well-designed V.
        """

    # ═══════════════════════════════════════════════════════════════
    #  State / target construction helpers
    #  (called once per solver() invocation in admm.py)
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def initial_state(self, scene_start: np.ndarray) -> np.ndarray:
        """
        Convert the scene editor's start vector to this vehicle's state.

        The scene editor always stores start as a shape-(5,) array:
            [x, y, theta, v=0.0, omega=0.0]
        (All vehicles share the same scene editor; each vehicle takes
        what it needs and ignores the rest.)

        

        Parameters
        ----------
        scene_start : shape (5,) — [x, y, theta, v, omega]

        Returns
        -------
        x0 : shape (n,)
        """

    @abstractmethod
    def goal_target(
        self,
        goal_xy : np.ndarray,
        N       : int,
    ) -> np.ndarray:
        """
        Build the flat desired trajectory r_target used in the cost function.

        Constructs a full-state target for every timestep 0 … N, where the
        desired state is: at the goal position, zero velocity everywhere.

        Parameters
        ----------
        goal_xy : shape (2,) — (x, y) from scene editor
        N       : planning horizon

        Returns
        -------
        r_target : shape (n * (N+1),)

        
        """
