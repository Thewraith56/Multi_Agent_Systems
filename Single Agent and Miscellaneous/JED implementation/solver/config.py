"""
solver/config.py — Single source of truth for every tunable constant in the
JED / DTCBF drone-navigation stack.

Nothing in barriers.py / jed_solver.py / planner.py / tracker.py / pu_bound.py
should hard-code a numeric constant that belongs here — import it instead, so
the tuning dashboard and the live sim always agree on what's "current".
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  Timing / world
# ═══════════════════════════════════════════════════════════════════
DT = 0.1                              # control timestep (s)
WORLD_BOUNDS = ((0.0, 10.0), (0.0, 10.0))   # (xmin,xmax), (ymin,ymax)

# ═══════════════════════════════════════════════════════════════════
#  Vehicle / actuation
# ═══════════════════════════════════════════════════════════════════
R_ROBOT = 0.15                        # robot radius, used to inflate every obstacle (m)
A_MAX   = 2.5                         # max commanded acceleration per axis (m/s^2)
U_MAX   = A_MAX                       # hard box bound enforced in the JED QP

# ═══════════════════════════════════════════════════════════════════
#  DTCBF / JED
# ═══════════════════════════════════════════════════════════════════
ALPHA          = 0.97                 # DTCBF decay rate, alpha in (0,1); paper suggests close to 1
SLACK_PENALTY  = 5.0e3                # quadratic penalty on the feasibility slack s >= 0
T_MIN, T_MAX   = 1.0e-3, 50.0         # box bounds for the LSE perspective variable t (t>0)

# Sensing horizon for which obstacles are "active" in the polytope this step.
# R_sense = ||v||*dt + 3*sigma_d  (sigma_d: representative disturbance std)
SENSE_SIGMA_MULT = 3.0

# ═══════════════════════════════════════════════════════════════════
#  Noise: d_k ~ N(mu, Sigma), additive process noise (nonzero mean, anisotropic)
#  State order is [x, y, vx, vy]; noise here is modeled as entering the full
#  state update (see solver/noise.py + solver/dynamics.py).
# ═══════════════════════════════════════════════════════════════════
NOISE_MU    = np.array([0.0, 0.0, 0.01, -0.01])            # small biased drift
NOISE_SIGMA = np.diag([0.0009, 0.0009, 0.0025, 0.0025])    # anisotropic covariance (m^2, (m/s)^2)

# Representative disturbance std used only for R_sense / A* inflation (scalar
# summary of an anisotropic covariance — take the largest positional std).
SIGMA_D_REPRESENTATIVE = float(np.sqrt(max(NOISE_SIGMA[0, 0], NOISE_SIGMA[1, 1])))

# ═══════════════════════════════════════════════════════════════════
#  A* global planner
# ═══════════════════════════════════════════════════════════════════
GRID_RES        = 0.2                 # A* grid cell size (m)
PLANNER_BUFFER  = 0.10                # extra safety buffer added on top of r_robot (m)
REPRESENTATIVE_SPEED = 1.0            # v used for A*'s velocity-dependent inflation (m/s), since A* has no velocity state

# ═══════════════════════════════════════════════════════════════════
#  LQR waypoint tracker
# ═══════════════════════════════════════════════════════════════════
LQR_Q = np.diag([10.0, 10.0, 1.0, 1.0])   # state cost: position >> velocity
LQR_R = np.diag([0.5, 0.5])               # control effort cost
SWITCH_RADIUS   = 0.35                    # advance to next waypoint within this distance (m)
GOAL_TOLERANCE  = 0.15                    # "reached goal" distance (m)
GOAL_SPEED_TOL  = 0.10                    # "stopped" speed threshold at goal (m/s)

# ═══════════════════════════════════════════════════════════════════
#  Closed-loop / Monte-Carlo validation
# ═══════════════════════════════════════════════════════════════════
N_STEPS_DEFAULT   = 150               # max closed-loop steps per rollout
K_HORIZON_DEFAULT = 100               # K used for Theorem 5's Pu bound
GAMMA_SWEEP       = [0.0, 1.0, 2.0, 3.0]
N_TRIALS_DEFAULT  = 500

# ═══════════════════════════════════════════════════════════════════
#  Theorem 5 M (upper bound on h)
# ═══════════════════════════════════════════════════════════════════
def compute_M(R_sense: float, arena_bounds: tuple = WORLD_BOUNDS) -> float:
    """
    h(x) = -max_i((C x)_i - w_i) is unbounded above away from obstacles
    (empty active set -> the polytope constraint is vacuous). We saturate it:

        M = min(R_sense, arena_diagonal)

    i.e. h can never be "trusted" to certify more clearance than either the
    sensing radius (nothing farther is even represented in the polytope) or
    the arena's own diagonal extent (h cannot mean more than "impossibly far
    from every wall").
    """
    (xmin, xmax), (ymin, ymax) = arena_bounds
    arena_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    return float(min(R_sense, arena_diag))
