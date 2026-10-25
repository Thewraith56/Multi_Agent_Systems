"""
solver/dynamics.py — 2-D double-integrator drone dynamics.

State  x = [x, y, vx, vy]      n = 4
Control u = [ax, ay]            m = 2

Discrete-time (Euler / Heun-equivalent for a linear system) update:

    x_{k+1} = A x_k + B u_k + d_k

This module owns the ONE definition of A, B used everywhere else (JED QP,
LQR design, closed-loop stepping, Monte-Carlo rollouts) so nothing can
silently disagree about the model.
"""

from __future__ import annotations

import numpy as np

from . import config


def build_AB(dt: float = config.DT) -> tuple[np.ndarray, np.ndarray]:
    """Discrete double-integrator matrices for timestep dt."""
    A = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.5 * dt ** 2, 0.0          ],
        [0.0,           0.5 * dt ** 2],
        [dt,            0.0          ],
        [0.0,            dt          ],
    ])
    return A, B


# NOTE: DroneVehicle itself lives in solver/vehicles/drone.py (BaseVehicle
# subclass) — this module only owns the A, B construction so both the
# vehicle class and anything else needing raw matrices (LQR design, the JED
# QP) share one definition.
