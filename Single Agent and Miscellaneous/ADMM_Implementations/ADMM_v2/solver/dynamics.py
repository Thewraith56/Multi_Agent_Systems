

import numpy as np


# ── single step ────────────────────────────────────────────────────────────

def unicycle_step(state: np.ndarray, ctrl: np.ndarray, dt: float) -> np.ndarray:
    """
    State : [x, y, theta, v, omega]
    Ctrl  : [a, alpha_acc]
    """
    x1, x2, theta, v, omega = state
    a, alpha_acc = ctrl
    
    return np.array([
        x1    + v * np.cos(theta) * dt,  # Update X using current velocity
        x2    + v * np.sin(theta) * dt,  # Update Y using current velocity
        theta + omega             * dt,  # Update heading using current turning rate
        v     + a                 * dt,  # Update velocity using acceleration
        omega + alpha_acc         * dt,  # Update turning rate using angular accel
    ])




# ── scipy equality constraint ──────────────────────────────────────────────

def unicycle_step_jac(state: np.ndarray, ctrl: np.ndarray, dt: float):
    """Analytic Jacobian of unicycle_step wrt state (5x5) and ctrl (5x2)."""
    x1, x2, theta, v, omega = state
    Jx = np.array([
        [1.0, 0.0, -v * np.sin(theta) * dt, np.cos(theta) * dt, 0.0],
        [0.0, 1.0,  v * np.cos(theta) * dt, np.sin(theta) * dt, 0.0],
        [0.0, 0.0, 1.0,                     0.0,                dt ],
        [0.0, 0.0, 0.0,                     1.0,                0.0],
        [0.0, 0.0, 0.0,                     0.0,                1.0],
    ])
    Ju = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [dt,  0.0],
        [0.0, dt ],
    ])
    return Jx, Ju


def build_dynamics_constraint(
    x0_state: np.ndarray,
    n: int,
    m: int,
    N: int,
    dt: float,
) -> dict:
    size_x = n * (N + 1)
    total  = size_x + m * N

    def dynamics(z: np.ndarray) -> np.ndarray:
        x_flat = z[:size_x]
        u_flat = z[size_x:]

        residuals = np.zeros(size_x)
        residuals[:n] = x_flat[:n] - x0_state

        for t in range(N):
            xt     = x_flat[ t      * n : (t + 1) * n]
            ut     = u_flat[ t      * m : (t + 1) * m]
            x_next = unicycle_step(xt, ut, dt)
            residuals[(t + 1) * n : (t + 2) * n] = (
                x_flat[(t + 1) * n : (t + 2) * n] - x_next
            )

        return residuals

    def dynamics_jac(z: np.ndarray) -> np.ndarray:
        x_flat = z[:size_x]
        u_flat = z[size_x:]

        J = np.zeros((size_x, total))
        J[:n, :n] = np.eye(n)

        for t in range(N):
            xt = x_flat[t * n:(t + 1) * n]
            ut = u_flat[t * m:(t + 1) * m]
            Jx, Ju = unicycle_step_jac(xt, ut, dt)
            row = slice((t + 1) * n, (t + 2) * n)
            J[row, row] = np.eye(n)                                # d/d x_{t+1}
            J[row, t * n:(t + 1) * n] = -Jx                          # d/d x_t
            J[row, size_x + t * m: size_x + (t + 1) * m] = -Ju       # d/d u_t

        return J

    return {"type": "eq", "fun": dynamics, "jac": dynamics_jac}