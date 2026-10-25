

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

def build_dynamics_constraint(
    x0_state: np.ndarray,
    n: int,
    m: int,
    N: int,
    dt: float,
) -> dict:
    """
    Build a scipy equality-constraint dict encoding:

        x[0]     = x0_state               (initial condition)
        x[t+1]   = f(x[t], u[t], dt)      for t = 0 … N-1

    The optimisation variable z passed to the constraint is flat:
        z = [ x_flat  (n*(N+1),),   u_flat  (m*N,) ]

    Returns
    -------
    dict  compatible with scipy.optimize.minimize 'constraints' argument.
    """
    size_x = n * (N + 1)

    def dynamics(z: np.ndarray) -> np.ndarray:
        x_flat = z[:size_x]
        u_flat = z[size_x:]

        residuals = np.zeros(size_x)

        # Initial condition
        residuals[:n] = x_flat[:n] - x0_state

        # Dynamics at each step
        for t in range(N):
            xt     = x_flat[ t      * n : (t + 1) * n]
            ut     = u_flat[ t      * m : (t + 1) * m]
            x_next = unicycle_step(xt, ut, dt)
            residuals[(t + 1) * n : (t + 2) * n] = (
                x_flat[(t + 1) * n : (t + 2) * n] - x_next
            )

        return residuals

    return {"type": "eq", "fun": dynamics}
