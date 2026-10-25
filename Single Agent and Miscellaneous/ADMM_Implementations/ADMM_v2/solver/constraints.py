
import numpy as np


def obstacle_penalty(
    r_flat    : np.ndarray,
    obstacles : list,
    n         : int,
    N         : int,
    weight    : float = 10000.0,
) -> float:
    """
    Soft penalty that grows when any reference position enters an obstacle.

    Parameters
    ----------
    r_flat    : flat reference trajectory, shape (n*(N+1),)
    obstacles : list of dicts from scene_editor, each with:
                  {"type": "rectangle", "params": (x_min, y_min, x_max, y_max)}
                  {"type": "circle",    "params": (cx, cy, radius)}
    n         : state dimension  (3 for unicycle)
    N         : planning horizon
    weight    : penalty coefficient — larger = harder obstacle boundary

    Returns
    -------
    float  — 0 when no time step is inside any obstacle, > 0 otherwise
    """
    penalty = 0.0

    for t in range(N + 1):
        rx = r_flat[t * n]
        ry = r_flat[t * n + 1]

        for obs in obstacles:

            if obs["type"] == "rectangle":
                x_min, y_min, x_max, y_max = obs["params"]
                inside_x = x_min < rx < x_max
                inside_y = y_min < ry < y_max
                if inside_x and inside_y:
                    # Depth of penetration = distance to nearest wall
                    depth = min(
                        rx - x_min, x_max - rx,
                        ry - y_min, y_max - ry,
                    )
                    penalty += weight * depth ** 2

            elif obs["type"] == "circle":
                cx, cy, radius = obs["params"]
                dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2) + 1e-9
                penetration = max(0.0, radius - dist)
                penalty += weight * penetration ** 2

    return penalty


def obstacle_penalty_grad(
    r_flat    : np.ndarray,
    obstacles : list,
    n         : int,
    N         : int,
    weight    : float = 10000.0,
) -> np.ndarray:
    """Analytic gradient of obstacle_penalty wrt the flat r vector."""
    grad = np.zeros_like(r_flat)

    for t in range(N + 1):
        ix, iy = t * n, t * n + 1
        rx, ry = r_flat[ix], r_flat[iy]

        for obs in obstacles:
            if obs["type"] == "rectangle":
                x_min, y_min, x_max, y_max = obs["params"]
                if x_min < rx < x_max and y_min < ry < y_max:
                    cands = [rx - x_min, x_max - rx, ry - y_min, y_max - ry]
                    idx = int(np.argmin(cands))
                    depth = cands[idx]
                    if idx == 0:
                        grad[ix] += weight * 2 * depth * 1.0
                    elif idx == 1:
                        grad[ix] += weight * 2 * depth * -1.0
                    elif idx == 2:
                        grad[iy] += weight * 2 * depth * 1.0
                    else:
                        grad[iy] += weight * 2 * depth * -1.0

            elif obs["type"] == "circle":
                cx, cy, radius = obs["params"]
                dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2) + 1e-9
                penetration = radius - dist
                if penetration > 0:
                    grad[ix] += weight * 2 * penetration * (-(rx - cx) / dist)
                    grad[iy] += weight * 2 * penetration * (-(ry - cy) / dist)

    return grad