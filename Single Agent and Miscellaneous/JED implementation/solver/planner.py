"""
solver/planner.py — Global A* planner (layer 1 of the 3-layer architecture).

Runs once at the start (and can be re-invoked if the drone deviates
significantly or the obstacle set changes). Plans over a grid using
INFLATED obstacle geometry: r_robot + buffer + the same velocity-dependent
stopping-distance term used in barriers.py's JED hyperplanes (with a
representative speed, since A* has no velocity state) — this keeps A* from
proposing a path JED will have to fight near corners.
"""

from __future__ import annotations

import heapq

import numpy as np

from . import config


def _inflated_radius(base_r: float, buffer: float, rep_speed: float, a_max: float) -> float:
    """Same braking-buffer formula as barriers.get_separating_hyperplane, using
    a representative (worst-case approach) speed since A* has no velocity state."""
    braking = (rep_speed ** 2) / (2.0 * a_max)
    return base_r + buffer + braking


def _point_blocked(p: np.ndarray, obstacles: list[dict], inflate: float) -> bool:
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy, R = obs["params"]
            if np.linalg.norm(p - np.array([cx, cy])) <= R + inflate:
                return True
        elif obs["type"] == "rectangle":
            x_min, y_min, x_max, y_max = obs["params"]
            if (x_min - inflate <= p[0] <= x_max + inflate and
                    y_min - inflate <= p[1] <= y_max + inflate):
                return True
        else:
            raise ValueError(f"Unknown obstacle type: {obs['type']!r}")
    return False


def _segment_blocked(p1: np.ndarray, p2: np.ndarray, obstacles: list[dict],
                      inflate: float, n_samples: int = 20) -> bool:
    for t in np.linspace(0.0, 1.0, n_samples):
        p = p1 + t * (p2 - p1)
        if _point_blocked(p, obstacles, inflate):
            return True
    return False


_NEIGHBORS_8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def astar_plan(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[dict],
    grid_res: float = config.GRID_RES,
    r_robot: float = config.R_ROBOT,
    buffer: float = config.PLANNER_BUFFER,
    world_bounds: tuple = config.WORLD_BOUNDS,
    rep_speed: float = config.REPRESENTATIVE_SPEED,
    a_max: float = config.A_MAX,
) -> list[np.ndarray]:
    """
    Grid-based A* over the 2-D workspace. Returns an ordered list of
    waypoints [g_1, ..., g_N] (world coordinates, g_N = goal). Returns an
    empty list if no path is found.
    """
    (xmin, xmax), (ymin, ymax) = world_bounds
    inflate = _inflated_radius(r_robot, buffer, rep_speed, a_max)

    def to_cell(p):
        return (int(round((p[0] - xmin) / grid_res)), int(round((p[1] - ymin) / grid_res)))

    def to_world(c):
        return np.array([xmin + c[0] * grid_res, ymin + c[1] * grid_res])

    nx = int(round((xmax - xmin) / grid_res)) + 1
    ny = int(round((ymax - ymin) / grid_res)) + 1

    start_c = to_cell(start[:2])
    goal_c = to_cell(goal[:2])

    def in_bounds(c):
        return 0 <= c[0] < nx and 0 <= c[1] < ny

    def blocked(c):
        return _point_blocked(to_world(c), obstacles, inflate)

    if not in_bounds(start_c) or not in_bounds(goal_c):
        return []
    if blocked(goal_c):
        return []

    def h_heuristic(c):
        return float(np.hypot(c[0] - goal_c[0], c[1] - goal_c[1])) * grid_res

    open_heap = [(h_heuristic(start_c), 0.0, start_c)]
    came_from = {}
    g_score = {start_c: 0.0}
    visited = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_c:
            break

        for dx, dy in _NEIGHBORS_8:
            nbr = (current[0] + dx, current[1] + dy)
            if not in_bounds(nbr) or blocked(nbr):
                continue
            step_cost = grid_res * (np.hypot(dx, dy))
            tentative_g = g + step_cost
            if tentative_g < g_score.get(nbr, np.inf):
                g_score[nbr] = tentative_g
                came_from[nbr] = current
                f = tentative_g + h_heuristic(nbr)
                heapq.heappush(open_heap, (f, tentative_g, nbr))

    if goal_c not in came_from and goal_c != start_c:
        return []

    # reconstruct
    path_cells = [goal_c]
    node = goal_c
    while node != start_c:
        node = came_from[node]
        path_cells.append(node)
    path_cells.reverse()

    waypoints = [to_world(c) for c in path_cells]
    waypoints[0] = np.array(start[:2], dtype=float)
    waypoints[-1] = np.array(goal[:2], dtype=float)
    return waypoints


def shortcut_waypoints(
    waypoints: list[np.ndarray],
    obstacles: list[dict],
    r_robot: float = config.R_ROBOT,
    buffer: float = config.PLANNER_BUFFER,
    rep_speed: float = config.REPRESENTATIVE_SPEED,
    a_max: float = config.A_MAX,
) -> list[np.ndarray]:
    """
    Simple line-of-sight shortcutting: greedily connect the farthest-ahead
    waypoint that has a clear straight segment, to avoid jagged grid-aligned
    paths.
    """
    if len(waypoints) <= 2:
        return list(waypoints)

    inflate = _inflated_radius(r_robot, buffer, rep_speed, a_max)
    result = [waypoints[0]]
    i = 0
    n = len(waypoints)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if not _segment_blocked(waypoints[i], waypoints[j], obstacles, inflate):
                break
            j -= 1
        result.append(waypoints[j])
        i = j
    return result
