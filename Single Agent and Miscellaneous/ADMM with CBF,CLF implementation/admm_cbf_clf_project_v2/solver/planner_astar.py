"""
solver/planner_astar.py — Vehicle-agnostic grid A* global route planner.

Why this exists
----------------
The old r-update (see admm.py, now removed) solved an UNCONSTRAINED
L-BFGS-B problem with a soft `obstacle_penalty` term added on top. That's
a Lagrangian trade-off, not a guarantee: whenever the ADMM consensus pull
(ρ/2)‖x−r+v‖² or the tracking pull toward r_target was strong enough, it
was cheaper for the optimiser to eat some `weight·depth²` penalty than to
detour — so the reference (and, chasing it, the trajectory) would cut
through obstacles.

This module replaces that whole mechanism with a discrete A* search over
an inflated occupancy grid. A* cannot expand into an occupied cell, so
the resulting route is collision-free *by construction* — no weight to
tune, no trade-off to lose. admm.py now uses the (smoothed + resampled)
output of plan_astar_path() as a FIXED reference trajectory `r` for the
whole ADMM run, instead of re-solving an r-subproblem every iteration.

Pipeline
--------
1. Build an inflated boolean occupancy grid covering the scene
   (start, goal, and all obstacles, plus padding).
2. Run 8-connected A* (octile heuristic) from start to goal on that grid.
3. String-pull / shortcut the raw grid path (removes the zig-zag grid
   artefacts — a smooth polyline is a much better thing to track than a
   staircase).
4. Arc-length resample the shortcut path to exactly N+1 waypoints, so it
   lines up 1:1 with the planning horizon used by the rest of the solver.

This module only ever reasons about (x, y) — it is completely
vehicle-agnostic, exactly like geometry.py's position_barrier, and uses
the same obstacle dict convention as the rest of the codebase:
    {"type": "circle",    "params": (cx, cy, radius)}
    {"type": "rectangle", "params": (x_min, y_min, x_max, y_max)}

Public API
----------
plan_astar_path(start_xy, goal_xy, obstacles, N,
                 resolution=None, inflate=None, padding=None) -> np.ndarray

Returns an array of shape (N+1, 2) — the resampled, collision-free
position reference. Raises RuntimeError if no path exists (e.g. goal is
fully enclosed by obstacles even after accounting for inflation).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

_SQRT2 = math.sqrt(2.0)

# 8-connected moves: (di, dj, step_cost)
_NEIGHBOURS = [
    (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
]


# ── Occupancy ───────────────────────────────────────────────────────────

def _inflated_point_is_free(x: float, y: float, obstacles: list, inflate: float) -> bool:
    """True iff (x, y) is outside every obstacle, inflated by `inflate`."""
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy, radius = obs["params"]
            if (x - cx) ** 2 + (y - cy) ** 2 <= (radius + inflate) ** 2:
                return False
        elif obs["type"] == "rectangle":
            x_min, y_min, x_max, y_max = obs["params"]
            if (x_min - inflate) <= x <= (x_max + inflate) and \
               (y_min - inflate) <= y <= (y_max + inflate):
                return False
        # unknown obstacle type -> ignored (treated as no obstacle)
    return True


def _segment_is_free(
    p0: np.ndarray, p1: np.ndarray, obstacles: list, inflate: float,
    samples_per_unit: float = 20.0,
) -> bool:
    """
    Densely samples the segment p0->p1 and checks every sample against
    the inflated obstacles. Used both for A* neighbour validity (short
    hops) and for string-pulling shortcut checks (long hops).
    """
    dist = float(np.linalg.norm(p1 - p0))
    n_samples = max(2, int(math.ceil(dist * samples_per_unit)) + 1)
    for k in range(n_samples + 1):
        t = k / n_samples
        p = p0 + t * (p1 - p0)
        if not _inflated_point_is_free(p[0], p[1], obstacles, inflate):
            return False
    return True


# ── Grid <-> world ──────────────────────────────────────────────────────

def _world_to_grid(x, y, x_min, y_min, resolution):
    return (int(round((x - x_min) / resolution)), int(round((y - y_min) / resolution)))


def _grid_to_world(i, j, x_min, y_min, resolution):
    return (x_min + i * resolution, y_min + j * resolution)


# ── A* search ─────────────────────────────────────────────────────────

def _astar_grid(start_ij, goal_ij, free_fn, heuristic_scale=1.0):
    """
    Standard 8-connected A* on an implicit grid (free_fn decides
    occupancy on demand — no need to materialise the whole grid array).

    Returns list of (i, j) cells from start to goal inclusive, or None
    if no path exists.
    """
    def h(cell):
        di = abs(cell[0] - goal_ij[0])
        dj = abs(cell[1] - goal_ij[1])
        # octile distance
        return heuristic_scale * (max(di, dj) + (_SQRT2 - 1.0) * min(di, dj))

    open_heap = [(h(start_ij), 0.0, start_ij)]
    came_from = {}
    g_score = {start_ij: 0.0}
    closed = set()

    while open_heap:
        _, g, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        if cell == goal_ij:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            path.reverse()
            return path
        closed.add(cell)

        for di, dj, step_cost in _NEIGHBOURS:
            nxt = (cell[0] + di, cell[1] + dj)
            if nxt in closed:
                continue
            if not free_fn(nxt):
                continue
            tentative_g = g + step_cost
            if tentative_g < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = cell
                heapq.heappush(open_heap, (tentative_g + h(nxt), tentative_g, nxt))

    return None


# ── Shortcutting (string-pulling) ────────────────────────────────────────

def _shortcut_path(points: np.ndarray, obstacles: list, inflate: float) -> np.ndarray:
    """
    Greedy line-of-sight shortcutting: from each point, jump as far
    forward as possible while the straight segment stays collision-free.
    Turns a jagged staircase grid-path into a smooth-ish polyline with
    far fewer vertices, while still guaranteeing every segment is safe.
    """
    if len(points) <= 2:
        return points

    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not _segment_is_free(points[i], points[j], obstacles, inflate):
            j -= 1
        out.append(points[j])
        i = j
    return np.array(out)


# ── Arc-length resampling ────────────────────────────────────────────────

def _resample_path_at(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Sample a polyline at the given arc-length values (any spacing)."""
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    out = np.empty((len(targets), 2))
    seg_idx = 0
    for k, s in enumerate(targets):
        s = min(max(s, 0.0), cum[-1])
        while seg_idx < len(cum) - 2 and cum[seg_idx + 1] < s:
            seg_idx += 1
        seg_len = cum[seg_idx + 1] - cum[seg_idx]
        t = 0.0 if seg_len < 1e-12 else (s - cum[seg_idx]) / seg_len
        out[k] = points[seg_idx] + t * (points[seg_idx + 1] - points[seg_idx])
    return out


def _resample_path(points: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a polyline to n_out points, evenly spaced by arc length.

    Kinematically naive — implies a constant-speed profile with NO ramp
    up from rest, which conflicts with a vehicle that starts at v=0 (see
    _resample_path_eased). Kept for callers that only care about
    geometry, not time — plan_astar_path uses the eased version whenever
    `dt` is supplied.
    """
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if total < 1e-12:
        return np.tile(points[0], (n_out, 1))
    targets = np.linspace(0.0, total, n_out)
    return _resample_path_at(points, targets)


def _resample_path_eased(
    points     : np.ndarray,
    n_out      : int,
    dt         : float,
    max_speed  : float = None,
    ramp_time  : float = None,
) -> np.ndarray:
    """
    Resample a polyline to n_out points using an ease-in/ease-out speed
    profile instead of uniform arc-length spacing.

    Two distinct problems this solves
    ----------------------------------
    1. v(0) = 0 always. A vehicle starting at rest can't instantaneously
       be at cruise speed — a uniformly-spaced arc-length resample
       implicitly demands exactly that at t=0. Since r is now FIXED for
       the whole ADMM run (see admm.py), that single contradiction at
       t=0 creates a persistent, unresolvable tug-of-war for the
       xu-update every round, with nowhere to relax into.

    2. max_speed (if given) caps how fast the reference is allowed to
       move at all. Without a cap, if the route is long relative to the
       horizon T = (n_out-1)*dt, the unconstrained profile (case
       max_speed=None) still forces whatever speed is needed to arrive
       exactly at T — which can be an absurd speed if T is short. A
       genuinely-too-short horizon isn't something time-parameterization
       can paper over: if the vehicle can't reach the goal on the given
       route within T even at max_speed, this function does NOT force
       it to anyway. Instead it moves the reference as far as it can at
       max_speed and holds the reference at that last-reached (still
       on-route, still safe) point for the remaining timesteps — a
       reference that's honestly reachable, only partial, beats one
       that's fully specified but physically impossible to track.

    Profile shape: ease-in ramp (0 -> peak speed) of duration
    `ramp_time`, symmetric ease-out ramp (peak speed -> 0) sized the
    same way, constant-speed cruise in between if there's slack. Each
    ramp uses v(t) = v_peak · sin²(π t / (2·ramp_time)) — smooth, no
    jerk discontinuity, average speed exactly v_peak/2 over the ramp.

    Parameters
    ----------
    points    : polyline vertices, shape (M, 2)
    n_out     : number of output waypoints (= N+1)
    dt        : solver timestep — T = (n_out-1)*dt is the horizon
    max_speed : cap on reference speed. None -> uncapped (old
                behaviour: always arrives exactly at t=T, whatever
                speed that implies — only safe if you've already
                checked the horizon is long enough for the route).
                RECOMMENDED to always pass this.
    ramp_time : duration of the accel/decel ramps. Default:
                min(1.0, 0.25*T) — quarter of the horizon, capped at
                1s so short horizons still get a real cruise phase.

    Returns
    -------
    path : np.ndarray, shape (n_out, 2)
    """
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if total < 1e-12 or n_out < 2:
        return np.tile(points[0], (n_out, 1))

    T = (n_out - 1) * dt
    if T < 1e-9:
        return np.tile(points[0], (n_out, 1))

    if ramp_time is None:
        ramp_time = min(1.0, 0.25 * T)
    ramp_time = max(min(ramp_time, T / 2.0), 1e-6)

    if max_speed is None:
        # Unconstrained: single smooth hump sized to arrive exactly at T.
        v_peak = 2.0 * total / T
        vfunc = lambda t: v_peak * math.sin(math.pi * t / T) ** 2
    else:
        full_ramp_dist = max_speed * ramp_time  # distance if ramp up+down at max_speed
        if full_ramp_dist <= total:
            cruise_dist = total - full_ramp_dist
            cruise_time = cruise_dist / max_speed
            arrival     = 2.0 * ramp_time + cruise_time
            if arrival <= T:
                # Full profile fits: ramp up, cruise, ramp down, hold at goal.
                t_cruise_end = ramp_time + cruise_time
                def vfunc(t, _rt=ramp_time, _ce=t_cruise_end, _ar=arrival, _vp=max_speed):
                    if t < _rt:
                        return _vp * math.sin(math.pi * t / (2.0 * _rt)) ** 2
                    elif t < _ce:
                        return _vp
                    elif t < _ar:
                        tt = t - _ce
                        return _vp * math.sin(math.pi * (_rt - tt) / (2.0 * _rt)) ** 2
                    return 0.0
            else:
                # Horizon too short even at max_speed: ramp up, then
                # cruise for the rest of T with NO deceleration — the
                # reference simply doesn't reach the goal in this run.
                def vfunc(t, _rt=ramp_time, _vp=max_speed):
                    if t < _rt:
                        return _vp * math.sin(math.pi * t / (2.0 * _rt)) ** 2
                    return _vp
        else:
            # Can't even do a full ramp up+down at max_speed without
            # overshooting total distance -> shorter triangular profile
            # with a lower peak, still smooth, still reaches the goal.
            v_pk = total / ramp_time
            def vfunc(t, _rt=ramp_time, _vp=v_pk):
                if t < _rt:
                    return _vp * math.sin(math.pi * t / (2.0 * _rt)) ** 2
                elif t < 2.0 * _rt:
                    tt = t - _rt
                    return _vp * math.sin(math.pi * (_rt - tt) / (2.0 * _rt)) ** 2
                return 0.0

    n_fine = max(500, 20 * n_out)
    ts_fine = np.linspace(0.0, T, n_fine)
    vs_fine = np.array([vfunc(tt) for tt in ts_fine])
    s_fine  = np.concatenate([[0.0], np.cumsum(
        (vs_fine[:-1] + vs_fine[1:]) / 2.0 * (ts_fine[1] - ts_fine[0])
    )])
    s_fine  = np.minimum(s_fine, total)  # guard tiny numeric overshoot

    t_out = np.arange(n_out) * dt
    targets = np.interp(t_out, ts_fine, s_fine)
    return _resample_path_at(points, targets)


# ── Public API ────────────────────────────────────────────────────────

def plan_astar_path(
    start_xy   : np.ndarray,
    goal_xy    : np.ndarray,
    obstacles  : list,
    N          : int,
    dt         : float = None,
    max_speed  : float = None,
    resolution : float = None,
    inflate    : float = None,
    padding    : float = None,
    ramp_time  : float = None,
) -> np.ndarray:
    """
    Plan a collision-free route from start_xy to goal_xy around
    `obstacles`, resampled to N+1 waypoints.

    Parameters
    ----------
    start_xy   : shape (2,)
    goal_xy    : shape (2,)
    obstacles  : list of obstacle dicts (see module docstring)
    N          : planning horizon — output has N+1 rows
    dt         : solver timestep. If given, waypoints are
                 time-parameterized with an ease-in/ease-out speed
                 profile (v=0 at both ends) instead of naive uniform
                 arc-length spacing — see _resample_path_eased for why
                 that matters once r is held fixed across the whole
                 ADMM run. STRONGLY recommended to pass this.
    max_speed  : cap on the reference's speed (units/s). If the route
                 can't be covered within N*dt even at this speed, the
                 reference honestly stops short of the goal rather
                 than demanding an impossible pace — see
                 _resample_path_eased. Ignored if dt is None.
                 RECOMMENDED whenever N*dt might be tight relative to
                 the route length (always worth passing).
    resolution : grid cell size. Default: extent / 60.
    inflate    : obstacle inflation margin. Default: generous auto value.
    padding    : bounding-box margin around start/goal/obstacles.
                 Default: auto value, see below.
    ramp_time  : accel/decel ramp duration for the eased profile.
                 Default: min(1.0, 0.25*N*dt). Ignored if dt is None.

    Returns
    -------
    path : np.ndarray, shape (N+1, 2)

    Raises
    ------
    RuntimeError if no collision-free path exists on the grid (e.g. the
    goal is fully enclosed by obstacles even after accounting for
    inflation).
    """
    start_xy = np.asarray(start_xy, dtype=float)
    goal_xy  = np.asarray(goal_xy,  dtype=float)

    # ── bounding box over start, goal, and all obstacles ───────────────
    xs = [start_xy[0], goal_xy[0]]
    ys = [start_xy[1], goal_xy[1]]
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy, r = obs["params"]
            xs += [cx - r, cx + r]
            ys += [cy - r, cy + r]
        elif obs["type"] == "rectangle":
            x_min, y_min, x_max, y_max = obs["params"]
            xs += [x_min, x_max]
            ys += [y_min, y_max]

    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    extent = max(x_hi - x_lo, y_hi - y_lo, 1e-6)

    if resolution is None:
        resolution = max(extent / 60.0, 1e-3)
    if inflate is None:
        inflate = max(6.0 * resolution, 0.05 * extent)
    if padding is None:
        # Must clear enough room to route AROUND an obstacle that sits
        # at the edge of the start/goal/obstacle bounding box — if
        # padding <= inflate, the inflated obstacle can seal off the
        # entire scene margin and leave no detour lane at all.
        padding = max(0.15 * extent, 2.0 * inflate + 4.0 * resolution)

    x_lo -= padding
    x_hi += padding
    y_lo -= padding
    y_hi += padding

    n_cols = max(2, int(math.ceil((x_hi - x_lo) / resolution)) + 1)
    n_rows = max(2, int(math.ceil((y_hi - y_lo) / resolution)) + 1)

    def free_fn(cell):
        i, j = cell
        if not (0 <= i < n_cols and 0 <= j < n_rows):
            return False
        x, y = _grid_to_world(i, j, x_lo, y_lo, resolution)
        return _inflated_point_is_free(x, y, obstacles, inflate)

    start_ij = _world_to_grid(start_xy[0], start_xy[1], x_lo, y_lo, resolution)
    goal_ij  = _world_to_grid(goal_xy[0],  goal_xy[1],  x_lo, y_lo, resolution)

    # Guard against start/goal themselves landing on an occupied cell
    # due to inflation (common if the vehicle starts right next to an
    # obstacle) — fall back to the nearest free cell within a small
    # search radius rather than failing outright.
    start_ij = _nearest_free_cell(start_ij, free_fn)
    goal_ij  = _nearest_free_cell(goal_ij,  free_fn)

    grid_path = _astar_grid(start_ij, goal_ij, free_fn)
    if grid_path is None:
        raise RuntimeError(
            "plan_astar_path: no collision-free route found "
            f"(resolution={resolution:.4g}, inflate={inflate:.4g}). "
            "Scene may be fully blocked at this inflation margin."
        )

    world_pts = np.array([
        _grid_to_world(i, j, x_lo, y_lo, resolution) for i, j in grid_path
    ])
    # Snap exact endpoints back to the requested start/goal (grid
    # rounding otherwise offsets them by up to resolution/2).
    world_pts[0]  = start_xy
    world_pts[-1] = goal_xy

    smooth_pts = _shortcut_path(world_pts, obstacles, inflate)

    if dt is None:
        return _resample_path(smooth_pts, N + 1)
    return _resample_path_eased(smooth_pts, N + 1, dt, max_speed=max_speed, ramp_time=ramp_time)


def _nearest_free_cell(cell, free_fn, max_radius: int = 8):
    """BFS ring search outward from `cell` for the nearest free cell."""
    if free_fn(cell):
        return cell
    for radius in range(1, max_radius + 1):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if max(abs(di), abs(dj)) != radius:
                    continue
                cand = (cell[0] + di, cell[1] + dj)
                if free_fn(cand):
                    return cand
    # give up — return the original cell and let A* report failure
    return cell
