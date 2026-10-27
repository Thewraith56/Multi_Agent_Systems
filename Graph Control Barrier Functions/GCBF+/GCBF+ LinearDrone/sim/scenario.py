"""Scenario description (agents, targets, spherical obstacles with constant velocities) + random generation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Scenario:
    dim: int = 3                       # 2 or 3
    area: float = 2.0                  # side length of the square / cube [0, area]^dim
    starts: np.ndarray = None          # (N, dim)
    goals: np.ndarray = None           # (N, dim)
    goal_vel: np.ndarray = None        # (N, dim) constant target velocity
    obs_pos: np.ndarray = None         # (M, dim)
    obs_radius: np.ndarray = None      # (M,)
    obs_vel: np.ndarray = None         # (M, dim) constant obstacle velocity
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        d = self.dim
        z = lambda *s: np.zeros(s, dtype=np.float32)
        self.starts = z(0, d) if self.starts is None else np.asarray(self.starts, np.float32).reshape(-1, d)
        n = len(self.starts)
        self.goals = z(n, d) if self.goals is None else np.asarray(self.goals, np.float32).reshape(-1, d)
        self.goal_vel = z(n, d) if self.goal_vel is None else np.asarray(self.goal_vel, np.float32).reshape(-1, d)
        self.obs_pos = z(0, d) if self.obs_pos is None else np.asarray(self.obs_pos, np.float32).reshape(-1, d)
        m = len(self.obs_pos)
        self.obs_radius = z(m) if self.obs_radius is None else np.asarray(self.obs_radius, np.float32).reshape(-1)
        self.obs_vel = z(m, d) if self.obs_vel is None else np.asarray(self.obs_vel, np.float32).reshape(-1, d)

    @property
    def n_agents(self) -> int:
        return len(self.starts)

    @property
    def n_obs(self) -> int:
        return len(self.obs_pos)

    def copy(self) -> "Scenario":
        return Scenario(self.dim, self.area, self.starts.copy(), self.goals.copy(), self.goal_vel.copy(),
                        self.obs_pos.copy(), self.obs_radius.copy(), self.obs_vel.copy(), list(self.warnings))

    def to_json(self, path: str):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k != "warnings"}
        with open(path, "w") as f:
            json.dump(d, f)

    @staticmethod
    def from_json(path: str) -> "Scenario":
        with open(path) as f:
            return Scenario(**json.load(f))


class _Grid:
    """Spatial hash for fast 'is any existing point closer than r' queries."""

    def __init__(self, cell: float, dim: int):
        self.cell, self.dim, self.cells = cell, dim, {}
        self._offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).reshape(dim, -1).T

    def _key(self, p):
        return tuple((p // self.cell).astype(int))

    def too_close(self, p, r):
        k = np.array(self._key(p))
        for off in self._offsets:
            for q in self.cells.get(tuple(k + off), ()):
                if np.linalg.norm(q - p) <= r:
                    return True
        return False

    def add(self, p):
        self.cells.setdefault(self._key(p), []).append(p)


def _rand_dirs(rng, n, dim):
    v = rng.normal(size=(n, dim))
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def random_velocities(rng, n, dim, speed_range) -> np.ndarray:
    lo, hi = speed_range
    return (_rand_dirs(rng, n, dim) * rng.uniform(lo, hi, size=(n, 1))).astype(np.float32)


def random_scenario(dim: int, n_agents: int, n_obs: int, area: float, drone_radius: float = 0.05,
                    obs_radius_range: Tuple[float, float] = (0.075, 0.15), max_travel: float = None,
                    obs_speed: Tuple[float, float] = (0., 0.), goal_speed: Tuple[float, float] = (0., 0.),
                    seed: int = 0, max_tries: int = 2000) -> Scenario:
    """Uniform random obstacles, then start/goal points that keep 4*drone_radius clearance from each other
    (starts vs starts, goals vs goals) and from obstacles - the same rule as the original `reset`."""
    rng = np.random.default_rng(seed)
    warn = []
    obs_pos = rng.uniform(0, area, size=(n_obs, dim)).astype(np.float32)
    obs_r = rng.uniform(obs_radius_range[0], obs_radius_range[1], size=n_obs).astype(np.float32)
    min_d = 4 * drone_radius

    def free_of_obstacles(p):
        return n_obs == 0 or not np.any(np.linalg.norm(obs_pos - p, axis=1) <= obs_r + min_d)

    starts, goals = np.zeros((n_agents, dim), np.float32), np.zeros((n_agents, dim), np.float32)
    gs, gg = _Grid(min_d, dim), _Grid(min_d, dim)
    failed = 0
    for i in range(n_agents):
        for placing, grid, arr in (("start", gs, starts), ("goal", gg, goals)):
            for t in range(max_tries):
                if placing == "goal" and max_travel is not None:
                    p = starts[i] + rng.uniform(-max_travel, max_travel, size=dim)
                    if np.any(p < 0) or np.any(p > area) or np.linalg.norm(p - starts[i]) > max_travel:
                        continue
                else:
                    p = rng.uniform(0, area, size=dim)
                if free_of_obstacles(p) and not grid.too_close(p, min_d):
                    break
            else:
                failed += 1
            arr[i] = p
            grid.add(arr[i])
    if failed:
        warn.append(f"{failed} start/goal points could not keep full clearance (area too small for this many "
                    f"agents/obstacles) - enlarge the area or reduce counts.")
    return Scenario(dim, area, starts, goals, random_velocities(rng, n_agents, dim, goal_speed),
                    obs_pos, obs_r, random_velocities(rng, n_obs, dim, obs_speed), warn)


def default_area(n_agents: int, dim: int, base: float = 2.0) -> float:
    """Keep agent density roughly constant relative to the paper's 8 agents in a 2-unit area."""
    return float(max(base, base * (n_agents / 8.0) ** (1.0 / dim)))


def default_steps(area: float, dim: int, dt: float = 0.03) -> int:
    """Enough time to cross the arena at ~0.35 of max speed, clamped to [256, 2048]."""
    return int(np.clip(area * np.sqrt(dim) / (0.35 * dt), 256, 2048))


def convert_scenario(sc: Scenario, dim: int) -> Scenario:
    """2D <-> 3D: keep xy, put everything at mid-height when going up, drop z when going down."""
    if sc.dim == dim:
        return sc.copy()
    f = lambda a, fill: (np.concatenate([a, np.full((len(a), 1), fill, np.float32)], 1) if dim == 3 else a[:, :2])
    return Scenario(dim, sc.area, f(sc.starts, sc.area / 2), f(sc.goals, sc.area / 2), f(sc.goal_vel, 0.),
                    f(sc.obs_pos, sc.area / 2), sc.obs_radius.copy(), f(sc.obs_vel, 0.))


def resize_scenario(sc: Scenario, n_agents: int, n_obs: int, drone_radius: float, radius_range, obs_speed,
                    goal_speed, seed: int = 0, tries: int = 500) -> Scenario:
    """Truncate or extend while keeping existing (possibly hand-edited) items; new items are placed randomly with the
    usual clearance against everything already there."""
    rng, s, d, A = np.random.default_rng(seed), sc.copy(), sc.dim, sc.area
    min_d = 4 * drone_radius
    s.starts, s.goals, s.goal_vel = s.starts[:n_agents], s.goals[:n_agents], s.goal_vel[:n_agents]
    s.obs_pos, s.obs_radius, s.obs_vel = s.obs_pos[:n_obs], s.obs_radius[:n_obs], s.obs_vel[:n_obs]
    for _ in range(n_obs - s.n_obs):
        s.obs_pos = np.vstack([s.obs_pos, rng.uniform(0, A, d)]).astype(np.float32)
        s.obs_radius = np.append(s.obs_radius, rng.uniform(*radius_range)).astype(np.float32)
        s.obs_vel = np.vstack([s.obs_vel, random_velocities(rng, 1, d, obs_speed)])
    bad = 0
    for _ in range(n_agents - s.n_agents):
        pts = []
        for existing in (s.starts, s.goals):
            for _t in range(tries):
                p = rng.uniform(0, A, d)
                ok_pts = len(existing) == 0 or np.all(np.linalg.norm(existing - p, axis=1) > min_d)
                ok_obs = s.n_obs == 0 or np.all(np.linalg.norm(s.obs_pos - p, axis=1) > s.obs_radius + min_d)
                if ok_pts and ok_obs:
                    break
            else:
                bad += 1
            pts.append(p)
        s.starts = np.vstack([s.starts, pts[0]]).astype(np.float32)
        s.goals = np.vstack([s.goals, pts[1]]).astype(np.float32)
        s.goal_vel = np.vstack([s.goal_vel, random_velocities(rng, 1, d, goal_speed)])
    s.warnings = ([f"{bad} new points could not keep full clearance"] if bad else [])
    return s
