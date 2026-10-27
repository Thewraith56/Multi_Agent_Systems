"""Simulation engine for the pretrained GCBF+ LinearDrone policy and moving scenarios."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from gcbfplus.algo.inference import GCBFPlusPolicy
from gcbfplus.env import make_env
from .scenario import Scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# UI-facing environment keys -> (core env id, scenario dim, planar embedding?)
ENV_KEYS = {
    "LinearDrone2D": ("LinearDrone", 2, True),  # planar view of the released 3D model; z fixed
    "LinearDrone": ("LinearDrone", 3, False),
}
DT = 0.03


# ----------------------------------------------------------------------------------------------- checkpoints
@dataclass
class Checkpoint:
    env: str
    algo: str
    path: str      # directory containing config.yaml and models/
    step: int

    @property
    def label(self) -> str:
        return f"{self.env}/{self.algo}  step {self.step}  [{os.path.relpath(self.path, ROOT)}]"


def find_checkpoints(root: str = ROOT) -> List[Checkpoint]:
    out = []
    for cfg in glob.glob(f"{root}/pretrained/LinearDrone/gcbf+/config.yaml"):
        d = os.path.dirname(cfg)
        mdir = os.path.join(d, "models")
        steps = [int(x) for x in os.listdir(mdir) if x.isdigit()] if os.path.isdir(mdir) else []
        if not steps:
            continue
        try:
            c = yaml.load(open(cfg), Loader=yaml.UnsafeLoader)
        except Exception:
            continue
        out.append(Checkpoint(getattr(c, "env", ""), getattr(c, "algo", ""), d, max(steps)))
    return sorted(out, key=lambda c: (c.env, c.algo, c.path))


def checkpoints_for(env_core: str, algo: str) -> List[Checkpoint]:
    return [c for c in find_checkpoints() if c.env == env_core and c.algo == algo]


# ----------------------------------------------------------------------------------------------- bundle
def _auto_caps(n_agents: int, n_obs: int, params: dict) -> dict:
    """Large scenarios: bound the graph size (see README, 'Scalability'). Small ones stay exact."""
    p = dict(params)
    if n_agents > 96:
        p.setdefault("max_neighbors", 24)
        p.setdefault("lidar_chunk", 32)
    if n_obs > 24:
        p.setdefault("max_obs_per_agent", 24)
    return p


class Bundle:
    """A compiled (env, controller, step) triple for one (env_key, N, M, params, algo, checkpoint)."""

    def __init__(self, env_key: str, n_agents: int, n_obs: int, area: float, params: Optional[dict],
                 ckpt: Optional[Checkpoint], max_step: int = 256):
        self.env_key, self.n_agents, self.n_obs, self.area = env_key, n_agents, n_obs, area
        self.core, self.dim, self.planar = ENV_KEYS[env_key]
        self.z0 = area / 2.0
        self.dt = float((params or {}).get("dt", DT))
        self.env = make_env(self.core, n_agents, area_size=area, max_step=max_step, num_obs=n_obs,
                            params=_auto_caps(n_agents, n_obs, params or {}))
        self.params = self.env.params
        self.ckpt = ckpt
        if ckpt is None:
            raise FileNotFoundError("Missing pretrained/LinearDrone/gcbf+ checkpoint.")
        with open(os.path.join(ckpt.path, "config.yaml"), encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        get_cfg = lambda k, d: getattr(cfg, k, d)
        controller = GCBFPlusPolicy(
            env=self.env, node_dim=self.env.node_dim, edge_dim=self.env.edge_dim,
            state_dim=self.env.state_dim, action_dim=self.env.action_dim, n_agents=n_agents,
            gnn_layers=get_cfg("gnn_layers", 1), seed=get_cfg("seed", 0))
        controller.load(os.path.join(ckpt.path, "models"), ckpt.step)
        self.act = controller.act
        self.advance = jax.jit(self._advance)

    # --- scenario -> env state -------------------------------------------------------------------------
    def _lift(self, p: np.ndarray) -> np.ndarray:
        """2D scenario coordinates -> coordinates of the underlying env (adds z for the planar embedding)."""
        if self.planar:
            return np.concatenate([p, np.full((len(p), 1), self.z0, np.float32)], axis=1)
        return p

    def initial(self, sc: Scenario):
        assert sc.n_agents == self.n_agents and sc.n_obs == self.n_obs
        sd = self.env.state_dim
        pad = lambda p: jnp.concatenate([jnp.asarray(self._lift(p)), jnp.zeros((len(p), sd - self._lift(p).shape[1]))], 1)
        center = jnp.asarray(self._lift(sc.obs_pos)).reshape(sc.n_obs, -1)
        obstacles = self.env.create_obstacles(center, jnp.asarray(sc.obs_radius))
        es = self.env.EnvState(pad(sc.starts), pad(sc.goals), obstacles)
        lift_v = lambda v: (np.concatenate([v, np.zeros((len(v), 1), np.float32)], 1) if self.planar else v)
        return self.env.get_graph(es), jnp.asarray(lift_v(sc.obs_vel)), jnp.asarray(lift_v(sc.goal_vel))

    # --- one simulation step (pure, jittable) ----------------------------------------------------------
    def _advance(self, graph, ovel, gvel):
        env, D = self.env, ovel.shape[1]
        es = graph.env_states
        action = self.act(graph)
        if self.planar:
            action = action.at[:, 2].set(0.)                       # no vertical command
        agent = env.agent_step(es.agent, action)
        if self.planar:
            agent = agent.at[:, 2].set(self.z0).at[:, 5].set(0.)   # stay exactly in the plane

        def move(pos, vel):
            nxt = pos + vel * self.dt
            out = (nxt < 0) | (nxt > self.area)
            return jnp.clip(nxt, 0., self.area), jnp.where(out, -vel, vel)

        if self.planar:                                            # z is not moved
            c, ov = move(es.obstacle.center, ovel)
            c, ov = c.at[:, 2].set(self.z0), ov.at[:, 2].set(0.)
            gp, gv = move(es.goal[:, :D], gvel)
            gp, gv = gp.at[:, 2].set(self.z0), gv.at[:, 2].set(0.)
        else:
            c, ov = move(es.obstacle.center, ovel)
            gp, gv = move(es.goal[:, :D], gvel)
        goal = es.goal.at[:, :D].set(gp)
        g2 = env.get_graph(env.EnvState(agent, goal, es.obstacle._replace(center=c)))
        return g2, ov, gv, action



_CACHE: Dict[tuple, Bundle] = {}


def get_bundle(env_key, n_agents, n_obs, area, params=None, ckpt: Optional[Checkpoint] = None,
               max_step=256) -> Bundle:
    key = (env_key, n_agents, n_obs, round(float(area), 6), tuple(sorted((params or {}).items())),
           ckpt.path if ckpt else None)
    if key not in _CACHE:
        if len(_CACHE) >= 4:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = Bundle(env_key, n_agents, n_obs, area, params, ckpt, max_step)
    return _CACHE[key]


# ----------------------------------------------------------------------------------------------- live engine
class SimEngine:
    """Stateful wrapper used by the UI: holds the current graph and exposes numpy snapshots."""

    def __init__(self):
        self.bundle: Optional[Bundle] = None
        self.scenario: Optional[Scenario] = None
        self.t = 0

    def configure(self, env_key: str, scenario: Scenario, params: dict, ckpt: Optional[Checkpoint]):
        self.scenario = scenario
        self.bundle = get_bundle(env_key, scenario.n_agents, scenario.n_obs, scenario.area, params, ckpt)
        self.reset()

    def reset(self):
        self.graph, self.ovel, self.gvel = self.bundle.initial(self.scenario)
        self.t = 0
        self.collided = np.zeros(self.bundle.n_agents, bool)
        self.reached = np.zeros(self.bundle.n_agents, bool)
        return self.snapshot()

    def step(self):
        b = self.bundle
        self.graph, self.ovel, self.gvel, _ = b.advance(self.graph, self.ovel, self.gvel)
        self.t += 1
        return self.snapshot(update_flags=True)

    def snapshot(self, update_flags: bool = False) -> dict:
        b, es, d = self.bundle, self.graph.env_states, self.bundle.dim
        coll = np.asarray(b.env.collision_mask(self.graph))
        fin = np.asarray(b.env.finish_mask(self.graph))
        if update_flags:
            self.collided |= coll
            self.reached |= fin
        return dict(t=self.t, agent=np.asarray(es.agent[:, :d]), goal=np.asarray(es.goal[:, :d]),
                    obs=np.asarray(es.obstacle.center[:, :d]), obs_r=np.asarray(es.obstacle.radius),
                    colliding=coll, at_goal=fin, ever_collided=self.collided.copy(), ever_reached=self.reached.copy(),
                    obs_vel=np.asarray(self.ovel[:, :d]), goal_vel=np.asarray(self.gvel[:, :d]))
