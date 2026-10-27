from typing import Optional

from .base import MultiAgentEnv
from .linear_drone import LinearDrone

ENV = {
    "LinearDrone": LinearDrone,
}

DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        params: Optional[dict] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f"Environment {env_id} not implemented."
    p = dict(ENV[env_id].PARAMS)
    if params:
        p.update(params)
    dt = p.pop("dt", 0.03)
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        p["n_obs"] = num_obs
    if n_rays is not None:
        p["n_rays"] = n_rays
    return ENV[env_id](
        num_agents=num_agents, area_size=area_size, max_step=max_step,
        max_travel=max_travel, dt=dt, params=p)
