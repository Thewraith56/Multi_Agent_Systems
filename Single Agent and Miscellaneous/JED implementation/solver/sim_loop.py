"""
solver/sim_loop.py — Wires the 3-layer architecture together:

    A* (planner.py) -> LQR (tracker.py) -> JED safety filter (jed_solver.py)
                                          -> dynamics.py (noisy step)

and provides Monte-Carlo validation of Theorem 5's Pu bound against
empirical exit rates (pu_bound.py).
"""

from __future__ import annotations

import numpy as np

from . import config
from .vehicles.drone import DroneVehicle
from .noise import GaussianProcessNoise
from .planner import astar_plan, shortcut_waypoints
from .tracker import design_lqr_gain, WaypointTracker
from .jed_solver import jed_qp_solve
from .barriers import build_active_polytope, sense_radius, h_polytope
from . import pu_bound as pub


def plan_and_track(start_xy: np.ndarray, goal_xy: np.ndarray, obstacles: list[dict],
                    vehicle: DroneVehicle, params: dict) -> WaypointTracker | None:
    """Layer 1 + set up layer 2. Returns None if A* finds no path."""
    waypoints = astar_plan(
        start_xy, goal_xy, obstacles,
        grid_res=params.get("grid_res", config.GRID_RES),
        r_robot=params.get("r_robot", config.R_ROBOT),
        buffer=params.get("planner_buffer", config.PLANNER_BUFFER),
        world_bounds=params.get("world_bounds", config.WORLD_BOUNDS),
        rep_speed=params.get("rep_speed", config.REPRESENTATIVE_SPEED),
        a_max=params.get("a_max", config.A_MAX),
    )
    if not waypoints:
        return None
    waypoints = shortcut_waypoints(
        waypoints, obstacles,
        r_robot=params.get("r_robot", config.R_ROBOT),
        buffer=params.get("planner_buffer", config.PLANNER_BUFFER),
        rep_speed=params.get("rep_speed", config.REPRESENTATIVE_SPEED),
        a_max=params.get("a_max", config.A_MAX),
    )
    K_gain = design_lqr_gain(vehicle.A, vehicle.B,
                              Q=params.get("lqr_Q", config.LQR_Q),
                              R=params.get("lqr_R", config.LQR_R))
    tracker = WaypointTracker(
        waypoints, K_gain,
        switch_radius=params.get("switch_radius", config.SWITCH_RADIUS),
        goal_tolerance=params.get("goal_tolerance", config.GOAL_TOLERANCE),
        goal_speed_tol=params.get("goal_speed_tol", config.GOAL_SPEED_TOL),
    )
    return tracker


def run_closed_loop(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[dict],
    vehicle: DroneVehicle,
    params: dict,
    real_noise: GaussianProcessNoise,
    n_steps: int = config.N_STEPS_DEFAULT,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    One full closed-loop rollout. Returns a dict with x_hist, u_hist, h_hist,
    slack_hist, waypoints, reached_goal, min_h, n_slack_active.
    """
    alpha = params.get("alpha", config.ALPHA)
    r_robot = params.get("r_robot", config.R_ROBOT)
    a_max = params.get("a_max", config.A_MAX)
    dt = params.get("dt", config.DT)

    tracker = plan_and_track(start_xy, goal_xy, obstacles, vehicle, params)
    x0 = vehicle.initial_state(np.array([start_xy[0], start_xy[1], 0.0, 0.0, 0.0]))

    result = {
        "x_hist": [x0.copy()], "u_hist": [], "h_hist": [], "slack_hist": [],
        "t_used_hist": [], "waypoints": None, "reached_goal": False,
        "min_h": np.inf, "n_slack_active": 0, "planning_failed": tracker is None,
    }
    if tracker is None:
        return result
    result["waypoints"] = tracker.waypoints

    x = x0.copy()
    for _ in range(n_steps):
        v = vehicle.velocity(x)
        R_sense = sense_radius(v, dt=dt, sigma_d=params.get("sigma_d", config.SIGMA_D_REPRESENTATIVE))
        C, w = build_active_polytope(x, obstacles, R_sense, r_robot=r_robot, v=v, a_max=a_max)[:2]

        u_nom = tracker.get_u_nom(x)

        M_sat = pub.compute_M(R_sense, params.get("world_bounds", config.WORLD_BOUNDS))
        jed = jed_qp_solve(
            x, u_nom, vehicle.A, vehicle.B,
            params.get("mu", config.NOISE_MU), params.get("Sigma", config.NOISE_SIGMA),
            C, w, alpha=alpha, M_saturate=M_sat,
        )

        d_k = real_noise.sample() if rng is None else real_noise.rng.multivariate_normal(real_noise.mu, real_noise.Sigma)
        x = vehicle.step(x, jed.u, d_k)

        result["x_hist"].append(x.copy())
        result["u_hist"].append(jed.u.copy())
        result["h_hist"].append(jed.h_x)
        result["slack_hist"].append(jed.slack)
        result["t_used_hist"].append(jed.t_used)
        if jed.slack and jed.slack > 1e-6:
            result["n_slack_active"] += 1
        result["min_h"] = min(result["min_h"], jed.h_x)

        if tracker.reached_goal(x):
            result["reached_goal"] = True
            break

    result["x_hist"] = np.array(result["x_hist"])
    return result


def run_monte_carlo(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[dict],
    vehicle: DroneVehicle,
    params: dict,
    n_trials: int = config.N_TRIALS_DEFAULT,
    K_horizon: int = config.K_HORIZON_DEFAULT,
    gammas=config.GAMMA_SWEEP,
    seed: int | None = 0,
) -> dict:
    """
    Runs n_trials closed-loop rollouts and computes empirical Pu(K, gamma)
    (fraction of trials where min_k h(xk) < -gamma over the first K steps),
    alongside the theoretical Theorem 5 bound for the same gammas.

    Also reports the fraction of trials/steps where the feasibility slack
    was active (breaks the strict guarantee).
    """
    rng = np.random.default_rng(seed)
    alpha = params.get("alpha", config.ALPHA)
    mu = params.get("mu", config.NOISE_MU)
    Sigma = params.get("Sigma", config.NOISE_SIGMA)

    h_traj_all = []
    slack_active_steps = 0
    total_steps = 0
    trials_with_slack = 0
    h0 = None

    for trial in range(n_trials):
        real_noise = GaussianProcessNoise(mu, Sigma, rng=rng)
        rollout = run_closed_loop(start_xy, goal_xy, obstacles, vehicle, params,
                                   real_noise, n_steps=K_horizon, rng=rng)
        h_hist = rollout["h_hist"]
        if not h_hist:
            continue
        if h0 is None:
            R_sense0 = sense_radius(None, dt=params.get("dt", config.DT))
            M0 = pub.compute_M(R_sense0, params.get("world_bounds", config.WORLD_BOUNDS))
            C0, w0 = build_active_polytope(rollout["x_hist"][0], obstacles, R_sense0,
                                            r_robot=params.get("r_robot", config.R_ROBOT))[:2]
            h0 = h_polytope(rollout["x_hist"][0], C0, w0, M_saturate=M0)

        h_pad = h_hist + [h_hist[-1]] * max(0, K_horizon - len(h_hist))
        h_traj_all.append(h_pad[:K_horizon])

        n_slack = rollout["n_slack_active"]
        slack_active_steps += n_slack
        total_steps += len(h_hist)
        if n_slack > 0:
            trials_with_slack += 1

    h_traj_all = np.array(h_traj_all) if h_traj_all else np.zeros((0, K_horizon))

    R_sense_rep = sense_radius(np.array([config.REPRESENTATIVE_SPEED, 0.0]))
    M = pub.compute_M(R_sense_rep, params.get("world_bounds", config.WORLD_BOUNDS))
    delta = pub.compute_delta()   # 0 for the LSE-relaxation polytope barrier
    h0 = h0 if h0 is not None else 0.0

    empirical_pu = []
    theoretical_pu = []
    for gamma in gammas:
        if h_traj_all.shape[0] > 0:
            min_h_per_trial = h_traj_all.min(axis=1)
            emp = float(np.mean(min_h_per_trial < -gamma))
        else:
            emp = float("nan")
        empirical_pu.append(emp)
        theoretical_pu.append(pub.pu_bound(alpha, delta, M, gamma, h0, K_horizon))

    return {
        "gammas": list(gammas),
        "empirical_pu": empirical_pu,
        "theoretical_pu": theoretical_pu,
        "n_trials": n_trials,
        "K_horizon": K_horizon,
        "M": M, "delta": delta, "h0": h0, "alpha": alpha,
        "slack_active_steps": slack_active_steps,
        "total_steps": total_steps,
        "frac_steps_with_slack": (slack_active_steps / total_steps) if total_steps else 0.0,
        "trials_with_slack": trials_with_slack,
        "frac_trials_with_slack": (trials_with_slack / n_trials) if n_trials else 0.0,
    }
