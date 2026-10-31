"""Numerical simulation loop for the two retained J2 architectures."""

import math
import numpy as np

from config import deg2rad
from src.dynamics import state_derivative, wrap_angle, compute_geometry
from src.attacker import attacker_png
from src.guidance_2021 import guidance_2021
from src.guidance_wb_j2 import guidance_wb_j2

ARCHITECTURES = ("2021 Baseline", "WB J2")


def run_simulation(architecture, cfg, overrides=None):
    """Run one engagement and return time histories plus terminal outcome."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    cfg_eff = dict(cfg)
    if overrides:
        cfg_eff.update(overrides)

    state = np.array([
        0.0, 0.0, deg2rad(cfg_eff["gamma_a0_deg"]),
        0.0, 0.0, deg2rad(cfg_eff["gamma_d0_deg"]),
        cfg_eff["r_am0"] * math.cos(deg2rad(cfg_eff["lambda_am0_deg"])),
        cfg_eff["r_am0"] * math.sin(deg2rad(cfg_eff["lambda_am0_deg"])),
        deg2rad(cfg_eff["gamma_m0_deg"]),
    ], dtype=float)

    state_dict = {}
    a_a_actual = 0.0
    a_d_actual = 0.0
    rng = np.random.default_rng(cfg_eff["noise_seed"])
    min_r_dm = float("inf")
    dt = cfg_eff["dt"]

    scalar_names = (
        "t", "a_a", "a_d", "a_a_actual", "a_d_actual", "a_m",
        "a_d_unsat", "a_a_unsat", "S", "phi_rm",
    )
    geom_names = (
        "r_am", "r_dm", "r_ad", "lambda_am", "lambda_dm", "lambda_ad",
        "rdot_am", "rdot_dm", "rdot_ad", "lambdadot_am",
        "lambdadot_dm", "lambdadot_ad",
    )
    logs = {name: [] for name in scalar_names + geom_names}
    state_log = []

    t = 0.0
    outcome = "timeout"
    while t < cfg_eff["t_end"]:
        geom_true = compute_geometry(state, cfg_eff)

        if cfg_eff["enable_noise"]:
            geom_guidance = dict(geom_true)
            if cfg_eff["sigma_lambda"] > 0:
                for key in ("lambda_am", "lambda_dm", "lambda_ad"):
                    geom_guidance[key] = wrap_angle(
                        geom_guidance[key]
                        + rng.normal(0.0, cfg_eff["sigma_lambda"])
                    )
            if cfg_eff["sigma_lambdadot"] > 0:
                for key in ("lambdadot_am", "lambdadot_dm", "lambdadot_ad"):
                    geom_guidance[key] += rng.normal(
                        0.0, cfg_eff["sigma_lambdadot"]
                    )
        else:
            geom_guidance = geom_true

        a_m, _ = attacker_png(geom_true, cfg_eff)

        if architecture == "2021 Baseline":
            a_a_cmd, a_d_cmd, info = guidance_2021(
                state, geom_guidance, a_m, cfg_eff, state_dict
            )
        else:
            a_a_cmd, a_d_cmd, info = guidance_wb_j2(
                state, geom_guidance, a_m, cfg_eff, state_dict
            )

        if cfg_eff["enable_lag"]:
            a_a_actual += dt * (a_a_cmd - a_a_actual) / cfg_eff["tau_ap"]
            a_d_actual += dt * (a_d_cmd - a_d_actual) / cfg_eff["tau_ap"]
        else:
            a_a_actual = a_a_cmd
            a_d_actual = a_d_cmd

        values = {
            "t": t,
            "a_a": a_a_cmd,
            "a_d": a_d_cmd,
            "a_a_actual": a_a_actual,
            "a_d_actual": a_d_actual,
            "a_m": a_m,
            "a_d_unsat": info["a_d_unsat"],
            "a_a_unsat": info["a_a_unsat"],
            "S": info["S"],
            "phi_rm": info["phi_rm"],
        }
        for key, value in values.items():
            logs[key].append(value)
        for key in geom_names:
            logs[key].append(geom_true[key])
        state_log.append(state.copy())
        min_r_dm = min(min_r_dm, geom_true["r_dm"])

        derivative = state_derivative(
            state, a_a_actual, a_d_actual, a_m, cfg_eff
        )
        state = state + dt * derivative
        state[2] = wrap_angle(state[2])
        state[5] = wrap_angle(state[5])
        state[8] = wrap_angle(state[8])
        t += dt

        if geom_true["r_dm"] < cfg_eff["hit_range_dm"]:
            outcome = "defender_intercept"
            break
        if geom_true["r_am"] < cfg_eff["hit_range_am"]:
            outcome = "attacker_hit"
            break

    result = {key: np.asarray(value, dtype=float) for key, value in logs.items()}
    result["state"] = np.asarray(state_log, dtype=float).reshape(-1, 9)
    result["outcome"] = outcome
    result["t_final"] = t
    result["miss_distance"] = min_r_dm
    result["config"] = cfg_eff
    result["architecture"] = architecture
    return result
