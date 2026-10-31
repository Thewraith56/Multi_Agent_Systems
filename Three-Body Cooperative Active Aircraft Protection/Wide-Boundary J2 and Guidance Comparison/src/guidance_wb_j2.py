"""WB J2: Wide-Boundary J2 guidance.

This implementation is retained from project v2's src/guidance_2021.py.
It uses a saturation reaching law with boundary width smc_eps_sat.
"""
import math

from src.dynamics import wrap_angle
from src.attacker import _clip


def sat(x):
    """Unit saturation: x in [-1, 1], clipped outside that interval."""
    return max(-1.0, min(1.0, x))


def guidance_wb_j2(state, geom, a_m, cfg, state_dict):
    eta = cfg["eta"]
    Gamma = cfg["Gamma"]

    phi_rm = wrap_angle(geom["lambda_am"] - geom["lambda_dm"])
    phi_rm_dot = geom["lambdadot_am"] - geom["lambdadot_dm"]
    S = phi_rm_dot + eta * phi_rm

    reach = cfg["M2_wb"] * sat(S / cfg["smc_eps_wb"])

    a1 = math.cos(geom["g_a"] - geom["lambda_am"]) / geom["r_am"]
    a2 = math.cos(geom["g_d"] - geom["lambda_dm"]) / geom["r_dm"]

    F = (
        2.0 * geom["rdot_dm"] * geom["lambdadot_dm"] / geom["r_dm"]
        - 2.0 * geom["rdot_am"] * geom["lambdadot_am"] / geom["r_am"]
        + eta * phi_rm_dot
        + reach
    )

    denom = a2 * a2 + (a1 * Gamma) ** 2
    if abs(denom) < 1e-9:
        a_d_unsat = 0.0
        a_a_unsat = 0.0
    else:
        a_d_unsat = -a2 * F / denom
        a_a_unsat = a1 * Gamma ** 2 * F / denom

    a_d = _clip(a_d_unsat, -cfg["a_d_max"], cfg["a_d_max"])
    a_a = _clip(a_a_unsat, -cfg["a_a_max"], cfg["a_a_max"])

    info = {
        "S": S,
        "phi_rm": phi_rm,
        "F": F,
        "a_d_unsat": a_d_unsat,
        "a_a_unsat": a_a_unsat,
    }
    return a_a, a_d, info
