"""2021 Baseline: Narrow-Boundary J2 (NB J2) cooperative guidance.

This implementation is retained from project v1. Its reaching term is
M2_2021 * tanh(S / smc_eps).
"""
import math
from src.dynamics import wrap_angle
from src.attacker import _clip


def guidance_2021(state, geom, a_m, cfg, state_dict):
    eta = cfg["eta"]
    Gamma = cfg["Gamma"]

    phi_rm = wrap_angle(geom["lambda_am"] - geom["lambda_dm"])
    phi_rm_dot = geom["lambdadot_am"] - geom["lambdadot_dm"]
    S = phi_rm_dot + eta * phi_rm
    sgn_S = math.tanh(S / cfg["smc_eps"])

    a1 = math.cos(geom["g_a"] - geom["lambda_am"]) / geom["r_am"]
    a2 = math.cos(geom["g_d"] - geom["lambda_dm"]) / geom["r_dm"]

    F = (2.0 * geom["rdot_dm"] * geom["lambdadot_dm"] / geom["r_dm"]
         - 2.0 * geom["rdot_am"] * geom["lambdadot_am"] / geom["r_am"]
         + eta * phi_rm_dot
         + cfg["M2_2021"] * sgn_S)

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
        "S": S, "phi_rm": phi_rm, "F": F,
        "a_d_unsat": a_d_unsat, "a_a_unsat": a_a_unsat,
    }
    return a_a, a_d, info
