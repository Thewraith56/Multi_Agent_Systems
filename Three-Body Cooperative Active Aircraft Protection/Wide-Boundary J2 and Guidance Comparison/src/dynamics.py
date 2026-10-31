"""Planar point-mass kinematics and engagement geometry."""
import math
import numpy as np


def state_derivative(state, a_a, a_d, a_m, cfg):
    speeds = (cfg["v_a"], cfg["v_d"], cfg["v_m"])
    accels = (a_a, a_d, a_m)
    deriv = np.zeros(9)
    for i in range(3):
        g = float(state[3 * i + 2])
        v = speeds[i]
        deriv[3 * i] = v * math.cos(g)
        deriv[3 * i + 1] = v * math.sin(g)
        deriv[3 * i + 2] = accels[i] / v
    return deriv


def wrap_angle(x):
    """Wrap to [-pi, pi]; odd multiples of pi keep the sign of x."""
    two_pi = 2.0 * math.pi
    y = math.fmod(x + math.pi, two_pi)
    if y < 0.0:
        y += two_pi
    y -= math.pi
    if y == -math.pi and x > 0.0:
        y = math.pi
    return y


def _pair(x_x, y_x, g_x, v_x, x_y, y_y, g_y, v_y):
    dx = x_y - x_x
    dy = y_y - y_x
    r = math.sqrt(dx * dx + dy * dy)
    lam = wrap_angle(math.atan2(dy, dx))
    rdot = v_y * math.cos(g_y - lam) - v_x * math.cos(g_x - lam)
    if r < 1e-9:
        lamdot = 0.0
    else:
        lamdot = (v_y * math.sin(g_y - lam) - v_x * math.sin(g_x - lam)) / r
    return r, lam, rdot, lamdot


def compute_geometry(state, cfg=None):
    """Speeds come from cfg (defaults to CONFIG when cfg is None)."""
    if cfg is None:
        from config import CONFIG
        cfg = CONFIG
    v_a = cfg["v_a"]
    v_d = cfg["v_d"]
    v_m = cfg["v_m"]
    x_a, y_a, g_a = float(state[0]), float(state[1]), float(state[2])
    x_d, y_d, g_d = float(state[3]), float(state[4]), float(state[5])
    x_m, y_m, g_m = float(state[6]), float(state[7]), float(state[8])

    r_am, lambda_am, rdot_am, lambdadot_am = _pair(x_a, y_a, g_a, v_a, x_m, y_m, g_m, v_m)
    r_dm, lambda_dm, rdot_dm, lambdadot_dm = _pair(x_d, y_d, g_d, v_d, x_m, y_m, g_m, v_m)
    r_ad, lambda_ad, rdot_ad, lambdadot_ad = _pair(x_a, y_a, g_a, v_a, x_d, y_d, g_d, v_d)

    return {
        "r_am": r_am, "r_dm": r_dm, "r_ad": r_ad,
        "lambda_am": lambda_am, "lambda_dm": lambda_dm, "lambda_ad": lambda_ad,
        "rdot_am": rdot_am, "rdot_dm": rdot_dm, "rdot_ad": rdot_ad,
        "lambdadot_am": lambdadot_am, "lambdadot_dm": lambdadot_dm,
        "lambdadot_ad": lambdadot_ad,
        "g_a": g_a, "g_d": g_d, "g_m": g_m,
    }


def compute_lambda_ddot_am(state, a_a, a_m, geom, cfg):
    r_am = geom["r_am"]
    if r_am < 1e-9:
        return 0.0
    lam = geom["lambda_am"]
    num = (-2.0 * geom["rdot_am"] * geom["lambdadot_am"]
           + a_m * math.cos(geom["g_m"] - lam)
           - a_a * math.cos(geom["g_a"] - lam))
    return num / r_am
