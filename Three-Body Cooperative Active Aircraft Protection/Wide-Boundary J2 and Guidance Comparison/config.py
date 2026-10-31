"""Simulation parameters and the five retained evaluation scenarios."""

import math

CONFIG = {
    # Vehicle speeds (m/s)
    "v_a": 100.0,
    "v_d": 200.0,
    "v_m": 200.0,

    # Acceleration limits (m/s^2)
    "a_a_max": 10.0 * 9.81,
    "a_d_max": 20.0 * 9.81,
    "a_m_max": 20.0 * 9.81,

    # Initial engagement geometry
    "r_am0": 5000.0,
    "gamma_a0_deg": 45.0,
    "gamma_m0_deg": 150.0,
    "gamma_d0_deg": 0.0,
    "lambda_am0_deg": 0.0,

    # Attacker proportional navigation
    "N_png": 5.0,

    # Shared J2 guidance parameters
    "eta": 10.0,
    "Gamma": 0.5,          # FIXED: matches w_a/w_d = 10g/20g in the paper

    # 2021 Baseline: narrow-boundary tanh reaching law
    "M2_2021": 0.1,        # FIXED: matches M2 = 0.1 in the paper
    "smc_eps": 0.1,        # Set to 0.1 so it behaves as a true smooth sigmoid

    # WB J2: wide-boundary saturation reaching law
    "M2_wb": 0.1,          # Keep equal to M2_2021 for a fair comparison
    "smc_eps_wb": 1.0,

    # Numerical integration and termination
    "dt": 0.001,
    "t_end": 30.0,
    "hit_range_dm": 5.0,
    "hit_range_am": 10.0,

    # Non-ideal conditions
    "enable_lag": False,
    "tau_ap": 0.05,
    "enable_noise": False,
    "sigma_lambda": 0.0,
    "sigma_lambdadot": 0.0,
    "noise_seed": 42,
}

SCENARIOS = {
    "ideal": {},
    "lag": {"enable_lag": True, "tau_ap": 0.05},
    "noise": {
        "enable_noise": True,
        "sigma_lambda": 0.0,
        "sigma_lambdadot": 0.002,
    },
    "noise+lag": {
        "enable_lag": True,
        "tau_ap": 0.05,
        "enable_noise": True,
        "sigma_lambda": 0.0,
        "sigma_lambdadot": 0.002,
    },
    "tight_miss": {"hit_range_dm": 1.0},
}

def deg2rad(value):
    return value * math.pi / 180.0

def rad2deg(value):
    return value * 180.0 / math.pi