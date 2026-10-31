"""The six requested performance submetrics."""

import numpy as np


def _integral_squared(acceleration, time):
    if len(time) < 2:
        return 0.0
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(np.asarray(acceleration) ** 2, time))


def compute_metrics(simulation):
    time = simulation["t"]
    defender_actual = simulation["a_d_actual"]
    attacker_actual = simulation["a_m"]
    defender_command = simulation["a_d"]
    defender_unsaturated = simulation["a_d_unsat"]

    int_a_d2 = _integral_squared(defender_actual, time)
    int_a_m2 = _integral_squared(attacker_actual, time)

    if len(time):
        defender_saturation = float(
            np.mean(np.abs(defender_unsaturated) > simulation["config"]["a_d_max"])
        )
        terminal_acc_ratio = float(
            abs(defender_actual[-1]) / max(abs(attacker_actual[-1]), 1e-9)
        )
    else:
        defender_saturation = 0.0
        terminal_acc_ratio = 0.0

    # Total variation of the defender's commanded acceleration over the
    # final one-second window. Lower values indicate less command chatter.
    final_window = defender_command[time >= (simulation["t_final"] - 1.0)]
    chattering_tv = (
        float(np.sum(np.abs(np.diff(final_window))))
        if len(final_window) >= 2 else 0.0
    )

    return {
        "relative_control_effort": int_a_m2 / max(int_a_d2, 1e-9),
        "miss_distance": float(simulation["miss_distance"]),
        "int_a_d2": int_a_d2,
        "defender_saturation": defender_saturation,
        "chattering_TV": chattering_tv,
        "terminal_acc_ratio": terminal_acc_ratio,
    }
