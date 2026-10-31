"""Attacker proportional navigation against the aircraft."""


def _clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def attacker_png(geom, cfg):
    a_m_unsat = cfg["N_png"] * cfg["v_m"] * geom["lambdadot_am"]
    a_m = _clip(a_m_unsat, -cfg["a_m_max"], cfg["a_m_max"])
    return a_m, a_m_unsat
