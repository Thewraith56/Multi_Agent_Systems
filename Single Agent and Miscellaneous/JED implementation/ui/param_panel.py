"""
ui/param_panel.py — Right-hand panel of JED / DTCBF parameters.

CHANGED from the ADMM/MPC-era param_panel.py:
  - No CLF params (lambda_clf, slack_penalty_clf) — no CLF exists anymore.
  - No probabilistic-CBF method selector (Hoeffding/Scenario/Markov/
    Cantelli) — this build only ever uses the JED (Theorem 6) controller
    with the LSE-relaxation polytope barrier.
  - New sections: A* planner, LQR tracker, DTCBF/JED, noise (mean +
    diagonal covariance), Monte-Carlo validation settings.

Public interface (called by app.py): ParamPanel(parent).get_params() -> dict
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import config as cfg


_DEFAULTS: dict[str, float | int | str] = {
    # DTCBF / JED
    "alpha"          : cfg.ALPHA,
    "r_robot"        : cfg.R_ROBOT,
    "a_max"          : cfg.A_MAX,
    "dt"             : cfg.DT,

    # A* planner
    "grid_res"       : cfg.GRID_RES,
    "planner_buffer" : cfg.PLANNER_BUFFER,
    "rep_speed"      : cfg.REPRESENTATIVE_SPEED,

    # LQR tracker
    "lqr_q_pos"      : float(cfg.LQR_Q[0, 0]),
    "lqr_q_vel"      : float(cfg.LQR_Q[2, 2]),
    "lqr_r"          : float(cfg.LQR_R[0, 0]),
    "switch_radius"  : cfg.SWITCH_RADIUS,
    "goal_tolerance" : cfg.GOAL_TOLERANCE,
    "goal_speed_tol" : cfg.GOAL_SPEED_TOL,

    # Noise: d_k ~ N(mu, Sigma) (mean + diagonal covariance)
    "mu_x"           : float(cfg.NOISE_MU[0]),
    "mu_y"           : float(cfg.NOISE_MU[1]),
    "mu_vx"          : float(cfg.NOISE_MU[2]),
    "mu_vy"          : float(cfg.NOISE_MU[3]),
    "sigma_x"        : float(np.sqrt(cfg.NOISE_SIGMA[0, 0])),
    "sigma_y"        : float(np.sqrt(cfg.NOISE_SIGMA[1, 1])),
    "sigma_vx"       : float(np.sqrt(cfg.NOISE_SIGMA[2, 2])),
    "sigma_vy"       : float(np.sqrt(cfg.NOISE_SIGMA[3, 3])),

    # Closed loop / Monte Carlo
    "n_steps"        : cfg.N_STEPS_DEFAULT,
    "n_trials"       : cfg.N_TRIALS_DEFAULT,
    "K_horizon"      : cfg.K_HORIZON_DEFAULT,
}

_SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("DTCBF / JED safety filter", [
        ("alpha  —  DTCBF decay (close to 1)", "alpha", "float"),
        ("r_robot  —  robot radius (m)",       "r_robot", "float"),
        ("a_max  —  accel limit (m/s^2)",      "a_max", "float"),
        ("dt  —  timestep (s)",                "dt", "float"),
    ]),
    ("A* global planner", [
        ("grid_res  —  A* cell size (m)",      "grid_res", "float"),
        ("planner_buffer  —  extra inflation (m)", "planner_buffer", "float"),
        ("rep_speed  —  A* inflation speed (m/s)", "rep_speed", "float"),
    ]),
    ("LQR waypoint tracker", [
        ("Q position weight",                  "lqr_q_pos", "float"),
        ("Q velocity weight",                  "lqr_q_vel", "float"),
        ("R control weight",                   "lqr_r", "float"),
        ("switch_radius (m)",                  "switch_radius", "float"),
        ("goal_tolerance (m)",                 "goal_tolerance", "float"),
        ("goal_speed_tol (m/s)",               "goal_speed_tol", "float"),
    ]),
    ("Noise mean  d~N(mu,Sigma)", [
        ("mu_x (m)",  "mu_x", "float"), ("mu_y (m)",  "mu_y", "float"),
        ("mu_vx (m/s)", "mu_vx", "float"), ("mu_vy (m/s)", "mu_vy", "float"),
    ]),
    ("Noise std (diagonal)", [
        ("sigma_x (m)", "sigma_x", "float"), ("sigma_y (m)", "sigma_y", "float"),
        ("sigma_vx (m/s)", "sigma_vx", "float"), ("sigma_vy (m/s)", "sigma_vy", "float"),
    ]),
    ("Closed loop / Monte Carlo", [
        ("Max closed-loop steps", "n_steps", "int"),
        ("MC trials",             "n_trials", "int"),
        ("Pu horizon K",          "K_horizon", "int"),
    ]),
]


class ParamPanel:
    def __init__(self, parent: ttk.Frame) -> None:
        self._vars: dict[str, tuple[tk.StringVar, str]] = {}
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        ttk.Label(body, text="JED / DTCBF Parameters", font=("", 11, "bold")).pack(
            anchor="w", pady=(4, 8), padx=4)

        for title, rows in _SECTIONS:
            ttk.Label(body, text=title, font=("", 9, "bold")).pack(anchor="w", padx=4, pady=(6, 0))
            for label, key, dtype in rows:
                self._add_row(body, label, key, dtype)
            ttk.Separator(body, orient="horizontal").pack(fill="x", pady=6)

        notes = (
            "alpha: kept close to 1 (0.95-0.99) — the LSE-relaxation\n"
            "  constraint already accounts for the full noise distribution,\n"
            "  so unlike a naive deterministic CBF it doesn't need to be\n"
            "  tuned low to stay feasible.\n\n"
            "c_J = 0 for this barrier (see solver/pu_bound.py) — the LSE\n"
            "  bound already directly bounds E[h], no separate Jensen\n"
            "  correction is layered on top."
        )
        ttk.Label(body, text=notes, justify="left", foreground="#555555", font=("", 8)).pack(
            anchor="w", padx=4)
        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(body, text="\u21ba  Reset to defaults", command=self._reset).pack(
            anchor="center", pady=(0, 8))

    def _add_row(self, parent, label, key, dtype) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=30, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(_DEFAULTS[key]))
        entry = ttk.Entry(row, textvariable=var, width=8, justify="right")
        entry.pack(side="right")
        self._vars[key] = (var, dtype)

    def _reset(self) -> None:
        for key, (var, _) in self._vars.items():
            var.set(str(_DEFAULTS[key]))

    def get_params(self) -> dict:
        raw: dict = {}
        for key, (var, dtype) in self._vars.items():
            s = var.get().strip()
            try:
                raw[key] = int(s) if dtype == "int" else float(s)
            except ValueError:
                raw[key] = _DEFAULTS[key]

        params = {
            "alpha": raw["alpha"], "r_robot": raw["r_robot"], "a_max": raw["a_max"],
            "dt": raw["dt"], "grid_res": raw["grid_res"], "planner_buffer": raw["planner_buffer"],
            "rep_speed": raw["rep_speed"],
            "lqr_Q": np.diag([raw["lqr_q_pos"], raw["lqr_q_pos"], raw["lqr_q_vel"], raw["lqr_q_vel"]]),
            "lqr_R": np.diag([raw["lqr_r"], raw["lqr_r"]]),
            "switch_radius": raw["switch_radius"], "goal_tolerance": raw["goal_tolerance"],
            "goal_speed_tol": raw["goal_speed_tol"],
            "mu": np.array([raw["mu_x"], raw["mu_y"], raw["mu_vx"], raw["mu_vy"]]),
            "Sigma": np.diag([raw["sigma_x"] ** 2, raw["sigma_y"] ** 2,
                               raw["sigma_vx"] ** 2, raw["sigma_vy"] ** 2]),
            "sigma_d": max(raw["sigma_x"], raw["sigma_y"]),
            "n_steps": raw["n_steps"], "n_trials": raw["n_trials"], "K_horizon": raw["K_horizon"],
        }
        return params
