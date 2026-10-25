"""
ui/param_panel.py — Right-hand panel of solver parameters.

All parameters that solver/admm.py reads from the params dict are
exposed here as labelled entry widgets with sensible defaults.

Public interface (called by app.py)
------------------------------------
    ParamPanel(parent)
    .get_params()  ->  dict
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


# Default values — match the variable names in solver/admm.py
_DEFAULTS: dict[str, float | int] = {
    "N"        : 10,
    "rho"      : 1.0,
    "dt"       : 0.1,
    "tol"      : 1e-3,
    "max_iter" : 100,
    "weight_ratio" : 100.0,  
}
# Human-readable labels and types for each parameter
_PARAM_META: list[tuple[str, str, str]] = [
    # (label text,                  key,       type)
    ("N  —  horizon steps",         "N",       "int"),
    ("ρ  —  ADMM penalty",          "rho",     "float"),
    ("dt  —  timestep  (s)",        "dt",      "float"),
    ("Tolerance  (‖x−r‖²)",         "tol",     "float"),
    ("Max iterations",              "max_iter","int"),
    ("Weight Ratio  (gamma)",           "weight_ratio", "float"),
]


class ParamPanel:
    """Tkinter widget panel that owns and exposes all solver parameters."""

    def __init__(self, parent: ttk.Frame) -> None:
        self._vars: dict[str, tuple[tk.StringVar, str]] = {}
        self._build(parent)

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def _build(self, parent: ttk.Frame) -> None:
        # ── section title ─────────────────────────────────────────────
        ttk.Label(
            parent,
            text="Solver Parameters",
            font=("", 11, "bold"),
        ).pack(anchor="w", pady=(4, 10))

        # ── parameter rows ────────────────────────────────────────────
        for label, key, dtype in _PARAM_META:
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=4, pady=3)

            ttk.Label(row, text=label, width=26, anchor="w").pack(side="left")

            var = tk.StringVar(value=str(_DEFAULTS[key]))
            self._vars[key] = (var, dtype)

            entry = ttk.Entry(row, textvariable=var, width=9, justify="right")
            entry.pack(side="right")

        # ── separator + notes ─────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=12)

        notes = (
            "N:   number of timesteps in the\n"
            "     planning horizon.\n\n"
            "dt :  timestep duration in seconds.\n\n"
            "ρ:   ADMM augmented Lagrangian\n"
            "     penalty (called P in code).\n\n"
            "gamma:  weight for  goal priority over control cost.\n"
            "Tol: convergence stops when\n"
            "     ‖x − r‖² < tol."
        )
        ttk.Label(
            parent, text=notes, justify="left",
            foreground="#555555", font=("", 8),
        ).pack(anchor="w", padx=4)

        # ── reset button ──────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)
        ttk.Button(
            parent, text="↺  Reset to defaults", command=self._reset,
        ).pack(anchor="center")

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _reset(self) -> None:
        for key, (var, _) in self._vars.items():
            var.set(str(_DEFAULTS[key]))

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get_params(self) -> dict:
        """
        Read all entry widgets and return a params dict for solver().

        Falls back to the default value for any field that cannot be parsed.
        """
        result: dict = {}
        for key, (var, dtype) in self._vars.items():
            raw = var.get().strip()
            try:
                result[key] = int(raw) if dtype == "int" else float(raw)
            except ValueError:
                result[key] = _DEFAULTS[key]
        return result
