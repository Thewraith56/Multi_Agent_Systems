"""
ui/param_panel.py — Right-hand panel of solver parameters.

Updated for multi-architecture, multi-vehicle support.

Changes from original param_panel.py
--------------------------------------
  ADDED  — gamma_cbf, lambda_clf, slack_penalty, slack_penalty_clf,
           terminal_scale params
  ADDED  — architecture combobox ('i', 'ii')
  ADDED  — _apply_arch_state() — greys out irrelevant fields per architecture
  ADDED  — get_params() auto-zeroes terminal_scale for arch ii
  FIXED  — slack_penalty and slack_penalty_clf used to share one entry
           field, so tuning "p" while on arch ii silently had no effect
           (admm.py reads slack_penalty_clf for arch ii, not
           slack_penalty). They're now two separate fields, each active
           only for the architecture that actually uses it.

Public interface (called by app.py)
------------------------------------
    ParamPanel(parent)
    .get_params()  ->  dict   (includes 'architecture' key)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


# ── Numeric parameter defaults ────────────────────────────────────────────

_DEFAULTS: dict[str, float | int] = {
    # Core ADMM
    "N"             : 20,
    "rho"           : 10.0,
    "dt"            : 0.1,
    "tol"           : 1e-4,
    "max_iter"      : 100,
    "weight_ratio"  : 100.0,
    # CBF / CLF / slack
    "gamma_cbf"         : 0.5,
    "lambda_clf"        : 0.3,
    "slack_penalty"     : 1000.0,
    "slack_penalty_clf" : 10.0,
    "terminal_scale"    : 80.0,
}

# ── Core ADMM params — always active ─────────────────────────────────────

_CORE_PARAMS: list[tuple[str, str, str]] = [
    # (display label,                key,            dtype)
    ("N  —  horizon steps",         "N",            "int"),
    ("ρ  —  ADMM penalty",          "rho",          "float"),
    ("dt  —  timestep  (s)",        "dt",           "float"),
    ("Tolerance  (‖x−r‖²)",         "tol",          "float"),
    ("Max iterations",              "max_iter",     "int"),
    ("Weight ratio  (α)",           "weight_ratio", "float"),
]

# ── Safety / convergence params — some greyed per architecture ────────────

_SAFETY_PARAMS: list[tuple[str, str, str]] = [
    ("γ_cbf  —  CBF decay",         "gamma_cbf",    "float"),
    ("λ_clf  —  CLF decay",         "lambda_clf",   "float"),
    ("p  —  slack penalty (CBF)",   "slack_penalty","float"),
    ("p_clf — slack penalty (CLF)", "slack_penalty_clf", "float"),
    ("Terminal scale",              "terminal_scale","float"),
]

# ── Architecture metadata ─────────────────────────────────────────────────

_ARCH_LABEL = {
    "i"  : "CBF(soft) + terminal — no CLF",
    "ii" : "CBF(hard) + CLF(soft slack) — no terminal",
}

# Which safety params are ACTIVE for each architecture
# (inactive params are still read but grayed out for clarity)
_ARCH_ACTIVE: dict[str, set[str]] = {
    "i"  : {"gamma_cbf", "slack_penalty",     "terminal_scale"},
    "ii" : {"gamma_cbf", "lambda_clf",        "slack_penalty_clf"},
}

_SAFETY_KEYS = {p[1] for p in _SAFETY_PARAMS}   # fast lookup


class ParamPanel:
    """Tkinter widget panel that owns and exposes all solver parameters."""

    def __init__(self, parent: ttk.Frame) -> None:
        self._vars    : dict[str, tuple[tk.StringVar, str]] = {}
        self._entries : dict[str, ttk.Entry]                = {}
        self._arch_var     = tk.StringVar(value="i")
        self._arch_lbl_var = tk.StringVar(value=_ARCH_LABEL["i"])
        self._build(parent)

    # ── Construction ──────────────────────────────────────────────────────

    def _build(self, parent: ttk.Frame) -> None:

        # ── title ─────────────────────────────────────────────────────
        ttk.Label(
            parent,
            text="Solver Parameters",
            font=("", 11, "bold"),
        ).pack(anchor="w", pady=(4, 8))

        # ── core ADMM params ──────────────────────────────────────────
        for label, key, dtype in _CORE_PARAMS:
            self._add_row(parent, label, key, dtype)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=7)

        # ── architecture selector ─────────────────────────────────────
        ttk.Label(
            parent,
            text="Architecture",
            font=("", 9, "bold"),
        ).pack(anchor="w", padx=4)

        arch_frame = ttk.Frame(parent)
        arch_frame.pack(fill="x", padx=4, pady=(2, 4))

        self._arch_combo = ttk.Combobox(
            arch_frame,
            textvariable=self._arch_var,
            values=["i", "ii"],
            state="readonly",
            width=4,
        )
        self._arch_combo.pack(side="left")
        self._arch_combo.bind("<<ComboboxSelected>>", self._on_arch_changed)

        ttk.Label(
            arch_frame,
            textvariable=self._arch_lbl_var,
            foreground="#1a6faf",
            font=("", 8),
        ).pack(side="left", padx=(8, 0))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=7)

        # ── CBF / CLF params ──────────────────────────────────────────
        ttk.Label(
            parent,
            text="CBF / CLF Parameters",
            font=("", 9, "bold"),
        ).pack(anchor="w", padx=4)

        for label, key, dtype in _SAFETY_PARAMS:
            self._add_row(parent, label, key, dtype)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=7)

        # ── notes ─────────────────────────────────────────────────────
        notes = (
            "ρ:  ADMM augmented Lagrangian penalty.\n"
            "α:  weight ratio, goal vs control effort.\n\n"
            "γ_cbf: HOCBF decay per step (drives both stages\n"
            "    of the barrier chain — see cbf.py).\n"
            "    Lower → more conservative.\n"
            "    Active for all architectures.\n\n"
            "λ_clf: required V reduction per step.\n"
            "    Active for arch ii only.\n\n"
            "p (CBF):  slack penalty on CBF violations.\n"
            "    Active for arch i.  Higher → tighter safety.\n\n"
            "p_clf (CLF):  slack penalty on CLF violations.\n"
            "    Active for arch ii.  Keep this LOW (~10) —\n"
            "    unlike p, a large p_clf stalls the solver at\n"
            "    the very first iteration (see admm.py notes).\n\n"
            "Terminal scale: weight on final state.\n"
            "    Auto-zeroed for arch ii\n"
            "    (CLF drives convergence instead)."
        )
        ttk.Label(
            parent,
            text=notes,
            justify="left",
            foreground="#555555",
            font=("", 8),
        ).pack(anchor="w", padx=4)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=7)

        # ── reset button ──────────────────────────────────────────────
        ttk.Button(
            parent,
            text="↺  Reset to defaults",
            command=self._reset,
        ).pack(anchor="center")

        # Apply initial greyed-out state for arch 'i'
        self._apply_arch_state("i")

    def _add_row(
        self,
        parent : ttk.Frame,
        label  : str,
        key    : str,
        dtype  : str,
    ) -> None:
        """Create one labelled entry row and register it."""
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)

        ttk.Label(row, text=label, width=26, anchor="w").pack(side="left")

        var   = tk.StringVar(value=str(_DEFAULTS[key]))
        entry = ttk.Entry(row, textvariable=var, width=9, justify="right")
        entry.pack(side="right")

        self._vars[key]    = (var, dtype)
        self._entries[key] = entry

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_arch_changed(self, _event=None) -> None:
        arch = self._arch_var.get()
        self._arch_lbl_var.set(_ARCH_LABEL.get(arch, ""))
        self._apply_arch_state(arch)

    def _apply_arch_state(self, arch: str) -> None:
        """
        Enable/disable safety-param entry fields based on architecture.

        Fields that are unused by the chosen architecture are shown in
        a disabled (greyed-out) state so the user knows they have no effect.
        Core ADMM params (N, rho, …) are always enabled.
        """
        active = _ARCH_ACTIVE.get(arch, set())
        for key, entry in self._entries.items():
            if key in _SAFETY_KEYS:
                entry.config(state="normal" if key in active else "disabled")

    def _reset(self) -> None:
        """Restore all fields and the architecture combo to their defaults."""
        for key, (var, _) in self._vars.items():
            var.set(str(_DEFAULTS[key]))
        self._arch_var.set("i")
        self._arch_lbl_var.set(_ARCH_LABEL["i"])
        self._apply_arch_state("i")

    # ── Public interface ──────────────────────────────────────────────────

    def get_params(self) -> dict:
        """
        Read all entry widgets and return a params dict for solver().

        Falls back silently to the default value for any field that cannot
        be parsed (e.g. empty or non-numeric entry).

        Architecture-specific overrides applied automatically:
          arch 'ii' → terminal_scale forced to 0.0
              (CLF constraint drives goal convergence; terminal cost
               would create redundant and potentially conflicting pressure)
        """
        result: dict = {}

        for key, (var, dtype) in self._vars.items():
            raw = var.get().strip()
            try:
                result[key] = int(raw) if dtype == "int" else float(raw)
            except ValueError:
                result[key] = _DEFAULTS[key]

        arch = self._arch_var.get()
        result["architecture"] = arch

        # Auto-zero terminal_scale for architectures that use a CLF
        if arch== "ii" :
            result["terminal_scale"] = 0.0

        return result
