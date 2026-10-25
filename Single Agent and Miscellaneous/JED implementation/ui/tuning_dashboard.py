"""
ui/tuning_dashboard.py — Interactive offline tuning dashboard for exploring
Theorem 5's Pu bound before deployment. Separate from the live sim view.

Sliders: Mission Horizon K, Tolerable Violation gamma, Controller Decay
alpha, Noise Scale sigma (multiplier on the project's base Sigma), plus
secondary overridable fields for M and h(x0) (defaulting to
solver.pu_bound.compute_M(...) / a representative starting h value).

Everything recomputes and redraws live as sliders move — no re-run button.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import config as cfg
from solver import pu_bound as pub

_GAMMA_SWEEP_MAX = 5.0
_N_GAMMA_POINTS = 60
_WARN_THRESHOLD = 0.01   # 1% default


class TuningDashboard:
    def __init__(self, parent: tk.Tk, default_M: float | None = None,
                 default_h0: float | None = None) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title("Theorem 5 — Pu Bound Tuning Dashboard")
        self.win.geometry("980x720")

        R_sense_default = cfg.SENSE_SIGMA_MULT * cfg.SIGMA_D_REPRESENTATIVE
        self._default_M = default_M if default_M is not None else pub.compute_M(R_sense_default)
        self._default_h0 = default_h0 if default_h0 is not None else 0.5 * self._default_M

        self._build()
        self._redraw()

    # ── layout ───────────────────────────────────────────────────────
    def _build(self) -> None:
        top = ttk.Frame(self.win, padding=8)
        top.pack(side="top", fill="x")

        self.banner = tk.Label(top, text="", font=("", 11, "bold"), fg="white", padx=10, pady=6)
        self.banner.pack(fill="x", pady=(0, 6))

        sliders = ttk.Frame(self.win, padding=8)
        sliders.pack(side="top", fill="x")

        self.k_var = tk.IntVar(value=cfg.K_HORIZON_DEFAULT)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.alpha_var = tk.DoubleVar(value=cfg.ALPHA)
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.M_var = tk.DoubleVar(value=self._default_M)
        self.h0_var = tk.DoubleVar(value=self._default_h0)
        self.thresh_var = tk.DoubleVar(value=_WARN_THRESHOLD)

        self._add_slider(sliders, "Mission Horizon K", self.k_var, 1, 200, 1, row=0)
        self._add_slider(sliders, "Tolerable Violation \u03b3", self.gamma_var, 0.0, _GAMMA_SWEEP_MAX, 0.05, row=1)
        self._add_slider(sliders, "Decay Rate \u03b1", self.alpha_var, 0.80, 0.999, 0.001, row=2)
        self._add_slider(sliders, "Noise Scale \u03c3 (multiplier)", self.sigma_var, 0.1, 5.0, 0.05, row=3)

        adv = ttk.LabelFrame(sliders, text="Advanced (override defaults)", padding=6)
        adv.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._add_slider(adv, "M  (upper bound on h)", self.M_var, 0.1, 20.0, 0.1, row=0)
        self._add_slider(adv, "h(x0)  (starting barrier value)", self.h0_var, -5.0, 20.0, 0.1, row=1)
        self._add_slider(adv, "Warning threshold", self.thresh_var, 0.001, 0.5, 0.001, row=2)

        info = ttk.Frame(self.win, padding=(8, 0))
        info.pack(side="top", fill="x")
        self.delta_lbl = ttk.Label(info, text="", font=("", 10))
        self.delta_lbl.pack(side="left", padx=4)
        self.case_lbl = ttk.Label(info, text="", font=("", 10, "bold"))
        self.case_lbl.pack(side="left", padx=20)

        self.fig, self.ax = plt.subplots(figsize=(8.5, 4.8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=8, pady=8)

    def _add_slider(self, parent, label, var, lo, hi, step, row) -> None:
        ttk.Label(parent, text=label, width=26, anchor="w").grid(row=row, column=0, sticky="w", padx=4, pady=3)
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                           command=lambda _e: self._on_change())
        scale.grid(row=row, column=1, sticky="ew", padx=4)
        parent.columnconfigure(1, weight=1)
        val_lbl = ttk.Label(parent, width=8, anchor="e")
        val_lbl.grid(row=row, column=2, padx=4)

        def _update_label(*_):
            v = var.get()
            val_lbl.config(text=(f"{int(v)}" if isinstance(var, tk.IntVar) else f"{v:.3f}"))
        var.trace_add("write", _update_label)
        _update_label()

    # ── live recompute ──────────────────────────────────────────────
    def _on_change(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        K = int(self.k_var.get())
        gamma = float(self.gamma_var.get())
        alpha = float(self.alpha_var.get())
        sigma_mult = float(self.sigma_var.get())
        M = float(self.M_var.get())
        h0 = float(self.h0_var.get())
        threshold = float(self.thresh_var.get())

        # Theorem 6: for the LSE-relaxation polytope barrier, lambda_max = 0
        # and c_J = 0 exactly (see solver/pu_bound.compute_delta). sigma_mult
        # scales the noise covariance the dashboard treats as isotropic for
        # tuning simplicity; since lambda_max = 0, delta stays exactly 0
        # regardless of sigma_mult for THIS barrier — shown explicitly so the
        # zero isn't mistaken for a placeholder.
        cov_trace_scaled = sigma_mult ** 2 * float(np.trace(cfg.NOISE_SIGMA))
        delta = pub.compute_delta(lambda_max=0.0, cov_trace=cov_trace_scaled, c_J=0.0)

        gammas = np.linspace(0.0, _GAMMA_SWEEP_MAX, _N_GAMMA_POINTS)
        curve = pub.pu_bound_sweep(alpha, delta, M, gammas, h0, K)
        current_pu = pub.pu_bound(alpha, delta, M, gamma, h0, K)
        case = pub.which_case(delta, gamma, alpha)

        self.delta_lbl.config(
            text=f"\u03b4 (Thm 6) = c_J \u2212 \u03bb_max/2\u00b7tr(\u03a3) = 0 \u2212 0 = {delta:.5f}   "
                 f"(\u03bb_max=0 for the polytope barrier, c_J=0)")
        self.case_lbl.config(text=f"Regime: {case}")

        if current_pu > threshold:
            self.banner.config(text=f"\u26a0 UNSAFE at current settings — certified Pu = {current_pu:.4f} "
                                     f"exceeds threshold {threshold:.4f}", bg="#c0392b")
        else:
            self.banner.config(text=f"\u2713 OK — certified Pu = {current_pu:.4f}  "
                                     f"(threshold {threshold:.4f})", bg="#27ae60")

        self.ax.clear()
        self.ax.plot(gammas, curve, color="#1a6faf", linewidth=2, label="Pu bound (Thm 5)")
        self.ax.plot([gamma], [current_pu], "o", color="#d62728", markersize=9, zorder=5,
                     label=f"current \u03b3={gamma:.2f} \u2192 Pu={current_pu:.4f}")
        self.ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold={threshold:.3f}")
        self.ax.set_xlabel("\u03b3  (tolerable violation)")
        self.ax.set_ylabel("Pu bound")
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_title(f"Theorem 5 Pu bound vs \u03b3   (K={K}, \u03b1={alpha:.3f}, M={M:.2f}, h0={h0:.2f})")
        self.ax.legend(fontsize=8, loc="upper left")
        self.ax.grid(alpha=0.3)
        self.canvas.draw_idle()


def open_dashboard(parent: tk.Tk, default_M: float | None = None, default_h0: float | None = None) -> None:
    TuningDashboard(parent, default_M=default_M, default_h0=default_h0)
