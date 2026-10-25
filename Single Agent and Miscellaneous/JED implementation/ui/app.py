"""
ui/app.py — Root Tkinter window for the JED / DTCBF drone navigation stack.

CHANGED from the ADMM/MPC-era app.py:
  - "Run" now executes the 3-layer pipeline (A* -> LQR -> JED), not an
    MPC solve. There's no separate "nominal vs closed-loop" pair of solves
    any more — A* + LQR give u_nom on the fly, JED filters it every step,
    and the result is the single executed trajectory (real disturbance
    injected every step, same as before).
  - New "Run Monte-Carlo Validation" button: runs N_trials rollouts and
    shows the Theorem 5 Pu bound vs. empirical exit rate (paper's Fig. 2/4
    style plot), plus a slack-usage warning if any Theorem-5 guarantees
    were broken by the feasibility slack.
  - New "Tuning Dashboard" button: opens the offline, sim-independent
    Theorem-5 exploration tool (ui/tuning_dashboard.py) with live sliders.
  - No vehicle selector — drone-only, per the brief.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.scene_editor import SceneEditor
from ui.param_panel import ParamPanel
from ui.tuning_dashboard import open_dashboard

from solver.vehicles.drone import DroneVehicle
from solver.noise import GaussianProcessNoise
from solver.sim_loop import run_closed_loop, run_monte_carlo
from solver import config as cfg
from solver.barriers import sense_radius
from solver.pu_bound import compute_M

_WORLD_BOUNDS = cfg.WORLD_BOUNDS


class App:
    TITLE = "Drone JED — Discrete-Time Control Barrier Functions"
    MIN_W = 1250
    MIN_H = 740
    SCENE_WEIGHT = 3
    PARAM_WEIGHT = 1

    def __init__(self) -> None:
        self._rollout: dict | None = None
        self._solver_thread: threading.Thread | None = None
        self._last_goal: np.ndarray | None = None
        self.vehicle = DroneVehicle(dt=cfg.DT)

        self.root = tk.Tk()
        self.root.title(self.TITLE)
        self.root.minsize(self.MIN_W, self.MIN_H)

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────
    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=self.SCENE_WEIGHT)
        self.root.columnconfigure(1, weight=self.PARAM_WEIGHT)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        editor_frame = ttk.Frame(self.root, padding=4)
        editor_frame.grid(row=0, column=0, sticky="nsew")
        self.scene_editor = SceneEditor(editor_frame)

        param_frame = ttk.Frame(self.root, padding=8, relief="groove", borderwidth=1)
        param_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)
        self.param_panel = ParamPanel(param_frame)

        self._build_bottom_bar()

    def _build_bottom_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 5))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(4, weight=1)

        self.run_btn = ttk.Button(bar, text="\u25b6 Plan + Run JED", width=16, command=self._on_run)
        self.run_btn.grid(row=0, column=0, padx=(0, 8))

        self.analyse_btn = ttk.Button(bar, text="\U0001F4CA Show Analysis", width=16,
                                       command=self._on_analyse, state="disabled")
        self.analyse_btn.grid(row=0, column=1, padx=(0, 8))

        self.mc_btn = ttk.Button(bar, text="\U0001F3B2 Run Monte Carlo Validation", width=26,
                                  command=self._on_monte_carlo)
        self.mc_btn.grid(row=0, column=2, padx=(0, 8))

        self.dash_btn = ttk.Button(bar, text="\u2699 Tuning Dashboard", width=18,
                                    command=self._on_open_dashboard)
        self.dash_btn.grid(row=0, column=3, padx=(0, 8))

        self._status_var = tk.StringVar(
            value="Ready. Place Start (S), Goal (G), optional obstacles, then Run.")
        ttk.Label(bar, textvariable=self._status_var, anchor="e", foreground="#444444").grid(
            row=0, column=4, sticky="ew")

    # ── Run (single rollout) ─────────────────────────────────────────
    def _on_run(self) -> None:
        if self._solver_thread and self._solver_thread.is_alive():
            return
        scene = self.scene_editor.get_scene()
        if scene["start"] is None:
            messagebox.showwarning("Missing start", "No start point set.\n\nLeft-click and press Set Start.")
            return
        if scene["goal"] is None:
            messagebox.showwarning("Missing goal", "No goal point set.\n\nLeft-click and press Set Goal.")
            return

        params = self.param_panel.get_params()
        params["world_bounds"] = _WORLD_BOUNDS
        self._last_goal = scene["goal"]

        self._set_controls_enabled(False)
        self._rollout = None
        self._status("Planning (A*) then running JED closed-loop \u2026")

        self._solver_thread = threading.Thread(
            target=self._solver_worker, args=(scene, params), daemon=True)
        self._solver_thread.start()

    def _solver_worker(self, scene: dict, params: dict) -> None:
        try:
            real_noise = GaussianProcessNoise(params["mu"], params["Sigma"])
            rollout = run_closed_loop(
                scene["start"][:2], scene["goal"], scene["obstacles"],
                self.vehicle, params, real_noise, n_steps=int(params.get("n_steps", cfg.N_STEPS_DEFAULT)))
            self.root.after(0, self._on_solver_done, rollout, None)
        except Exception as exc:   # noqa: BLE001
            import traceback; traceback.print_exc()
            self.root.after(0, self._on_solver_done, None, exc)

    def _on_solver_done(self, rollout: dict | None, error: Exception | None) -> None:
        self._set_controls_enabled(True)
        if error is not None:
            self._status("Solver error \u2014 see terminal for traceback.")
            messagebox.showerror("Solver error", f"{type(error).__name__}: {error}")
            return
        if rollout.get("planning_failed"):
            self._status("A* found no feasible path to the goal.")
            messagebox.showwarning("Planning failed", "A* could not find a path — try removing/loosening obstacles.")
            return

        self._rollout = rollout
        self.scene_editor.draw_result(rollout["waypoints"], rollout["x_hist"])

        n_slack = rollout["n_slack_active"]
        self._status(
            f"reached_goal={rollout['reached_goal']}  steps={len(rollout['u_hist'])}  "
            f"worst h={rollout['min_h']:.3f}  slack-active steps={n_slack}"
            + ("  \u26a0 Thm5 guarantee broken on some steps" if n_slack else "")
        )
        self.analyse_btn.config(state="normal")

    def _on_analyse(self) -> None:
        if self._rollout is None:
            return
        from ui.analysis import show_run_analysis
        try:
            show_run_analysis(self._rollout, self._last_goal)
        except Exception as exc:   # noqa: BLE001
            messagebox.showerror("Analysis error", f"{type(exc).__name__}: {exc}")

    # ── Monte Carlo validation ───────────────────────────────────────
    def _on_monte_carlo(self) -> None:
        scene = self.scene_editor.get_scene()
        if scene["start"] is None or scene["goal"] is None:
            messagebox.showwarning("Missing start/goal", "Set both Start and Goal before running validation.")
            return
        params = self.param_panel.get_params()
        params["world_bounds"] = _WORLD_BOUNDS

        self._set_controls_enabled(False)
        self._status(f"Running {params['n_trials']} Monte-Carlo trials \u2026")

        t = threading.Thread(target=self._mc_worker, args=(scene, params), daemon=True)
        t.start()

    def _mc_worker(self, scene: dict, params: dict) -> None:
        try:
            mc = run_monte_carlo(
                scene["start"][:2], scene["goal"], scene["obstacles"], self.vehicle, params,
                n_trials=int(params["n_trials"]), K_horizon=int(params["K_horizon"]),
                gammas=cfg.GAMMA_SWEEP)
            self.root.after(0, self._on_mc_done, mc, None)
        except Exception as exc:   # noqa: BLE001
            import traceback; traceback.print_exc()
            self.root.after(0, self._on_mc_done, None, exc)

    def _on_mc_done(self, mc: dict | None, error: Exception | None) -> None:
        self._set_controls_enabled(True)
        if error is not None:
            self._status("Monte Carlo error \u2014 see terminal.")
            messagebox.showerror("Monte Carlo error", f"{type(error).__name__}: {error}")
            return

        warn = ""
        if mc["frac_trials_with_slack"] > 0:
            warn = (f"  \u26a0 slack active on {mc['frac_trials_with_slack']*100:.1f}% of trials "
                    f"({mc['frac_steps_with_slack']*100:.1f}% of steps) \u2014 Thm 5 guarantee not strict there")
        self._status(
            f"MC done: {mc['n_trials']} trials, K={mc['K_horizon']}, "
            f"\u03b4={mc['delta']:.4f}, M={mc['M']:.2f}, h0={mc['h0']:.2f}{warn}")

        win = tk.Toplevel(self.root)
        win.title("Pu Bound vs Monte Carlo")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(mc["gammas"], mc["theoretical_pu"], "r--", marker="o", label="Pu bound (Thm 5)")
        ax.plot(mc["gammas"], mc["empirical_pu"], "b-", marker="s", label="MC estimate")
        ax.set_xlabel("\u03b3 (tolerable violation)")
        ax.set_ylabel("Pu")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Pu bound vs MC ({mc['n_trials']} trials, K={mc['K_horizon']})")
        ax.legend()
        ax.grid(alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    # ── Tuning dashboard ──────────────────────────────────────────────
    def _on_open_dashboard(self) -> None:
        R_sense = sense_radius(None)
        M = compute_M(R_sense, _WORLD_BOUNDS)
        open_dashboard(self.root, default_M=M, default_h0=0.5 * M)

    # ── Helpers ───────────────────────────────────────────────────────
    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.run_btn.config(state=state)
        self.mc_btn.config(state=state)
        if enabled and self._rollout is not None:
            self.analyse_btn.config(state="normal")
        elif not enabled:
            self.analyse_btn.config(state="disabled")

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.root.update_idletasks()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
