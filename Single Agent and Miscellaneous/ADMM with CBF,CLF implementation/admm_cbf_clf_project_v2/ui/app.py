"""
ui/app.py — Root Tkinter window.

Changes from original app.py
-----------------------------
  ADDED  — vehicle selector combobox in the bottom bar
            (Unicycle and Drone; Bicycle slot is ready)
  CHANGED — _solver_worker now instantiates the chosen vehicle and passes
             it + architecture to solver.admm.solver()
  CHANGED — status bar includes vehicle name and architecture after each run
  KEPT   — all layout, threading, error-handling, and analysis logic

Window layout
-------------
┌──────────────────────────────┬──────────────────┐
│                              │                  │
│         SceneEditor          │   ParamPanel     │
│     (matplotlib canvas)      │  (N, rho, …)    │
│                              │  + arch combo    │
├──────────────────────────────┴──────────────────┤
│ [Vehicle ▾] [▶ Run] [📊 Analysis]   status …  │
└─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from ui.scene_editor import SceneEditor
except ImportError as _err:
    raise ImportError(
        "ui/scene_editor.py not found.\n"
        f"Original error: {_err}"
    ) from _err

try:
    from ui.param_panel import ParamPanel
except ImportError as _err:
    raise ImportError(
        "ui/param_panel.py not found.\n"
        f"Original error: {_err}"
    ) from _err


# ── Vehicle registry ──────────────────────────────────────────────────────
# Add Drone and Bicycle here once their vehicle files exist.
# Each entry: display_name -> dotted import path of the vehicle class.
_VEHICLE_REGISTRY: dict[str, str] = {
    "Unicycle" : "solver.vehicles.unicycle.UnicycleVehicle",
    "Drone"    : "solver.vehicles.drone.DroneVehicle",
    # "Bicycle"  : "solver.vehicles.bicycle.BicycleVehicle",  # enable later
}


def _import_vehicle(class_path: str):
    """
    Lazily import and instantiate a vehicle class by dotted path.

    Example
    -------
    _import_vehicle("solver.vehicles.unicycle.UnicycleVehicle")
    → returns UnicycleVehicle()
    """
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)
    return cls()


# ── App ───────────────────────────────────────────────────────────────────

class App:
    """Root application window."""

    TITLE       = "ADMM Path Planner — CBF/CLF Multi-Vehicle"
    MIN_W       = 1150
    MIN_H       = 700
    SCENE_WEIGHT = 3
    PARAM_WEIGHT = 1

    def __init__(self) -> None:
        self._solver_result : dict | None            = None
        self._solver_thread : threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title(self.TITLE)
        self.root.minsize(self.MIN_W, self.MIN_H)

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=self.SCENE_WEIGHT)
        self.root.columnconfigure(1, weight=self.PARAM_WEIGHT)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # scene editor (left)
        editor_frame = ttk.Frame(self.root, padding=4)
        editor_frame.grid(row=0, column=0, sticky="nsew")
        self.scene_editor = SceneEditor(editor_frame)

        # param panel (right)
        param_frame = ttk.Frame(
            self.root, padding=8, relief="groove", borderwidth=1,
        )
        param_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)
        self.param_panel = ParamPanel(param_frame)

        self._build_bottom_bar()

    def _build_bottom_bar(self) -> None:
        """
        Bottom bar layout
        -----------------
        col 0 : vehicle label + combobox
        col 1 : Run button
        col 2 : Analysis button
        col 3 : (spacer, expands)
        col 4 : status label
        """
        bar = ttk.Frame(self.root, padding=(8, 5))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(3, weight=1)

        # ── vehicle selector ──────────────────────────────────────────
        ttk.Label(bar, text="Vehicle:").grid(row=0, column=0, padx=(0, 4))

        self._vehicle_var = tk.StringVar(
            value=next(iter(_VEHICLE_REGISTRY))   # first entry = Unicycle
        )
        vehicle_combo = ttk.Combobox(
            bar,
            textvariable = self._vehicle_var,
            values       = list(_VEHICLE_REGISTRY.keys()),
            state        = "readonly",
            width        = 10,
        )
        vehicle_combo.grid(row=0, column=1, padx=(0, 16))

        # ── Run button ────────────────────────────────────────────────
        self.run_btn = ttk.Button(
            bar,
            text    = "▶  Run Solver",
            width   = 16,
            command = self._on_run,
        )
        self.run_btn.grid(row=0, column=2, padx=(0, 8))

        # ── Analysis button ───────────────────────────────────────────
        self.analyse_btn = ttk.Button(
            bar,
            text    = "📊  Show Analysis",
            width   = 18,
            command = self._on_analyse,
            state   = "disabled",
        )
        self.analyse_btn.grid(row=0, column=3, padx=(0, 8))

        # ── Status label ──────────────────────────────────────────────
        self._status_var = tk.StringVar(
            value="Ready.  Place Start (S), Goal (G), optional obstacles, then Run."
        )
        ttk.Label(
            bar,
            textvariable = self._status_var,
            anchor       = "e",
            foreground   = "#444444",
        ).grid(row=0, column=4, sticky="ew")

    # ── Button callbacks ──────────────────────────────────────────────────

    def _on_run(self) -> None:
        """Validate scene, then launch solver in a background thread."""
        if self._solver_thread and self._solver_thread.is_alive():
            return

        scene = self.scene_editor.get_scene()

        if scene["start"] is None:
            messagebox.showwarning(
                "Missing start",
                "No start point set.\n\nLeft-click the canvas and press Set Start.",
            )
            return

        if scene["goal"] is None:
            messagebox.showwarning(
                "Missing goal",
                "No goal point set.\n\nLeft-click the canvas and press Set Goal.",
            )
            return

        params       = self.param_panel.get_params()
        vehicle_name = self._vehicle_var.get()
        arch         = params.get("architecture", "i")

        self._set_controls_enabled(False)
        self._solver_result = None
        self._status(
            f"Running  {vehicle_name}  |  arch {arch}  …  "
            "(this may take several seconds)"
        )

        self._solver_thread = threading.Thread(
            target = self._solver_worker,
            args   = (scene, params, vehicle_name),
            daemon = True,
        )
        self._solver_thread.start()

    def _solver_worker(
        self,
        scene        : dict,
        params       : dict,
        vehicle_name : str,
    ) -> None:
        """
        Background thread: instantiate vehicle, run solver, post result.

        Lazy imports mean the UI opens even if solver files have errors.
        """
        try:
            # ── instantiate vehicle ───────────────────────────────────
            class_path = _VEHICLE_REGISTRY[vehicle_name]
            vehicle    = _import_vehicle(class_path)

            # ── convert scene start to vehicle state ──────────────────
            # scene["start"] is always shape (5,): [x, y, theta, v, omega]
            x0 = vehicle.initial_state(scene["start"])

            # ── run solver ────────────────────────────────────────────
            from solver.admm import solver

            result = solver(
                x0           = x0,
                goal         = scene["goal"],
                obstacles    = scene["obstacles"],
                params       = params,
                vehicle      = vehicle,
                architecture = params.get("architecture", "i"),
            )

            self.root.after(0, self._on_solver_done, result, None)

        except Exception as exc:   # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.root.after(0, self._on_solver_done, None, exc)

    def _on_solver_done(
        self,
        result : dict | None,
        error  : Exception | None,
    ) -> None:
        """Called on the main Tk thread once the background worker finishes."""
        self._set_controls_enabled(True)

        if error is not None:
            self._status("Solver error — see terminal for traceback.")
            messagebox.showerror(
                "Solver error",
                f"{type(error).__name__}: {error}\n\n"
                "Check the terminal for the full traceback.",
            )
            return

        self._solver_result = result

        # Draw the converged trajectories.
        # r_hist[-1] and x_hist[-1] are both shape (N+1, n).
        # scene_editor.draw_result uses only columns 0 and 1 (x, y).
        self.scene_editor.draw_result(
            r_final = result["r_hist"][-1],
            x_final = result["x_hist"][-1],
        )

        n_iter    = len(result["residuals"])
        final_res = result["residuals"][-1]
        vname     = result.get("vehicle_name", "?")
        arch      = result.get("architecture",  "?")

        self._status(
            f"{vname}  |  arch {arch}  —  "
            f"{n_iter} ADMM iterations,  "
            f"final residual ‖x−r‖² = {final_res:.2e}"
        )
        self.analyse_btn.config(state="normal")

    def _on_analyse(self) -> None:
        """Open analysis figures for the most recent solver result."""
        if self._solver_result is None:
            return
        try:
            from analysis.plots import show_all   # lazy import
            show_all(
                result = self._solver_result,
                scene  = self.scene_editor.get_scene(),
            )
        except Exception as exc:    # noqa: BLE001
            messagebox.showerror(
                "Analysis error",
                f"{type(exc).__name__}: {exc}",
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.run_btn.config(state=state)
        if enabled and self._solver_result is not None:
            self.analyse_btn.config(state="normal")
        elif not enabled:
            self.analyse_btn.config(state="disabled")

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.root.update_idletasks()

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()
