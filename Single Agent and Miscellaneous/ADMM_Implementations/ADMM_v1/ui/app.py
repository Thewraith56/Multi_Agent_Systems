"""
ui/app.py — Root Tkinter window.

Responsibilities
----------------
* Build and own the top-level Tk window.
* Lay out SceneEditor (left) and ParamPanel (right).
* Provide a "Run Solver" button that:
    1. Validates the scene (start + goal must be set).
    2. Runs solver.admm.solver() in a background thread so the UI
       stays responsive.
    3. Passes the result back to the main thread and draws trajectories.
* Provide a "Show Analysis" button that calls analysis.plots.show_all().

Interfaces expected from sub-modules
-------------------------------------
SceneEditor  (ui/scene_editor.py)
    .get_scene()  -> dict with keys:
        "start"     : np.ndarray shape (3,)  — (x, y, theta)  or None
        "goal"      : np.ndarray shape (2,)  — (x, y)         or None
        "obstacles" : list of dicts, each with keys:
            "type"   : "rectangle" | "circle"
            "params" : tuple  — (x_min,y_min,x_max,y_max) or (cx,cy,r)
    .draw_result(r_final, x_final)
        r_final : np.ndarray (N+1, 2)  — reference (x,y) positions
        x_final : np.ndarray (N+1, 3)  — state (x,y,theta) trajectory

ParamPanel  (ui/param_panel.py)
    .get_params()  -> dict with keys:
        "N"        : int   — planning horizon (number of steps)
        "rho"      : float — ADMM penalty parameter
        "dt"       : float — Euler time step (seconds)
        "tol"      : float — primal residual stopping threshold
        "max_iter" : int   — hard iteration cap

solver.admm.solver(x0, goal, obstacles, params)  -> result dict
    result keys: r_hist, x_hist, u_hist, v_hist, residuals, costs
    (see README for full schema)

analysis.plots.show_all(result, scene)
    Opens all five analysis figures.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

# ---------------------------------------------------------------------------
# SceneEditor and ParamPanel are imported here.
# If either file does not exist yet, the import error surfaces immediately
# with a clear message rather than crashing silently later.
# ---------------------------------------------------------------------------
try:
    from ui.scene_editor import SceneEditor
except ImportError as _err:
    raise ImportError(
        "ui/scene_editor.py not found — create it before running the app.\n"
        f"Original error: {_err}"
    ) from _err

try:
    from ui.param_panel import ParamPanel
except ImportError as _err:
    raise ImportError(
        "ui/param_panel.py not found — create it before running the app.\n"
        f"Original error: {_err}"
    ) from _err


class App:
    """
    Root application window.

    Window layout
    -------------
    ┌──────────────────────────────┬──────────────────┐
    │                              │                  │
    │         SceneEditor          │   ParamPanel     │
    │     (matplotlib canvas)      │  (N, rho, dt…)  │
    │                              │                  │
    ├──────────────────────────────┴──────────────────┤
    │ [▶ Run Solver]  [📊 Show Analysis]   status … │
    └──────────────────────────────────────────────────┘
    """

    # Window constants ---------------------------------------------------
    TITLE = "ADMM Turtle — Layered Control Path Planner"
    MIN_W = 1100
    MIN_H = 680

    # Relative column widths (scene editor gets ~75 % of horizontal space)
    SCENE_WEIGHT = 3
    PARAM_WEIGHT = 1

    def __init__(self) -> None:
        self._solver_result: dict | None = None
        self._solver_thread: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title(self.TITLE)
        self.root.minsize(self.MIN_W, self.MIN_H)

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Create all frames and widgets."""

        # ── grid weights ──────────────────────────────────────────────
        self.root.columnconfigure(0, weight=self.SCENE_WEIGHT)
        self.root.columnconfigure(1, weight=self.PARAM_WEIGHT)
        self.root.rowconfigure(0, weight=1)   # main content row expands
        self.root.rowconfigure(1, weight=0)   # bottom bar stays fixed

        # ── scene editor (left) ───────────────────────────────────────
        editor_frame = ttk.Frame(self.root, padding=4)
        editor_frame.grid(row=0, column=0, sticky="nsew")
        # SceneEditor is responsible for packing itself inside editor_frame.
        self.scene_editor = SceneEditor(editor_frame)

        # ── parameter panel (right) ───────────────────────────────────
        param_frame = ttk.Frame(self.root, padding=8, relief="groove", borderwidth=1)
        param_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)
        # ParamPanel packs itself inside param_frame.
        self.param_panel = ParamPanel(param_frame)

        # ── bottom bar ────────────────────────────────────────────────
        self._build_bottom_bar()

    def _build_bottom_bar(self) -> None:
        """Run button + analysis button + status label."""
        bar = ttk.Frame(self.root, padding=(8, 5))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Column 0: Run button
        # Column 1: Analysis button
        # Column 2: spacer
        # Column 3: status label (right-aligned)
        bar.columnconfigure(2, weight=1)

        # ── Run button ────────────────────────────────────────────────
        self.run_btn = ttk.Button(
            bar,
            text="▶  Run Solver",
            width=16,
            command=self._on_run,
        )
        self.run_btn.grid(row=0, column=0, padx=(0, 8))

        # ── Analysis button (disabled until a result exists) ──────────
        self.analyse_btn = ttk.Button(
            bar,
            text="📊  Show Analysis",
            width=18,
            command=self._on_analyse,
            state="disabled",
        )
        self.analyse_btn.grid(row=0, column=1, padx=(0, 8))

        # ── Separator ─────────────────────────────────────────────────
        ttk.Separator(bar, orient="vertical").grid(
            row=0, column=2, sticky="ns", padx=8
        )

        # ── Status label ──────────────────────────────────────────────
        self._status_var = tk.StringVar(
            value="Ready.  Place a start (S), a goal (G), optional obstacles, then Run."
        )
        status_lbl = ttk.Label(
            bar,
            textvariable=self._status_var,
            anchor="e",
            foreground="#444444",
        )
        status_lbl.grid(row=0, column=3, sticky="ew")

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        """
        Validate scene inputs, then kick off the solver in a background
        thread to keep the UI responsive.
        """
        # Prevent double-clicks while a run is in progress.
        if self._solver_thread and self._solver_thread.is_alive():
            return

        scene = self.scene_editor.get_scene()

        if scene["start"] is None:
            messagebox.showwarning(
                "Missing start",
                "No start point set.\n\nLeft-click the canvas and press S.",
            )
            return

        if scene["goal"] is None:
            messagebox.showwarning(
                "Missing goal",
                "No goal point set.\n\nLeft-click the canvas and press G.",
            )
            return

        params = self.param_panel.get_params()

        # Disable controls for the duration of the run.
        self._set_controls_enabled(False)
        self._solver_result = None
        self._status("Running ADMM solver …  (this may take a few seconds)")

        self._solver_thread = threading.Thread(
            target=self._solver_worker,
            args=(scene, params),
            daemon=True,  # thread dies with the process
        )
        self._solver_thread.start()

    def _solver_worker(self, scene: dict, params: dict) -> None:
        """
        Run in a background thread.
        Imports solver lazily so UI can open even if solver has a syntax error.
        Schedules _on_solver_done on the main thread when finished.
        """
        try:
            from solver.admm import solver  # lazy import

            result = solver(
                x0=scene["start"],
                goal=scene["goal"],
                obstacles=scene["obstacles"],
                params=params,
            )
            # Schedule the UI update back on the main thread.
            self.root.after(0, self._on_solver_done, result, None)

        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._on_solver_done, None, exc)

    def _on_solver_done(self, result: dict | None, error: Exception | None) -> None:
        """
        Called on the main Tk thread once the solver worker finishes.
        Updates UI and stores result for the analysis step.
        """
        self._set_controls_enabled(True)

        if error is not None:
            self._status(f"Solver error — see console for traceback.")
            messagebox.showerror(
                "Solver error",
                f"{type(error).__name__}: {error}\n\n"
                "Check the terminal for the full traceback.",
            )
            return

        self._solver_result = result

        # Draw the converged trajectories on the scene canvas.
        # r_hist[-1] is the final reference path, x_hist[-1] is the executed one.
        self.scene_editor.draw_result(
            r_final=result["r_hist"][-1],   # shape (N+1, 2)
            x_final=result["x_hist"][-1],   # shape (N+1, 3)
        )

        n_iter = len(result["residuals"])
        final_res = result["residuals"][-1]
        self._status(
            f"Done — {n_iter} ADMM iterations,  "
            f"final residual ‖x−r‖² = {final_res:.2e}"
        )
        self.analyse_btn.config(state="normal")

    def _on_analyse(self) -> None:
        """Open all five analysis figures."""
        if self._solver_result is None:
            # Should not happen because the button is disabled until a result
            # exists, but guard anyway.
            return

        try:
            from analysis.plots import show_all  # lazy import

            show_all(
                result=self._solver_result,
                scene=self.scene_editor.get_scene(),
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Analysis error",
                f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable interactive controls during a solver run."""
        state = "normal" if enabled else "disabled"
        self.run_btn.config(state=state)
        # Analysis button is only enabled when a result exists.
        if enabled and self._solver_result is not None:
            self.analyse_btn.config(state="normal")
        elif not enabled:
            self.analyse_btn.config(state="disabled")

    def _status(self, msg: str) -> None:
        """Update the status bar text and force a redraw."""
        self._status_var.set(msg)
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Hand control to the Tk event loop."""
        self.root.mainloop()
