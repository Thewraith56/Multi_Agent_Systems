"""
ui/scene_editor.py — Interactive scene builder (matplotlib canvas in Tk).

Interactions
------------
Left-click          : place cursor (pending position)
Left-click + drag   : draw a rectangle obstacle
Right-click         : draw a circle obstacle (dialog asks for radius)
[Set Start] button  : mark cursor as start  (theta set via spinbox)
[Set Goal]  button  : mark cursor as goal
[Undo]      button  : remove the last obstacle
[Clear]     button  : reset the entire scene

Public interface (called by app.py)
------------------------------------
    SceneEditor(parent)
    .get_scene()   -> dict {"start", "goal", "obstacles"}
    .draw_result(r_final, x_final)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog as simpledialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")                          # must be before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# World coordinate bounds — obstacles, start, and goal must live in here
_WORLD_X = (0.0, 10.0)
_WORLD_Y = (0.0, 10.0)

# Minimum drag distance (world units) to register a rectangle
_DRAG_THRESHOLD = 0.2


class SceneEditor:
    """Embeds a matplotlib Figure into a Tk Frame for interactive scene editing."""

    def __init__(self, parent: ttk.Frame) -> None:
        self._start: np.ndarray | None = None    # shape (3,): [x, y, theta]
        self._goal:  np.ndarray | None = None    # shape (2,): [x, y]
        self._obstacles: list[dict]    = []

        self._cursor_pos: np.ndarray | None = None   # pending click position
        self._drag_start: np.ndarray | None = None   # rectangle drag origin
        self._result_drawn = False

        self._build(parent)

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def _build(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=0)

        # ── matplotlib canvas ─────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.0))
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.08)

        self._canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("motion_notify_event",  self._on_motion)

        # ── instruction label ─────────────────────────────────────────
        ttk.Label(
            parent,
            text="Left-click: place cursor  │  Drag: rectangle  │  Right-click: circle",
            foreground="#555555",
            font=("", 9),
        ).grid(row=1, column=0, pady=(2, 2))

        # ── button bar ────────────────────────────────────────────────
        btn_bar = ttk.Frame(parent)
        btn_bar.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        ttk.Button(btn_bar, text="📍 Set Start",
                   command=self._set_start).pack(side="left", padx=4)
        ttk.Button(btn_bar, text="🎯 Set Goal",
                   command=self._set_goal).pack(side="left", padx=4)

        ttk.Separator(btn_bar, orient="vertical").pack(
            side="left", fill="y", padx=6)

        ttk.Label(btn_bar, text="θ₀ (rad):").pack(side="left")
        self._theta_var = tk.StringVar(value="auto")
        ttk.Spinbox(
            btn_bar, from_=-3.14, to=3.14, increment=0.1,
            textvariable=self._theta_var, width=5,
        ).pack(side="left", padx=(2, 8))

        ttk.Separator(btn_bar, orient="vertical").pack(
            side="left", fill="y", padx=6)

        ttk.Button(btn_bar, text="↩ Undo",  command=self._undo ).pack(side="left", padx=4)
        ttk.Button(btn_bar, text="🗑 Clear", command=self._clear).pack(side="left", padx=4)

        self._redraw()

    # ------------------------------------------------------------------ #
    # Mouse events                                                         #
    # ------------------------------------------------------------------ #

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax:
            return

        if event.button == 1:                             # left click
            pos = np.array([event.xdata, event.ydata])
            self._cursor_pos = pos
            self._drag_start = pos.copy()

        elif event.button == 3:                           # right click → circle
            self._add_circle(event.xdata, event.ydata)

        self._redraw()

    def _on_motion(self, event) -> None:
        """Preview rectangle while dragging."""
        if (event.button != 1
                or self._drag_start is None
                or event.inaxes != self.ax
                or event.xdata is None):
            return
        self._cursor_pos = np.array([event.xdata, event.ydata])
        self._redraw()

    def _on_release(self, event) -> None:
        if event.button != 1 or self._drag_start is None:
            self._drag_start = None
            return

        if event.inaxes == self.ax and event.xdata is not None:
            end  = np.array([event.xdata, event.ydata])
            dist = np.linalg.norm(end - self._drag_start)
            if dist >= _DRAG_THRESHOLD:
                x_min = min(self._drag_start[0], end[0])
                x_max = max(self._drag_start[0], end[0])
                y_min = min(self._drag_start[1], end[1])
                y_max = max(self._drag_start[1], end[1])
                self._obstacles.append({
                    "type":   "rectangle",
                    "params": (x_min, y_min, x_max, y_max),
                })

        self._drag_start = None
        self._redraw()

    # ------------------------------------------------------------------ #
    # Button callbacks                                                     #
    # ------------------------------------------------------------------ #

    def _set_start(self) -> None:
        if self._cursor_pos is None:
            return
            
        val = self._theta_var.get().strip().lower()
        if val == "auto":
            # If goal exists, calculate the straight-line angle. Otherwise default to 0.0
            if self._goal is not None:
                dy = self._goal[1] - self._cursor_pos[1]
                dx = self._goal[0] - self._cursor_pos[0]
                theta = float(np.arctan2(dy, dx))
            else:
                theta = 0.0
        else:
            try:
                theta = float(val)
            except ValueError:
                theta = 0.0

        # Spawn at X, Y, Theta, with 0 velocity and 0 angular velocity
        self._start = np.array([self._cursor_pos[0], self._cursor_pos[1], theta, 0.0, 0.0])
        self._redraw()

    def _set_goal(self) -> None:
        if self._cursor_pos is None:
            return
            
        self._goal = np.array([self._cursor_pos[0], self._cursor_pos[1]])
        
        # If start exists and UI is on "auto", rotate start to face the new goal
        if self._start is not None and self._theta_var.get().strip().lower() == "auto":
            dy = self._goal[1] - self._start[1]
            dx = self._goal[0] - self._start[0]
            self._start[2] = float(np.arctan2(dy, dx))
            
        self._redraw()

    def _undo(self) -> None:
        if self._obstacles:
            self._obstacles.pop()
        self._redraw()

    def _clear(self) -> None:
        self._start       = None
        self._goal        = None
        self._obstacles   = []
        self._cursor_pos  = None
        self._drag_start  = None
        self._result_drawn = False
        self._theta_var.set("auto")  # Reset the UI box back to auto
        self._redraw()

    def _add_circle(self, cx: float, cy: float) -> None:
        radius = simpledialog.askfloat(
            "Circle Obstacle",
            "Enter radius (world units):",
            initialvalue=0.5,
            minvalue=0.05,
        )
        if radius:
            self._obstacles.append({
                "type":   "circle",
                "params": (cx, cy, float(radius)),
            })

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _redraw(self) -> None:
        """Repaint the entire scene from scratch."""
        self.ax.clear()

        self.ax.set_xlim(*_WORLD_X)
        self.ax.set_ylim(*_WORLD_Y)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_title("Scene Editor", fontsize=11)
        self.ax.set_xlabel("x₁"); self.ax.set_ylabel("x₂")

        # Obstacles
        for obs in self._obstacles:
            if obs["type"] == "rectangle":
                x_min, y_min, x_max, y_max = obs["params"]
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=1.5, edgecolor="#1a6faf",
                    facecolor="#aed6f1", alpha=0.7, zorder=2,
                )
                self.ax.add_patch(rect)
            elif obs["type"] == "circle":
                cx, cy, r = obs["params"]
                circ = patches.Circle(
                    (cx, cy), r,
                    linewidth=1.5, edgecolor="#1a6faf",
                    facecolor="#aed6f1", alpha=0.7, zorder=2,
                )
                self.ax.add_patch(circ)

        # Rectangle drag preview
        if self._drag_start is not None and self._cursor_pos is not None:
            ds = self._drag_start
            cp = self._cursor_pos
            x_min, x_max = min(ds[0], cp[0]), max(ds[0], cp[0])
            y_min, y_max = min(ds[1], cp[1]), max(ds[1], cp[1])
            prev = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1, edgecolor="gray",
                facecolor="lightgray", alpha=0.4, linestyle="--", zorder=2,
            )
            self.ax.add_patch(prev)

        # Cursor crosshair
        if self._cursor_pos is not None and self._drag_start is None:
            self.ax.plot(
                *self._cursor_pos, "k+",
                markersize=14, markeredgewidth=2, zorder=4,
            )

        # Start marker
        if self._start is not None:
            self.ax.plot(
                self._start[0], self._start[1], "go",
                markersize=12, zorder=5,
                markeredgecolor="black", label="Start",
            )
            self.ax.annotate(
                f"Start (θ={self._start[2]:.2f})",
                self._start[:2],
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )

        # Goal marker
        if self._goal is not None:
            self.ax.plot(
                self._goal[0], self._goal[1], "r*",
                markersize=16, zorder=5,
                markeredgecolor="black", label="Goal",
            )
            self.ax.annotate(
                "Goal", self._goal,
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )

        if self._start is not None or self._goal is not None:
            self.ax.legend(loc="upper right", fontsize=8)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def draw_result(self, r_final: np.ndarray, x_final: np.ndarray) -> None:
        """
        Overlay the converged trajectories on the scene.
        Called by app.py after solver() completes.

        Parameters
        ----------
        r_final : shape (N+1, 3)  reference trajectory  [x1, x2, theta]
        x_final : shape (N+1, 3)  executed trajectory   [x1, x2, theta]
        """
        self.ax.plot(
            r_final[:, 0], r_final[:, 1],
            "r--", linewidth=2, label="Reference r", zorder=3,
        )
        self.ax.plot(
            x_final[:, 0], x_final[:, 1],
            color="limegreen", linewidth=2, label="Executed x", zorder=3,
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self._canvas.draw_idle()
        self._result_drawn = True

    def get_scene(self) -> dict:
        """
        Return the current scene state.

        Returns
        -------
        dict:
            "start"     : np.ndarray (3,) or None
            "goal"      : np.ndarray (2,) or None
            "obstacles" : list of dicts (copy)
        """
        return {
            "start":     self._start.copy() if self._start is not None else None,
            "goal":      self._goal.copy()  if self._goal  is not None else None,
            "obstacles": list(self._obstacles),
        }
