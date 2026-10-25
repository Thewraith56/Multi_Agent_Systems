"""
ui/analysis.py — Pop-up analysis plots for one closed-loop JED run.
"""

from __future__ import annotations

import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def show_run_analysis(rollout: dict, goal_xy: np.ndarray) -> None:
    win = tk.Toplevel()
    win.title("Run Analysis")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    h_hist = rollout["h_hist"]
    u_hist = np.array(rollout["u_hist"]) if rollout["u_hist"] else np.zeros((0, 2))
    x_hist = rollout["x_hist"]
    slack_hist = rollout["slack_hist"]

    ax = axes[0, 0]
    ax.plot(h_hist, color="#1a6faf")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("h(x_k) — DTCBF barrier value")
    ax.set_xlabel("step"); ax.set_ylabel("h")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if u_hist.shape[0] > 0:
        ax.plot(u_hist[:, 0], label="ax")
        ax.plot(u_hist[:, 1], label="ay")
        ax.legend(fontsize=8)
    ax.set_title("Control effort")
    ax.set_xlabel("step"); ax.set_ylabel("accel (m/s^2)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    dist = np.linalg.norm(x_hist[:, :2] - goal_xy[None, :], axis=1)
    ax.plot(dist, color="#2ca02c")
    ax.set_title("Distance to goal")
    ax.set_xlabel("step"); ax.set_ylabel("m")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(slack_hist, color="#d62728")
    ax.set_title("JED feasibility slack (>0 breaks Thm 5 guarantee)")
    ax.set_xlabel("step"); ax.set_ylabel("slack")
    ax.grid(alpha=0.3)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()
