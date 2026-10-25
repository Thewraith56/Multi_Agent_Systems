"""
analysis/plots.py — Post-run diagnostic figures.

Five standalone functions, each creating one matplotlib figure:
    1. plot_trajectory   — 2-D map: reference r vs executed x, with obstacles
    2. plot_convergence  — primal residual ‖x − r‖² vs ADMM iteration k
    3. plot_controls     — linear speed v(t) and angular rate ω(t)
    4. plot_states       — x1(t), x2(t), θ(t) over time
    5. plot_cost_history — augmented Lagrangian value vs iteration k

show_all(result, scene) calls all five and displays them non-blocking,
so the main Tk window stays responsive.

Each individual function returns the matplotlib Figure object so you
can save it with fig.savefig(...) if needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── helpers ────────────────────────────────────────────────────────────────

def _draw_obstacles(ax: plt.Axes, obstacles: list) -> None:
    """Draw all obstacles onto ax (shared by trajectory and scene plots)."""
    for obs in obstacles:
        if obs["type"] == "rectangle":
            x_min, y_min, x_max, y_max = obs["params"]
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1.5, edgecolor="#1a6faf", facecolor="#aed6f1", alpha=0.7,
                zorder=2,
            )
            ax.add_patch(rect)
        elif obs["type"] == "circle":
            cx, cy, radius = obs["params"]
            circle = patches.Circle(
                (cx, cy), radius,
                linewidth=1.5, edgecolor="#1a6faf", facecolor="#aed6f1", alpha=0.7,
                zorder=2,
            )
            ax.add_patch(circle)


# ── Figure 1 ───────────────────────────────────────────────────────────────

def plot_trajectory(result: dict, scene: dict) -> plt.Figure:
    """
    2-D map showing the converged reference trajectory r and the
    executed state trajectory x, overlaid on the obstacle scene.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title("Figure 1 — Trajectory")

    _draw_obstacles(ax, scene.get("obstacles", []))

    r_final = result["r_hist"][-1]   # (N+1, 3)
    x_final = result["x_hist"][-1]   # (N+1, 3)

    ax.plot(r_final[:, 0], r_final[:, 1],
            "r--", linewidth=2, label="Reference  r  (planner)", zorder=3)
    ax.plot(x_final[:, 0], x_final[:, 1],
            "g-",  linewidth=2, label="Executed   x  (controller)", zorder=3)

    # Start marker
    start = scene.get("start")
    if start is not None:
        ax.plot(start[0], start[1], "go", markersize=12, zorder=5,
                label="Start", markeredgecolor="black")

    # Goal marker
    goal = scene.get("goal")
    if goal is not None:
        ax.plot(goal[0], goal[1], "r*", markersize=16, zorder=5,
                label="Goal", markeredgecolor="black")

    ax.set_title("Trajectory: Reference vs Executed", fontsize=13)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ── Figure 2 ───────────────────────────────────────────────────────────────

def plot_convergence(result: dict) -> plt.Figure:
    """
    Primal residual ‖x − r‖² on a log scale vs ADMM iteration k.
    Shows how quickly the planner and controller layers agree.
    """
    residuals = result["residuals"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("Figure 2 — Convergence")

    ax.semilogy(residuals, "b-o", markersize=5, linewidth=1.5, label="‖x − r‖²")
    ax.axhline(residuals[-1], color="red", linestyle="--", alpha=0.6,
               label=f"Final = {residuals[-1]:.2e}")

    ax.set_title("ADMM Convergence — Primal Residual", fontsize=13)
    ax.set_xlabel("ADMM iteration  k")
    ax.set_ylabel("‖x − r‖²  (log scale)")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ── Figure 3 ───────────────────────────────────────────────────────────────

def plot_controls(result: dict) -> plt.Figure:
    """
    Control inputs v(t) and ω(t) of the converged solution over time.
    """
    dt      = result["dt"]
    u_final = result["u_hist"][-1]    # (N, 2)
    N       = u_final.shape[0]
    t       = np.arange(N) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.canvas.manager.set_window_title("Figure 3 — Control Inputs")

    ax1.plot(t, u_final[:, 0], "b-o", markersize=4, linewidth=1.5)
    ax1.set_ylabel("v  (linear speed)")
    ax1.set_title("Control Inputs over Time", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, u_final[:, 1], "r-o", markersize=4, linewidth=1.5)
    ax2.set_ylabel("ω  (angular rate,  rad/s)")
    ax2.set_xlabel("Time  (s)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Figure 4 ───────────────────────────────────────────────────────────────

def plot_states(result: dict) -> plt.Figure:
    """
    State components x₁(t), x₂(t), θ(t) of the converged trajectory.
    """
    dt      = result["dt"]
    x_final = result["x_hist"][-1]    # (N+1, 3)
    N       = x_final.shape[0] - 1
    t       = np.arange(N + 1) * dt

    labels = ["x₁  (m)", "x₂  (m)", "θ  (rad)"]
    colors = ["royalblue", "seagreen", "darkorange"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig.canvas.manager.set_window_title("Figure 4 — State Evolution")

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, x_final[:, i], color=color, marker="o", markersize=3,
                linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("State Evolution over Time", fontsize=13)
    axes[-1].set_xlabel("Time  (s)")
    fig.tight_layout()
    return fig


# ── Figure 5 ───────────────────────────────────────────────────────────────

def plot_cost_history(result: dict) -> plt.Figure:
    """
    Augmented Lagrangian L_ρ value at each ADMM iteration.
    Shows the objective decreasing (or oscillating) as layers coordinate.
    """
    costs = result["costs"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("Figure 5 — Cost History")

    ax.plot(costs, "g-o", markersize=5, linewidth=1.5)
    ax.set_title("Augmented Lagrangian Value vs ADMM Iteration", fontsize=13)
    ax.set_xlabel("ADMM iteration  k")
    ax.set_ylabel("L_ρ(x, u, r, v)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── entry point ────────────────────────────────────────────────────────────

def show_all(result: dict, scene: dict) -> None:
    """
    Open all five analysis figures as separate non-blocking windows.
    Called from ui/app.py after a successful solver run.
    """
    plot_trajectory(result, scene)
    plot_convergence(result)
    plot_controls(result)
    plot_states(result)
    plot_cost_history(result)
    plt.show(block=False)
    plt.pause(0.1)   # let the event loop render all windows
