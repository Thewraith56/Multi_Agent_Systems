"""
main.py — Entry point for the ADMM CBF/CLF Path Planner.

Run from the project root:
    python main.py

Project layout expected
-----------------------
project_root/
├── main.py                          ← this file
├── solver/
│   ├── __init__.py
│   ├── admm.py
│   ├── cbf.py                       ← HOCBF constraints
│   ├── clf.py
│   ├── costs.py
│   ├── constraints.py
│   ├── planner.py                   ← A* warm-start (not wired in yet)
│   └── vehicles/
│       ├── __init__.py
│       ├── base.py
│       └── unicycle.py
└── ui/
    ├── app.py
    ├── param_panel.py
    └── scene_editor.py

How to use the app
------------------
1. Left-click on the canvas to place a cursor (black crosshair).
2. Click  📍 Set Start  — marks the green circle at cursor position.
   θ₀ defaults to "auto" (faces the goal if goal already exists).
3. Left-click elsewhere, then click  🎯 Set Goal — marks the red star.
4. Draw obstacles (optional):
     Left-click + drag  → rectangle obstacle
     Right-click        → circle obstacle (dialog asks for radius)
   Undo / Clear buttons manage the obstacle list.
5. In the right panel, choose:
     Architecture:  i  = CBF(soft) + terminal scale, no CLF
                    ii = CBF(hard) + CLF(soft slack), no terminal
   Tune N, ρ, dt, γ_cbf, λ_clf, p (arch i), p_clf (arch ii),
   terminal_scale as needed.
6. Click  ▶ Run Solver.  The solver runs in a background thread;
   the UI stays responsive.  Progress is printed to this terminal.
7. After completion, the reference path (red dashed) and executed
   trajectory (green) are drawn on the canvas.
8. Click  📊 Show Analysis  for convergence / control plots.
"""

import sys
import os

# ── Make project root importable regardless of working directory ──────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Dependency check — give a clear error if core packages are missing ────
_REQUIRED = {
    "numpy"   : "numpy",
    "scipy"   : "scipy",
    "matplotlib": "matplotlib",
}

_missing = []
for _pkg, _import in _REQUIRED.items():
    try:
        __import__(_import)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(
        "\n[main.py] Missing required packages:\n"
        + "\n".join(f"    pip install {p}" for p in _missing)
        + "\nInstall them and re-run.\n"
    )
    sys.exit(1)

# ── Launch ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ui.app import App
    App().run()
