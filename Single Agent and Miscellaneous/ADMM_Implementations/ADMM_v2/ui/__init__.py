"""
ui — Tkinter-based graphical interface for ADMM Turtle.

Public surface
--------------
    from ui.app import App

    app = App()
    app.run()

Internal modules
----------------
    scene_editor  — matplotlib canvas embedded in Tk for interactive scene building
    param_panel   — right-hand panel with solver parameter controls
"""

from ui.app import App

__all__ = ["App"]
