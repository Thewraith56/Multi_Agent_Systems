"""
main.py — ADMM Turtle entry point.
 
Usage
-----
    python main.py
 
Launches the Tkinter UI.  Everything else (solver, analysis) is
imported lazily so the window opens even if a dependency is missing.
"""
 
import sys
import os
 
# Put the project root on sys.path so every sub-package resolves
# correctly regardless of the working directory.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
 
 
def _check_python() -> None:
    if sys.version_info < (3, 9):
        sys.exit(
            f"Python 3.9+ required — found {sys.version_info.major}.{sys.version_info.minor}"
        )
 
 
def _check_imports() -> None:
    """Give a clear error if a hard dependency is missing."""
    missing = []
    for pkg in ("numpy", "scipy", "matplotlib"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
 
    if missing:
        sys.exit(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nRun:  pip install -r requirements.txt"
        )
 
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.exit(
            "Tkinter not found.\n"
            "Linux:   sudo apt install python3-tk\n"
            "macOS:   install Python from python.org (includes Tk)\n"
            "Windows: re-run the Python installer and tick 'tcl/tk'"
        )
 
 
def main() -> None:
    _check_python()
    _check_imports()
 
    from ui.app import App
 
    app = App()
    app.run()
 
 
if __name__ == "__main__":
    main()
 