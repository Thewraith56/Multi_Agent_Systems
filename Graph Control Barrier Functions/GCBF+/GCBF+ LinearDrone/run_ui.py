"""Launch the simulator UI:   python run_ui.py"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from app.main import main

if __name__ == "__main__":
    main()
