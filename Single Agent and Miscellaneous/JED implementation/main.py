"""
main.py — Entry point. Run with:  python main.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import App

if __name__ == "__main__":
    App().run()
