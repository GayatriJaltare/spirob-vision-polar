"""Run mjlab's `train` with our polar tasks registered (sir's tasks are registered too).

Usage (from sir's repo folder):  uv run python <this repo>/tools/train_polar.py <TASK> [train options]
"""
import os
import sys
from importlib.metadata import entry_points

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polar_tasks  # noqa: F401,E402

eps = [e for e in entry_points(group="console_scripts") if e.name == "train"]
if not eps:
    sys.exit("could not find mjlab's `train` entry point")
sys.argv[0] = "train"
eps[0].load()()
