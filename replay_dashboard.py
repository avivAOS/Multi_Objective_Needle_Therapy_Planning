"""
Reopen an interactive Pareto dashboard window from a saved frontier snapshot
(`frontier.pkl` / `frontier_3d.pkl`, written by `main_demo.py`/`main_demo_3d.py`
after a DP run), without recomputing the paths graph or re-running DP.

Usage:
    python replay_dashboard.py outputs/dp/2d/<run_name>/frontier.pkl
    python replay_dashboard.py outputs/dp/3d/<run_name>/frontier_3d.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontier_path", type=Path, help="Path to a saved frontier .pkl file.")
    args = parser.parse_args(argv)

    with open(args.frontier_path, "rb") as f:
        bundle = pickle.load(f)

    kind = bundle["kind"]
    if kind == "2d":
        from main_demo import generate_pareto_dashboard
        fig, ani = generate_pareto_dashboard(
            bundle["plans"], bundle["start_key"], bundle["nodes"], bundle["world"],
            seed=bundle["seed"], step_size=bundle["step_size"],
        )
    elif kind == "3d":
        from main_demo_3d import generate_pareto_dashboard_3d
        fig, ani = generate_pareto_dashboard_3d(
            bundle["plans"], bundle["start_key"], bundle["nodes"], bundle["world"],
            seed=bundle["seed"], step_size=bundle["step_size"],
        )
    else:
        raise ValueError(f"Unknown frontier kind in {args.frontier_path}: {kind!r}")

    print(f"Loaded {len(bundle['plans'])} plans from {args.frontier_path}.", flush=True)
    print("Showing dashboard window (close it to finish).", flush=True)
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
