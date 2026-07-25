"""
Interactive 3D viewer for the steerable-needle plan.

Opens a rotatable 3D scene (drag with the mouse to orbit) showing the wall
obstacles, ball tumors, and the IRIS best path. A slider scrubs the needle
along the path; tumors turn green as injections heal them up to that step.

By default it runs the planner fresh. Pass --path to load a previously saved
outputs/iris/3d/<run_name>/path.json instead (faster, deterministic).

Examples:
    python view_3d.py
    python view_3d.py --path outputs/iris/3d/20260716-120000_seed0_steps11/path.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sim3d import World3D, configure_demo_map_3d, render_world_3d
from sim3d.path3d import Path3D, PathStep3D, apply_step, init_world_for_path
from main_demo_3d import START_POSE, build_world_and_path


def _load_path(path_json: Path) -> Path3D:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = [PathStep3D(**s) for s in data["steps"]]
    data = {k: v for k, v in data.items() if k != "steps"}
    return Path3D(steps=steps, **data)


def _fresh_world() -> World3D:
    sx, sy, sz, _ = START_POSE
    world = World3D()
    configure_demo_map_3d(world, start_x=sx, start_y=sy, start_z=sz)
    return world


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Interactive rotatable 3D viewer for the needle plan.")
    p.add_argument("--path", type=str, default=None, help="path.json to load (else run planner).")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retract", dest="allow_retract", action="store_true")
    args = p.parse_args(argv)

    if args.path:
        path = _load_path(Path(args.path))
        world = _fresh_world()
    else:
        world, path, _dp_frontier = build_world_and_path(
            steps=int(args.steps),
            step_size=float(args.step_size),
            seed=int(args.seed),
            allow_retract=bool(args.allow_retract),
        )

    n = len(path.steps)

    def world_at(step_idx: int) -> World3D:
        """Replay the path into a fresh world up to step_idx (inclusive)."""
        w = _fresh_world()
        init_world_for_path(w, path)
        for step in path.steps[:step_idx]:
            apply_step(w, step)
        return w

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.15)
    slider_ax = fig.add_axes([0.18, 0.05, 0.64, 0.03])
    step_slider = Slider(slider_ax, "Step", 0, n, valinit=n, valstep=1)

    first_draw = True

    def draw(step_idx: int) -> None:
        nonlocal first_draw
        step_idx = int(step_idx)
        w = world_at(step_idx)
        # Redrawing calls ax.cla(), which resets the axis limits (zoom) back to
        # the full world extent every time -- save/restore them so scroll-zoom
        # survives moving the slider. Orbit rotation (elev/azim) isn't touched
        # by cla() in this matplotlib version, so it already persists on its own.
        if not first_draw:
            xlim3d, ylim3d, zlim3d = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        render_world_3d(ax, w, clear=True)
        if not first_draw:
            ax.set_xlim3d(xlim3d)
            ax.set_ylim3d(ylim3d)
            ax.set_zlim3d(zlim3d)
        first_draw = False
        healed = int((w.healed_mask.sum()) if w.healed_mask is not None else 0)
        total = int(path.total_pois)
        ax.set_title(
            f"IRIS 3D needle path  |  step {step_idx}/{n}  |  POIs healed {healed}/{total}\n"
            f"(drag to rotate, use the slider to scrub)"
        )
        fig.canvas.draw_idle()

    step_slider.on_changed(draw)
    draw(n)
    plt.show()


if __name__ == "__main__":
    main()
