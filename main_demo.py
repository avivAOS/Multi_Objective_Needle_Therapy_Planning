"""
Demo runner / plumbing shared by the planners: build the world, generate the paths
graph, dispatch to a planner (random / iris / dp), then save and visualize results.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import pickle
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from sim2d import DEFAULT_VIEW_X_LIMITS, DEFAULT_VIEW_Y_LIMITS, Orientation, World2D
from sim2d.demo_map import configure_demo_map
from sim2d.paths_graph import (
    DOSAGE_LEVELS,
    generate_paths_graph,
    inject_coverage,
    make_dosage_fn,
    make_inject_coverage_fn,
    plot_paths_graph,
)
from sim2d.run_logging import RunTimer, configure_run_logging

from path2d import RandomPath, random_path_from_paths_graph

from graph_utils import Graph as PlannerGraph
from graph_utils import GraphPath as PlannerGraphPath
from graph_utils import NodeId, build_adjacency
from iris_on_graph import IrisPlanner, generate_iris_plan_on_graph, log_iris_planner_metrics
from pareto_dominance import EpsVec, Plan
from dp_on_graph import MAX_PLANS_PER_NODE, DPStats, reconstruct_graph_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------
def build_demo_world(
    *,
    start_x: float,
    start_y: float,
    steps: int,
    step_size: float,
) -> World2D:
    world = World2D()
    world.x_limits = DEFAULT_VIEW_X_LIMITS
    world.y_limits = DEFAULT_VIEW_Y_LIMITS
    configure_demo_map(world, start_x=start_x, start_y=start_y)
    return world


# ---------------------------------------------------------------------------
# Simulation & replay
# ---------------------------------------------------------------------------
class PathSimulator:
    def __init__(
        self,
        world: World2D,
        path: RandomPath,
        pause: float = 0.05,
        output_dir: Optional[Path] = None,
        no_gui: bool = False,
        title_prefix: str = "Path",
    ) -> None:
        self.world = world
        self.path = path
        self.pause = pause
        self.output_dir = output_dir
        self.no_gui = no_gui
        self.title_prefix = str(title_prefix)

    def initialize_state(self) -> None:
        start_orientation = Orientation[self.path.start_orientation]
        theta0 = start_orientation.to_angle()
        self.world.reset_trail()
        self.world.set_robot_state(self.path.start_x, self.path.start_y, theta0)
        self.world.append_trail_point(self.path.start_x, self.path.start_y)

    def run(self) -> None:
        if self.no_gui:
            import matplotlib

            matplotlib.use("Agg", force=True)

        fig, ax = plt.subplots()
        replay_requested = False
        if not self.no_gui:
            plt.ion()
            # Let the window actually get mapped to the screen before we
            # start capturing frames: on some backends (e.g. TkAgg), the
            # canvas reports a placeholder physical size until it's first
            # shown, which would otherwise make the very first captured
            # frame a different shape than the rest.
            fig.canvas.draw()
            plt.pause(0.01)

            def on_key(event) -> None:
                nonlocal replay_requested
                if event.key == "r":
                    replay_requested = True

            fig.canvas.mpl_connect("key_press_event", on_key)
            print(
                "The animation will loop automatically. Press 'r' in the plot "
                "window at any time to restart immediately; close the window to stop."
            )

        frames: list = []

        def play_once() -> str:
            """Run through all steps once. Returns 'completed', 'replay', or 'closed'."""
            nonlocal replay_requested, frames
            self.initialize_state()
            self.world.reset_injection_paint()
            frames = []
            inject_circles: List[Tuple[float, float, float]] = []
            first_frame_shape: Optional[tuple[int, int, int]] = None
            for step in self.path.steps:
                if not self.no_gui and not plt.fignum_exists(fig.number):
                    return "closed"
                if replay_requested:
                    return "replay"

                _apply_path_step(self.world, step)
                circle = _step_inject_circle(step)
                if circle is not None:
                    inject_circles.append(circle)

                collided = self.world.in_collision()
                self.world.render(ax=ax)
                _draw_inject_circles(ax, inject_circles)

                if step.is_inject:
                    ax.set_title(f"{self.title_prefix} | step {step.step_index + 1}/{self.path.num_steps} | inject")
                elif step.action == "retract":
                    ax.set_title(
                        f"{self.title_prefix} | step {step.step_index + 1}/{self.path.num_steps} | retract | "
                        f"heading {step.to_orientation} | collision: {collided}"
                    )
                else:
                    ax.set_title(
                        f"{self.title_prefix} | step {step.step_index + 1}/{self.path.num_steps} | "
                        f"heading {step.to_orientation} | collision: {collided}"
                    )

                fig.canvas.draw()
                if self.output_dir is not None:
                    canvas = fig.canvas
                    # Use physical (renderer) pixel dimensions, not logical ones:
                    # on a scaled display (e.g. Windows at 125%/150%), the GUI
                    # canvas backend's buffer is sized in physical pixels while
                    # get_width_height() defaults to logical pixels, which would
                    # otherwise make the reshape below fail.
                    w, h = canvas.get_width_height(physical=True)
                    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
                    argb = argb.reshape((h, w, 4))
                    image = argb[:, :, 1:4].copy()
                    if first_frame_shape is None:
                        first_frame_shape = image.shape
                    if image.shape == first_frame_shape:
                        frames.append(image)
                    else:
                        # Defensive: a manual window resize mid-run would change
                        # the physical canvas size; skip that frame rather than
                        # fail the whole GIF write with a shape mismatch.
                        print(
                            f"Warning: skipping frame with shape {image.shape} "
                            f"(expected {first_frame_shape}); window was likely resized mid-run."
                        )
                if not self.no_gui:
                    plt.pause(self.pause)
            return "completed"

        while True:
            replay_requested = False
            outcome = play_once()
            if outcome == "closed":
                break
            if self.no_gui:
                break  # headless: no interactive replay loop, just finish once
            if outcome == "replay":
                continue  # 'r' was pressed mid-animation; restart immediately right away

            # outcome == "completed": auto-loop, like the saved GIF does.
            # Hold on the finished frame briefly first so it doesn't flicker
            # straight back to step 1; 'r' (or closing the window) interrupts the hold.
            hold_until = time.monotonic() + 1.0
            while (
                plt.fignum_exists(fig.number)
                and not replay_requested
                and time.monotonic() < hold_until
            ):
                plt.pause(0.1)
            if not plt.fignum_exists(fig.number):
                break
            # else: loop back and play_once() again (auto-loop, or 'r' skipped the hold).

        if self.output_dir is not None and frames:
            gif_path = self.output_dir / "simulation.gif"
            try:
                import imageio.v3 as iio
            except ModuleNotFoundError:
                print(
                    "Warning: `imageio` not installed; skipping GIF output. "
                    f"Would have written: {gif_path}"
                )
            else:
                iio.imwrite(gif_path, frames, duration=self.pause or 0.5, loop=0)

        if not self.no_gui:
            plt.ioff()
            if plt.fignum_exists(fig.number):
                plt.close(fig)
        else:
            plt.close(fig)


def _paint_inject_step(world: World2D, step) -> None:
    """
    Paint one inject step. DP steps carry exact per-dose covered/damaged cells, so
    paint those for a cell-for-cell match; IRIS/random steps have no dosage damage
    and fall back to the fixed-radius `inject_at`.
    """
    if getattr(step, "inject_damaged_points", None) is not None:
        world.paint_injection_cells(
            step.inject_covered_points or (),
            step.inject_damaged_points or (),
        )
    else:
        world.inject_at(step.x, step.y)


def _apply_path_step(world: World2D, step) -> None:
    """Apply one RandomPath step to `world` (mirrors PathSimulator.run's body)."""
    world.state = np.array([step.x, step.y, step.theta], dtype=float)
    if step.is_inject:
        _paint_inject_step(world, step)
    elif step.action == "retract":
        world.retract_trail_step()
    else:
        world.append_trail_point(step.x, step.y)


def _step_inject_circle(step) -> Optional[Tuple[float, float, float]]:
    """The (x, y, radius) injection disk for a step, or None if it isn't a
    radius-carrying inject (legacy IRIS/random injects have no radius)."""
    if getattr(step, "is_inject", False) and getattr(step, "inject_radius", None):
        return (float(step.x), float(step.y), float(step.inject_radius))
    return None


def _draw_inject_circles(ax: "plt.Axes", circles: List[Tuple[float, float, float]]) -> None:
    """Outline each injection disk as a dashed circle, showing the true affected
    area over the discrete green/purple cell paint."""
    for cx, cy, r in circles:
        if r and r > 0.0:
            ax.add_patch(
                plt.Circle(
                    (cx, cy), r, fill=False, edgecolor="black",
                    linestyle="--", linewidth=1.2, alpha=0.7, zorder=4,
                )
            )


def _replay_world_to(
    world: World2D, path: RandomPath, idx: int
) -> Tuple[object, int, int, List[Tuple[float, float, float]]]:
    """
    Replay the path from scratch up to step `idx` (-1 = start pose), so it's correct
    no matter how the animation jumps between frames. Returns (last_step, cumulative
    healed cells, cumulative damaged cells, injection circles up to idx).
    """
    world.reset_injection_paint()
    world.reset_trail()
    start_ori = Orientation[path.start_orientation]
    world.set_robot_state(path.start_x, path.start_y, start_ori.to_angle())
    world.append_trail_point(path.start_x, path.start_y)

    last_step = None
    healed = 0
    damaged = 0
    circles: List[Tuple[float, float, float]] = []
    for s in path.steps[: idx + 1]:
        _apply_path_step(world, s)
        last_step = s
        if s.is_inject:
            if s.inject_covered_points_cumulative is not None:
                healed = len(s.inject_covered_points_cumulative)
            if s.inject_damaged_points_cumulative is not None:
                damaged = len(s.inject_damaged_points_cumulative)
            circle = _step_inject_circle(s)
            if circle is not None:
                circles.append(circle)
    return last_step, healed, damaged, circles


# The four DP objectives, as (label, accessor, optimization direction). Used by
# both the frontier-projection plots and the parallel-coordinates view so the
# axis set stays in one place if a 5th objective is ever added.
_OBJECTIVES = [
    ("length", lambda p: float(p.length), "min"),
    ("coverage", lambda p: p.coverage.bit_count(), "max"),
    ("damage", lambda p: p.damage.bit_count(), "min"),
    ("doses", lambda p: int(p.dose_count), "min"),
]


# ---------------------------------------------------------------------------
# Frontier selection & dashboard visualization
# ---------------------------------------------------------------------------
def save_frontier_snapshot(
    path: Path,
    *,
    kind: str,
    plans: List[Plan],
    start_key,
    nodes: dict,
    world,
    seed: int,
    step_size: float,
) -> None:
    """
    Pickle everything `generate_pareto_dashboard[_3d]` needs to reopen an
    interactive dashboard window later, without recomputing the graph or
    re-running DP. `kind` is `"2d"` or `"3d"` -- `replay_dashboard.py` uses it
    to pick which dashboard builder to call. `world` is saved as-is (it may
    already carry a completed replay's trail/paint from this run, but the
    dashboard builder resets a fresh copy per panel anyway, so that's harmless
    -- only the static geometry, tumor/healthy masks matter, and those are
    untouched by a replay).

    `nodes` is trimmed down to only the entries the three selected plans'
    replays actually touch (a few dozen steps each), not the whole paths
    graph -- untrimmed, this held every node the graph-gen BFS ever visited
    (tens of thousands in 3D) even though the dashboard only ever replays
    these three plans, ballooning the file to hundreds of MB for no benefit.
    """
    selections, _pool = _select_frontier_plans(plans)
    needed_keys = {start_key}
    for _name, plan, _color in selections:
        graph_path = reconstruct_graph_path(plan, start_node=start_key, seed=int(seed))
        for step in graph_path.steps:
            needed_keys.add(step.node)
            if step.previous_node is not None:
                needed_keys.add(step.previous_node)
            if step.next_node is not None:
                needed_keys.add(step.next_node)
    trimmed_nodes = {k: nodes[k] for k in needed_keys if k in nodes}

    bundle = {
        "kind": kind,
        "plans": plans,
        "start_key": start_key,
        "nodes": trimmed_nodes,
        "world": world,
        "seed": seed,
        "step_size": step_size,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def _select_frontier_plans(plans: List[Plan]) -> Tuple[list, list]:
    """
    Pick three representative plans -- Max Coverage, Min Damage, Min Length -- and
    return `(selections, treated_pool)`. Each pick is the true best-in-category
    plan among `treated_pool` (plans with `dose_count > 0`), so the labels are
    honest: "Min Damage" really is the minimum-damage plan, not the minimum
    damage plan within some arbitrary coverage-percentile cutoff.

    The only plans excluded are the trivial "never inject" ones (dose_count ==
    0). Every frontier has one -- it trivially wins length/damage/dose-count by
    doing nothing -- so without excluding it, Min Damage and Min Length would
    both degenerate to that single no-op plan instead of showing a real
    treatment trade-off.
    """
    treated_pool = [p for p in plans if p.dose_count > 0] or list(plans)

    max_coverage_plan = min(
        treated_pool, key=lambda p: (-p.coverage.bit_count(), p.damage.bit_count(), p.length)
    )
    min_damage_plan = min(
        treated_pool, key=lambda p: (p.damage.bit_count(), -p.coverage.bit_count(), p.length)
    )
    min_length_plan = min(
        treated_pool, key=lambda p: (p.length, p.damage.bit_count(), -p.coverage.bit_count())
    )
    selections = [
        ("Max Coverage", max_coverage_plan, "tab:orange"),
        ("Min Damage", min_damage_plan, "lightblue"),
        ("Min Length", min_length_plan, "teal"),
    ]
    return selections, treated_pool


def _draw_parallel_coords(ax: "plt.Axes", plans: List[Plan], selections: list) -> None:
    """Parallel-coordinates view of the 4D frontier: one faint polyline per plan
    across the four normalized objective axes, the three picks drawn bold."""
    values = [[acc(p) for p in plans] for (_n, acc, _d) in _OBJECTIVES]
    mins = [min(v) for v in values]
    maxs = [max(v) for v in values]

    def _norm(k: int, val: float) -> float:
        lo, hi = mins[k], maxs[k]
        return 0.5 if hi <= lo else (float(val) - lo) / (hi - lo)

    xs = list(range(len(_OBJECTIVES)))
    for p in plans:
        ys = [_norm(k, acc(p)) for k, (_n, acc, _d) in enumerate(_OBJECTIVES)]
        ax.plot(xs, ys, color="0.85", linewidth=0.5, alpha=0.4, zorder=1)
    for name, plan, color in selections:
        ys = [_norm(k, acc(plan)) for k, (_n, acc, _d) in enumerate(_OBJECTIVES)]
        ax.plot(xs, ys, color=color, linewidth=2.4, alpha=0.95, zorder=4,
                marker="o", markersize=5, label=name)
    for k in xs:
        ax.axvline(k, color="0.7", linewidth=0.8, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{n}\n({d})\n[{mins[k]:.0f}..{maxs[k]:.0f}]" for k, (n, _a, d) in enumerate(_OBJECTIVES)],
        fontsize=8,
    )
    ax.set_yticks([])
    ax.set_title("4D frontier: parallel coords (1 line = 1 plan)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)


def _draw_frontier_3d(ax: "plt.Axes", plans: List[Plan], selections: list) -> None:
    """
    3D scatter of the frontier in (coverage, damage, length) space -- the other
    three objectives are already covered by the parallel-coords plot, and a
    triangle needs exactly 3 axes. The three selected plans are drawn as bold
    markers joined by a translucent triangle, so their spread relative to the
    whole frontier (and to each other) is visible at a glance instead of
    needing several separate 2D projections.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cov = [p.coverage.bit_count() for p in plans]
    dam = [p.damage.bit_count() for p in plans]
    length = [float(p.length) for p in plans]
    ax.scatter(cov, dam, length, s=10, color="0.75", alpha=0.6, zorder=1, label="all plans")

    verts = []
    for name, plan, color in selections:
        x, y, z = plan.coverage.bit_count(), plan.damage.bit_count(), float(plan.length)
        verts.append((x, y, z))
        ax.scatter([x], [y], [z], s=90, color=color, edgecolors="black",
                   linewidths=1.0, zorder=5, label=name)

    if len(verts) >= 3:
        tri = Poly3DCollection([verts], alpha=0.18, facecolor="gray",
                               edgecolor="0.3", linewidths=1.2, zorder=4)
        ax.add_collection3d(tri)

    ax.set_xlabel("coverage (max)", fontsize=8)
    ax.set_ylabel("damage (min)", fontsize=8)
    ax.set_zlabel("length (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Pareto frontier: coverage x damage x length\n(triangle = the 3 picks)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def generate_pareto_dashboard(
    plans: List[Plan],
    start_key: NodeId,
    nodes: dict,
    world: World2D,
    *,
    seed: int = 0,
    step_size: float = 1.0,
    n_frames: int = 60,
    interval_ms: int = 120,
) -> Tuple["plt.Figure", FuncAnimation]:
    """
    Animated DP-frontier dashboard. Top row: three live sim panels (Max Coverage /
    Min Damage / Min Length), each replaying its plan in its own World2D copy.
    Bottom row: the 4D frontier as parallel coords + a 3D coverage/damage/length
    scatter with the three picks highlighted. Returns `(fig, ani)` -- the caller
    must keep `ani` alive or playback is garbage-collected.
    """
    if not plans:
        raise ValueError("generate_pareto_dashboard requires at least one plan")

    covs = [p.coverage.bit_count() for p in plans]
    total_ppois = int(world.tumor_mask.sum()) if world.tumor_mask is not None else max(covs, default=0)

    # --- Plan selection (shared with the frontier graphs below) ---
    selections, _treated_pool = _select_frontier_plans(plans)

    # --- Build one independent (path, world) simulation per selected plan ---
    sim_panels = []  # (name, plan, color, RandomPath, World2D)
    for name, plan, color in selections:
        graph_path = reconstruct_graph_path(plan, start_node=start_key, seed=int(seed))
        rp = _dp_randompath_from_graph_path(
            graph_path=graph_path, seed=int(seed), step_size=float(step_size), nodes=nodes
        )
        panel_world = copy.deepcopy(world)
        panel_world.reset_injection_paint()
        panel_world.reset_trail()
        sim_panels.append((name, plan, color, rp, panel_world))

    # Layout: top row = 3 animated sims; bottom row = parallel coords + 3D frontier.
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0])
    map_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    # --- Bottom row: the 4D frontier (static, drawn once) ---
    ax_pc = fig.add_subplot(gs[1, 0])
    _draw_parallel_coords(ax_pc, plans, selections)
    ax_3d = fig.add_subplot(gs[1, 1:], projection="3d")
    _draw_frontier_3d(ax_3d, plans, selections)

    fig.tight_layout()
    # Per-frame titles (two lines) are set later in `update`, so reserve headroom.
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)

    n_frames = max(int(n_frames), 1)
    hold_frames = 10
    total_frames = n_frames + hold_frames

    def update(frame_idx: int):
        # Map the frame onto a 0..1 progress fraction; the last `hold_frames`
        # frames clamp to 1.0 so the finished panels linger before looping.
        frac = min(frame_idx, n_frames - 1) / max(n_frames - 1, 1)
        for (name, plan, color, rp, panel_world), ax in zip(sim_panels, map_axes):
            n_steps = len(rp.steps)
            idx = int(round(frac * (n_steps - 1))) if n_steps > 0 else -1
            last_step, healed, damaged, circles = _replay_world_to(panel_world, rp, idx)
            # render() clears the axes and resets to the full world view every
            # call; without saving/restoring, any toolbar zoom/pan gets wiped
            # out on the very next animation tick. Skip on frame 0, when the
            # axes haven't been drawn yet (get_xlim/ylim would be matplotlib's
            # uninitialized (0, 1) default, not a real zoom to preserve).
            if frame_idx > 0:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
            panel_world.render(ax=ax, clear=True)
            if frame_idx > 0:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            _draw_inject_circles(ax, circles)

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            action = last_step.action if last_step is not None else "start"
            shown = (idx + 1) if n_steps > 0 else 0
            ax.set_title(
                f"{name}  cov={plan.coverage.bit_count()}/{total_ppois} "
                f"dam={plan.damage.bit_count()} doses={plan.dose_count} len={plan.length:.0f}\n"
                f"step {shown}/{n_steps} [{action}]  "
                f"healed={healed}  damaged={damaged}",
                fontsize=9,
            )
        return []

    ani = FuncAnimation(
        fig, update, frames=total_frames, interval=interval_ms, blit=False, repeat=True
    )
    return fig, ani


# ---------------------------------------------------------------------------
# Coverage / damage metrics
# ---------------------------------------------------------------------------
def _poi_universe_from_nodes(nodes: dict) -> set[tuple[float, float]]:
    """Union of all inject-coverage POIs across the paths graph."""
    all_pts: set[tuple[float, float]] = set()
    for node in nodes.values():
        for x, y in inject_coverage(node):
            all_pts.add((float(x), float(y)))
    return all_pts


def _covered_pois_from_random_path(path: RandomPath) -> set[tuple[float, float]]:
    """POIs covered after replay, from the last cumulative inject snapshot (if any)."""
    for step in reversed(path.steps):
        if step.inject_covered_points_cumulative is not None:
            return {(float(x), float(y)) for x, y in step.inject_covered_points_cumulative}
    return set()


def _healthy_universe_from_nodes(nodes: dict) -> set[tuple[float, float]]:
    """Union of all dosage-damage HPOIs across the paths graph (max dosage level)."""
    from sim2d.paths_graph import DOSAGE_LEVELS, dosage_damage

    all_pts: set[tuple[float, float]] = set()
    max_dose = max(DOSAGE_LEVELS)
    for node in nodes.values():
        for x, y in dosage_damage(node, max_dose):
            all_pts.add((float(x), float(y)))
    return all_pts


def _damaged_pois_from_random_path(path: RandomPath) -> set[tuple[float, float]]:
    """HPOIs damaged after replay, from the last cumulative damage snapshot (if any)."""
    for step in reversed(path.steps):
        if step.inject_damaged_points_cumulative is not None:
            return {(float(x), float(y)) for x, y in step.inject_damaged_points_cumulative}
    return set()


def log_hpoi_damage_after_run(path: RandomPath, nodes: dict) -> dict[str, int]:
    """Log how many HPOIs were damaged vs spared when a DP simulation run finishes."""
    universe = _healthy_universe_from_nodes(nodes)
    damaged = _damaged_pois_from_random_path(path)
    n_total = len(universe)
    n_damaged = len(damaged & universe)
    n_spared = n_total - n_damaged
    logger.info(
        "Run finished: HPOIs damaged=%s spared=%s (of %s)",
        n_damaged,
        n_spared,
        n_total,
    )
    return {"hpoi_damaged": n_damaged, "hpoi_spared": n_spared, "hpoi_total": n_total}


def log_poi_coverage_after_run(path: RandomPath, nodes: dict) -> dict[str, int]:
    """Log how many POIs were covered vs remaining when a simulation run finishes."""
    universe = _poi_universe_from_nodes(nodes)
    covered = _covered_pois_from_random_path(path)
    n_total = len(universe)
    n_covered = len(covered & universe)
    n_remaining = n_total - n_covered
    logger.info(
        "Run finished: POIs covered=%s remaining=%s (of %s)",
        n_covered,
        n_remaining,
        n_total,
    )
    return {"poi_covered": n_covered, "poi_remaining": n_remaining, "poi_total": n_total}


# ---------------------------------------------------------------------------
# Output & replay helpers (JSON, GIF replay, run dir, run index)
# ---------------------------------------------------------------------------
def save_path_json(path_obj: object, json_out: str | Path) -> None:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(asdict(path_obj), f, indent=2)


def replay_path_in_world(
    path: RandomPath,
    *,
    pause: float = 0.05,
    output_dir: Optional[Path] = None,
    no_gui: bool = False,
    world: Optional[World2D] = None,
    title_prefix: str = "Path",
) -> None:
    if world is None:
        world = build_demo_world(
            start_x=path.start_x,
            start_y=path.start_y,
            steps=path.num_steps,
            step_size=path.step_size,
        )
    PathSimulator(
        world,
        path,
        pause=pause,
        output_dir=output_dir,
        no_gui=no_gui,
        title_prefix=title_prefix,
    ).run()


# The only two valid `dim` values -- shared so the on-disk directory segment
# and the runs_index.csv "dim" column value can never drift apart by typo.
DIM_2D = "2d"
DIM_3D = "3d"


def _prepare_run_dir(*, outputs_root: Path, algo_name: str, dim: str, run_name: str) -> Path:
    d = outputs_root / algo_name / dim / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# Fixed column order for outputs/runs_index.csv. Every run appends exactly one
# row; fields that don't apply to a given algo are left blank rather than
# changing the column set, so the CSV stays easy to sort/filter across runs.
_RUN_INDEX_FIELDNAMES = [
    "timestamp", "run_name", "algo", "dim", "seed", "steps", "step_size",
    "num_nodes", "num_edges",
    "eps_len", "eps_cov", "eps_dam", "eps_ndose", "dosage_levels", "max_plans_per_node",
    "max_outer_iters", "max_expansions", "p0", "epsilon0",
    "chosen_g", "chosen_coverage", "chosen_damage", "chosen_ndose", "frontier_size",
    "poi_covered", "poi_remaining", "poi_total",
    "hpoi_damaged", "hpoi_spared", "hpoi_total",
    "dp_time_seconds", "dp_total_comparisons", "dp_peak_frontier_size",
    "dp_peak_memory_mb", "dp_reordering_enabled",
    "output_dir",
]


def _append_run_index(outputs_root: Path, row: dict) -> Path:
    """Append one summary row to outputs/runs_index.csv (one row per run, all algos),
    writing the header on first use.

    If an existing file's header doesn't match the current `_RUN_INDEX_FIELDNAMES`
    (e.g. an older run wrote it before a column was added), appending would
    silently misalign every column against the stale header -- instead, the
    old file is preserved under a `.bak` name and a fresh one is started with
    the current header.
    """
    index_path = outputs_root / "runs_index.csv"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            existing_header = f.readline().rstrip("\n\r")
        if existing_header != ",".join(_RUN_INDEX_FIELDNAMES):
            backup_path = index_path.with_suffix(".csv.bak")
            index_path.replace(backup_path)
            print(
                f"Note: runs_index.csv had an outdated column layout; "
                f"preserved the old file as {backup_path} and started a new one."
            )
    file_exists = index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RUN_INDEX_FIELDNAMES, restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return index_path


# ---------------------------------------------------------------------------
# Planner runners (random / iris / dp) and their path adapters
# ---------------------------------------------------------------------------
def run_random_alg(
    *,
    seed: int,
    num_steps: int,
    step_size: float,
    start_key: Tuple[float, float, int],
    nodes: dict,
    edges: set,
    world: World2D,
    out_dir: Path,
    no_gui: bool,
) -> dict:
    path = random_path_from_paths_graph(
        seed=int(seed),
        num_steps=int(num_steps),
        step_size=float(step_size),
        start_key=start_key,
        nodes=nodes,
        edges=edges,
    )
    save_path_json(path, json_out=out_dir / "path.json")
    replay_path_in_world(path, output_dir=out_dir, no_gui=bool(no_gui), world=world, title_prefix="Random")
    result = log_poi_coverage_after_run(path, nodes)
    result["output_dir"] = str(out_dir)
    return result


def _iris_randompath_from_graph_path(
    *,
    graph_path: PlannerGraphPath[Tuple[float, float, int]],
    seed: int,
    step_size: float,
    nodes: dict,
) -> RandomPath:
    """Convert an IRIS `GraphPath` into a `RandomPath` for 2D replay."""
    from path2d import PathStep

    start_key = graph_path.start_node
    first_node = nodes[start_key]
    start_x = float(first_node["x"])
    start_y = float(first_node["y"])
    start_orientation: Orientation = first_node["ori"]

    coverage_fn = make_inject_coverage_fn(nodes)
    cumulative_covered: set[tuple[float, float]] = set()
    steps: List[PathStep] = []

    for i, st in enumerate(graph_path.steps):
        action = str(st.action)
        pre_key = st.node
        if action == "move":
            post_key = st.next_node
            if post_key is None:
                raise ValueError("IRIS move step missing next_node.")
        elif action == "retract":
            post_key = st.previous_node
            if post_key is None:
                raise ValueError("IRIS retract step missing previous_node (post-retract node).")
        else:
            post_key = pre_key

        pre_node = nodes[pre_key]
        post_node = nodes[post_key]
        from_ori: Orientation = pre_node["ori"]
        to_ori: Orientation = post_node["ori"]
        theta = to_ori.to_angle()

        inject_covered_points: Optional[List[List[float]]] = None
        inject_covered_points_cumulative: Optional[List[List[float]]] = None
        if action == "inject":
            covered_now = set(coverage_fn(pre_key) or ())
            cumulative_covered |= covered_now
            inject_covered_points = [[float(x), float(y)] for x, y in sorted(covered_now)]
            inject_covered_points_cumulative = [
                [float(x), float(y)] for x, y in sorted(cumulative_covered)
            ]

        steps.append(
            PathStep(
                step_index=i,
                action=action,
                from_orientation=from_ori.name,
                to_orientation=to_ori.name,
                x=float(post_node["x"]),
                y=float(post_node["y"]),
                theta=float(theta),
                is_inject=(action == "inject"),
                inject_covered_points=inject_covered_points,
                inject_covered_points_cumulative=inject_covered_points_cumulative,
            )
        )

    return RandomPath(
        seed=int(seed),
        num_steps=int(len(steps)),
        start_x=float(start_x),
        start_y=float(start_y),
        start_orientation=start_orientation.name,
        step_size=float(step_size),
        steps=steps,
    )


def run_iris(
    *,
    seed: int,
    step_size: float,
    start_key: Tuple[float, float, int],
    nodes: dict,
    edges: set,
    world: World2D,
    out_dir: Path,
    no_gui: bool,
    max_outer_iters: int,
    max_expansions: int,
    p0: Optional[float] = None,
    epsilon0: Optional[float] = None,
    allow_retract: bool = False,
) -> dict:
    # Build graph for IRIS (uniform edge cost is assumed by the IRIS planner).
    adj = build_adjacency(edges, directed=True)
    graph = PlannerGraph(nodes=tuple(nodes.keys()), adjacency=adj, directed=True)

    coverage_fn = make_inject_coverage_fn(nodes)
    iris_plan = generate_iris_plan_on_graph(
        graph=graph,
        start_node=start_key,
        seed=int(seed),
        coverage_fn=coverage_fn,
        p0=p0,
        epsilon0=epsilon0,
        max_outer_iters=int(max_outer_iters),
        max_expansions=int(max_expansions),
        allow_retract=bool(allow_retract),
    )

    path = _iris_randompath_from_graph_path(
        graph_path=iris_plan,
        seed=int(seed),
        step_size=float(step_size),
        nodes=nodes,
    )
    save_path_json(path, json_out=out_dir / "path.json")
    replay_path_in_world(path, output_dir=out_dir, no_gui=bool(no_gui), world=world, title_prefix="IRIS")
    result = log_poi_coverage_after_run(path, nodes)
    result["output_dir"] = str(out_dir)
    return result


def _dp_randompath_from_graph_path(
    *,
    graph_path: PlannerGraphPath[Tuple[float, float, int]],
    seed: int,
    step_size: float,
    nodes: dict,
) -> RandomPath:
    """Convert a DP `GraphPath` into a `RandomPath` for 2D replay -- like
    `_iris_randompath_from_graph_path`, but also records per-dose damage points."""
    from path2d import PathStep
    from sim2d.paths_graph import dosage_coverage, dosage_damage, dosage_to_radius

    start_key = graph_path.start_node
    first_node = nodes[start_key]
    start_x = float(first_node["x"])
    start_y = float(first_node["y"])
    start_orientation: Orientation = first_node["ori"]

    cumulative_covered: set[tuple[float, float]] = set()
    cumulative_damaged: set[tuple[float, float]] = set()
    steps: List[PathStep] = []

    for i, st in enumerate(graph_path.steps):
        action = str(st.action)
        pre_key = st.node
        if action == "move":
            post_key = st.next_node
            if post_key is None:
                raise ValueError("DP move step missing next_node.")
        elif action == "retract":
            post_key = st.previous_node
            if post_key is None:
                raise ValueError("DP retract step missing previous_node (post-retract node).")
        else:
            post_key = pre_key

        pre_node = nodes[pre_key]
        post_node = nodes[post_key]
        from_ori: Orientation = pre_node["ori"]
        to_ori: Orientation = post_node["ori"]
        theta = to_ori.to_angle()

        inject_covered_points: Optional[List[List[float]]] = None
        inject_covered_points_cumulative: Optional[List[List[float]]] = None
        inject_damaged_points: Optional[List[List[float]]] = None
        inject_damaged_points_cumulative: Optional[List[List[float]]] = None
        inject_radius: Optional[float] = None
        if action == "inject":
            dosage = int(st.dosage) if st.dosage is not None else 0
            covered_now = dosage_coverage(pre_node, dosage)
            damaged_now = dosage_damage(pre_node, dosage)
            cumulative_covered |= covered_now
            cumulative_damaged |= damaged_now
            inject_covered_points = [[float(x), float(y)] for x, y in sorted(covered_now)]
            inject_covered_points_cumulative = [
                [float(x), float(y)] for x, y in sorted(cumulative_covered)
            ]
            inject_damaged_points = [[float(x), float(y)] for x, y in sorted(damaged_now)]
            inject_damaged_points_cumulative = [
                [float(x), float(y)] for x, y in sorted(cumulative_damaged)
            ]
            inject_radius = float(dosage_to_radius(dosage)[0])

        steps.append(
            PathStep(
                step_index=i,
                action=action,
                from_orientation=from_ori.name,
                to_orientation=to_ori.name,
                x=float(post_node["x"]),
                y=float(post_node["y"]),
                theta=float(theta),
                is_inject=(action == "inject"),
                inject_covered_points=inject_covered_points,
                inject_covered_points_cumulative=inject_covered_points_cumulative,
                inject_damaged_points=inject_damaged_points,
                inject_damaged_points_cumulative=inject_damaged_points_cumulative,
                inject_radius=inject_radius,
            )
        )

    return RandomPath(
        seed=int(seed),
        num_steps=int(len(steps)),
        start_x=float(start_x),
        start_y=float(start_y),
        start_orientation=start_orientation.name,
        step_size=float(step_size),
        steps=steps,
    )


def run_dp(
    *,
    seed: int,
    step_size: float,
    start_key: Tuple[float, float, int],
    nodes: dict,
    edges: set,
    world: World2D,
    out_dir: Path,
    no_gui: bool,
    dosage_levels: Tuple[int, ...],
    eps_vec: EpsVec,
    max_plans_per_node: Optional[int],
    depth_scaled_eps: bool = False,
    reorder_children: bool = True,
    timer: Optional[RunTimer] = None,
) -> dict:
    adj = build_adjacency(edges, directed=True)
    graph = PlannerGraph(nodes=tuple(nodes.keys()), adjacency=adj, directed=True)

    # Expanded (instead of the one-call `generate_dp_plan_on_graph` wrapper)
    # so we keep `plans` (the full frontier) around for the dashboard and
    # the run-index summary, not just the single reconstructed GraphPath.
    from dp_on_graph import bfs_depth, build_dosage_masks, compute_dp

    dosage_fn = make_dosage_fn(nodes)
    depth_map = bfs_depth(graph, start_key)
    timer = timer if timer is not None else RunTimer()
    with timer.section("dp.masks_build"):
        coverage_masks, damage_masks = build_dosage_masks(
            graph=graph, dosage_fn=dosage_fn, dosage_levels=dosage_levels
        )
    cap_display = "uncapped" if max_plans_per_node is None else str(int(max_plans_per_node))
    print(
        f"Running DP over {len(nodes)} nodes (max_plans_per_node={cap_display}). "
        "This can take a while; progress below:",
        flush=True,
    )
    dp_stats = DPStats()
    tracemalloc.start()
    with timer.section("dp.compute"):
        memo = compute_dp(
            graph=graph,
            start_node=start_key,
            depth=depth_map,
            dosage_levels=dosage_levels,
            coverage_masks=coverage_masks,
            damage_masks=damage_masks,
            eps_vec=eps_vec,
            max_plans_per_node=max_plans_per_node,
            progress=True,
            depth_scaled_eps=bool(depth_scaled_eps),
            reorder_children=bool(reorder_children),
            stats=dp_stats,
        )
    _current_bytes, dp_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plans = memo[start_key]
    dp_peak_memory_mb = round(dp_peak_bytes / (1024 * 1024), 3)
    print(
        f"DP stats: {len(plans)} plans at start, "
        f"{dp_stats.total_comparisons()} comparisons, "
        f"peak frontier {dp_stats.peak_frontier_size()}, "
        f"peak memory {dp_peak_memory_mb}MB "
        f"-- full breakdown saved to dp_stats.json / dp_node_stats.csv once the run finishes",
        flush=True,
    )

    with timer.section("dp.select_frontier"):
        selections, treated_pool = _select_frontier_plans(plans)
        # Use Min Damage (the true min-damage treated plan) as the primary output
        # plan so simulation.gif reflects the damage-aware trade-off, not just max
        # coverage.
        best = next(p for _name, p, _color in selections if _name == "Min Damage")
        dp_plan = reconstruct_graph_path(best, start_node=start_key, seed=int(seed))

        path = _dp_randompath_from_graph_path(
            graph_path=dp_plan,
            seed=int(seed),
            step_size=float(step_size),
            nodes=nodes,
        )
        save_path_json(path, json_out=out_dir / "path.json")

    # Always headless here: the per-step simulation window is superseded by
    # the animated Pareto dashboard below, which is what the user actually
    # wants to watch. `simulation.gif` is still written to disk either way.
    # PathSimulator.run() force-switches the global matplotlib backend to
    # the non-interactive "Agg" when no_gui=True, so restore the original
    # interactive backend afterward or the dashboard's plt.show() below
    # would silently no-op.
    import matplotlib

    original_backend = matplotlib.get_backend()
    with timer.section("dp.replay"):
        replay_path_in_world(path, output_dir=out_dir, no_gui=True, world=world, title_prefix="DP")
    if not no_gui:
        matplotlib.use(original_backend, force=True)
    poi_result = log_poi_coverage_after_run(path, nodes)
    hpoi_result = log_hpoi_damage_after_run(path, nodes)

    logger.info(
        "DP frontier: %d plans; treated (dose_count>0)=%d. Picks: %s",
        len(plans),
        len(treated_pool),
        "; ".join(
            f"{name}(cov={p.coverage.bit_count()}, dam={p.damage.bit_count()}, "
            f"doses={p.dose_count}, len={p.length:.0f})"
            for name, p, _c in selections
        ),
    )

    frontier_path = out_dir / "frontier.pkl"
    save_frontier_snapshot(
        frontier_path, kind="2d", plans=plans, start_key=start_key, nodes=nodes,
        world=world, seed=int(seed), step_size=float(step_size),
    )
    print(
        f"Saved frontier to {frontier_path} -- reopen it anytime with "
        f"`python replay_dashboard.py {frontier_path}` (no recomputation needed).",
        flush=True,
    )

    # The 4D frontier graphs now live in the dashboard's bottom row (see
    # generate_pareto_dashboard), so each gif's choice is explained alongside it.
    with timer.section("dp.dashboard_render"):
        dashboard_fig, dashboard_ani = generate_pareto_dashboard(
            plans, start_key, nodes, world, seed=int(seed), step_size=float(step_size)
        )
        dashboard_path = out_dir / "dashboard.gif"
        print("Rendering dashboard animation...", flush=True)
        dashboard_ani.save(dashboard_path, writer="pillow")
    print(f"Saved dashboard to {dashboard_path}", flush=True)
    logger.info("Saved animated Pareto dashboard to %s", dashboard_path)
    if not no_gui:
        print("Showing dashboard window (close it to finish).", flush=True)
        plt.show(block=True)
    plt.close(dashboard_fig)

    # Written last, once every timer.section(...) above has actually run, so
    # "timing_seconds" reflects the whole DP run (masks/compute/frontier
    # selection/replay/dashboard), not just the compute_dp() call itself.
    dp_stats_path = out_dir / "dp_stats.json"
    dp_stats.peak_memory_bytes = int(dp_peak_bytes)
    dp_stats_summary = {
        "algorithm": "dp",
        "graph": {"num_nodes": len(nodes), "num_edges": len(edges)},
        "eps_vec": {
            "length_tol": eps_vec.length_tol,
            "coverage_tol": eps_vec.coverage_tol,
            "damage_tol": eps_vec.damage_tol,
            "dose_count_tol": eps_vec.dose_count_tol,
        },
        "max_plans_per_node": cap_display,
        "timing_seconds": timer.durations(),
        "frontier_size_at_start": len(plans),
        **dp_stats.to_summary_dict(),
    }
    with open(dp_stats_path, "w", encoding="utf-8") as f:
        json.dump(dp_stats_summary, f, indent=2)

    node_rows = dp_stats.to_node_rows()
    if node_rows:
        with open(out_dir / "dp_node_stats.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
            writer.writeheader()
            writer.writerows(node_rows)
    print(f"Saved DP profiling stats to {dp_stats_path} and dp_node_stats.csv", flush=True)

    return {
        **poi_result,
        **hpoi_result,
        "chosen_g": best.length,
        "chosen_coverage": best.coverage.bit_count(),
        "chosen_damage": best.damage.bit_count(),
        "chosen_ndose": best.dose_count,
        "frontier_size": len(plans),
        "output_dir": str(out_dir),
        "dp_time_seconds": round(timer.durations().get("dp.compute", 0.0), 4),
        "dp_total_comparisons": dp_stats.total_comparisons(),
        "dp_peak_frontier_size": dp_stats.peak_frontier_size(),
        "dp_peak_memory_mb": dp_peak_memory_mb,
        "dp_reordering_enabled": bool(reorder_children),
    }


# ---------------------------------------------------------------------------
# CLI & entry point
# ---------------------------------------------------------------------------
def _max_plans_per_node_arg(raw: str) -> Optional[int]:
    """argparse type for --max-plans-per-node: "none"/"unlimited" -> uncapped."""
    if raw.strip().lower() in ("none", "unlimited", ""):
        return None
    return int(raw)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run different planners on the same paths graph.")
    parser.add_argument("--steps", type=int, default=60, help="Number of steps used to generate the paths graph.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--step-size", type=float, default=0.5, help="Step size for each primitive move.")
    parser.add_argument("--start-x", type=float, default=0.0, help="Robot start x position.")
    parser.add_argument(
        "--start-y",
        type=float,
        default=0.25,
        help="Robot start y position (default: 0.25, just clear of the y=0 boundary margin).",
    )
    parser.add_argument(
        "--start-orientation",
        type=str,
        default="UP",
        choices=[o.name for o in Orientation],
        help="Robot start heading, one of the 8 discrete orientations (default: UP).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Output run name, nested under outputs/<algo>/<dim>/. Defaults to "
            "{timestamp}_seed{seed}_steps{steps}, unique per run so "
            "reruns never overwrite a previous run's outputs. Pass this "
            "explicitly for a fixed, reusable folder name instead."
        ),
    )
    parser.add_argument("--no-gui", action="store_true", help="Run headless (save GIF but do not open a window).")
    parser.add_argument(
        "--max-outer-iters",
        type=int,
        default=IrisPlanner.MAX_OUTER_ITERS,
        help=f"IRIS outer iteration cap (default: {IrisPlanner.MAX_OUTER_ITERS}).",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=IrisPlanner.MAX_EXPANSIONS,
        help=f"IRIS search expansion budget (default: {IrisPlanner.MAX_EXPANSIONS}).",
    )
    parser.add_argument(
        "--p0",
        type=float,
        default=None,
        help=(
            "IRIS approximate dominance parameter p in (0,1]. "
            "Use p0=1 for strict (coverage superset) dominance."
        ),
    )
    parser.add_argument(
        "--epsilon0",
        type=float,
        default=None,
        help=(
            "IRIS approximate dominance parameter epsilon >= 0. "
            "Use epsilon0=0 for strict (cost) dominance."
        ),
    )
    parser.add_argument(
        "--retract",
        dest="allow_retract",
        action="store_true",
        help=(
            "IRIS: enable retract actions (and the trail-dominance check they "
            "require for sound pruning). Off by default -- can be dramatically "
            "slower on maps with real coverage-seeking search."
        ),
    )
    parser.add_argument(
        "--algo",
        choices=("iris", "random", "dp"),
        default="iris",
        help="Which planner to run (default: iris).",
    )
    parser.add_argument(
        "--eps-len", type=float, default=0.5, help="DP: epsilon factor for path length (default: 0.5)."
    )
    parser.add_argument(
        "--eps-cov", type=float, default=0.5, help="DP: epsilon factor for coverage (default: 0.5)."
    )
    parser.add_argument(
        "--eps-dam", type=float, default=0.5, help="DP: epsilon factor for damage (default: 0.5)."
    )
    parser.add_argument(
        "--eps-ndose", type=float, default=0.5, help="DP: epsilon factor for dosage count (default: 0.5)."
    )
    parser.add_argument(
        "--dosage-levels",
        type=str,
        default=",".join(str(d) for d in DOSAGE_LEVELS),
        help=f"DP: comma-separated dosage levels to consider (default: {DOSAGE_LEVELS}).",
    )
    parser.add_argument(
        "--max-plans-per-node",
        type=_max_plans_per_node_arg,
        default=MAX_PLANS_PER_NODE,
        help=(
            f"DP: hard cap on plans kept per node after compression (default: {MAX_PLANS_PER_NODE}). "
            'Pass "none" for an uncapped frontier.'
        ),
    )
    parser.add_argument(
        "--depth-scaled-eps",
        action="store_true",
        help=(
            "DP: treat --eps-* as a GLOBAL target and shrink it per node by depth "
            "(eps' = (1+eps)^(1/depth)-1) so error does not compound to (1+eps)^depth. "
            "Tighter frontier, slower; clean end-to-end guarantee only when the cap "
            "is not binding. Default off."
        ),
    )
    parser.add_argument(
        "--no-reorder-children",
        action="store_true",
        help=(
            "DP: disable merging each node's children strongest-first (default: "
            "enabled). Reordering is a pure scheduling optimization -- it changes "
            "how much transient insert/absorb work the compression step does, not "
            "the eps-approximation guarantee -- but since eps-dominance isn't "
            "transitive, it can converge to a different (equally eps-valid) "
            "frontier than the original graph-adjacency order. Pass this flag for "
            "an apples-to-apples comparison against the pre-reordering behavior."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    timer = RunTimer()

    outputs_root = Path("outputs")
    outputs_root.mkdir(exist_ok=True)

    timestamp_compact = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        str(args.run_name)
        if args.run_name is not None
        # Timestamped by default so reruns (even with identical params) never
        # silently overwrite a previous run's outputs -- pass --run-name
        # explicitly for a deterministic/reusable folder name instead.
        else f"{timestamp_compact}_seed{int(args.seed)}_steps{int(args.steps)}"
    )
    out_dir = _prepare_run_dir(outputs_root=outputs_root, algo_name=args.algo, dim=DIM_2D, run_name=run_name)

    log_path = configure_run_logging(out_dir)
    print(f"Logging to {log_path}")

    # Defaults to the origin facing UP (matches the map editor's centered canvas).
    start_x, start_y = float(args.start_x), float(args.start_y)
    start_orientation = Orientation[args.start_orientation]

    try:
        with timer.section("world.build"):
            world = build_demo_world(
                start_x=start_x,
                start_y=start_y,
                steps=int(args.steps),
                step_size=float(args.step_size),
            )

        with timer.section("map.render"):
            map_img_path = out_dir / "map.png"
            fig, ax = plt.subplots(figsize=(7, 7))
            world.render(
                ax=ax,
                clear=True,
                show_robot=False,
                show_trail=False,
                show_goal=False,
                show_healthy_overlay=True,
            )
            plt.tight_layout()
            fig.savefig(map_img_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved demo map to {map_img_path}")

        with timer.section("graph.generate"):
            nodes, edges, distances = generate_paths_graph(
                steps=int(args.steps),
                step_size=float(args.step_size),
                start_x=start_x,
                start_y=start_y,
                start_orientation=start_orientation,
                world=world,
            )

        if not nodes:
            raise RuntimeError(
                f"No roadmap generated: start position ({start_x}, {start_y}) is out of "
                f"bounds or in collision (world.x_limits={world.x_limits}, "
                f"world.y_limits={world.y_limits}, robot.radius={world.robot.radius}). "
                "Move the start marker (or the obstacle it's inside) and try again."
            )

        if int(args.steps) <= 5:
            with timer.section("graph.plot"):
                graph_img_path = out_dir / "paths_graph.png"
                plot_paths_graph(
                    nodes=nodes,
                    edges=edges,
                    distances=distances,
                    step_size=float(args.step_size),
                    start_x=start_x,
                    start_y=start_y,
                    start_orientation=start_orientation,
                    steps=int(args.steps),
                    output_path=graph_img_path,
                    no_gui=bool(args.no_gui),
                    world=world,
                )
                print(f"Saved paths graph to {graph_img_path} (nodes={len(nodes)}, edges={len(edges)})")
        else:
            print(f"Skipping paths graph image (steps > 5). (nodes={len(nodes)}, edges={len(edges)})")

        start_key = (round(float(start_x), 4), round(float(start_y), 4), int(start_orientation))

        index_row: dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": run_name,
            "algo": args.algo,
            "dim": DIM_2D,
            "seed": int(args.seed),
            "steps": int(args.steps),
            "step_size": float(args.step_size),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        }

        if args.algo == "random":
            result = run_random_alg(
                seed=int(args.seed),
                num_steps=int(args.steps),
                step_size=float(args.step_size),
                start_key=start_key,
                nodes=nodes,
                edges=edges,
                world=world,
                out_dir=out_dir,
                no_gui=bool(args.no_gui),
            )
        elif args.algo == "dp":
            dosage_levels = tuple(int(x) for x in str(args.dosage_levels).split(",") if x.strip() != "")
            eps_vec = EpsVec(
                length_tol=float(args.eps_len),
                coverage_tol=float(args.eps_cov),
                damage_tol=float(args.eps_dam),
                dose_count_tol=float(args.eps_ndose),
            )
            index_row.update(
                eps_len=eps_vec.length_tol,
                eps_cov=eps_vec.coverage_tol,
                eps_dam=eps_vec.damage_tol,
                eps_ndose=eps_vec.dose_count_tol,
                dosage_levels=",".join(str(d) for d in dosage_levels),
                max_plans_per_node=(
                    "uncapped" if args.max_plans_per_node is None else int(args.max_plans_per_node)
                ),
            )
            result = run_dp(
                seed=int(args.seed),
                step_size=float(args.step_size),
                start_key=start_key,
                nodes=nodes,
                edges=edges,
                world=world,
                out_dir=out_dir,
                no_gui=bool(args.no_gui),
                dosage_levels=dosage_levels,
                eps_vec=eps_vec,
                max_plans_per_node=args.max_plans_per_node,
                depth_scaled_eps=bool(args.depth_scaled_eps),
                reorder_children=not bool(args.no_reorder_children),
                timer=timer,
            )
        else:
            index_row.update(
                max_outer_iters=int(args.max_outer_iters),
                max_expansions=int(args.max_expansions),
                p0=("" if args.p0 is None else float(args.p0)),
                epsilon0=("" if args.epsilon0 is None else float(args.epsilon0)),
            )
            result = run_iris(
                seed=int(args.seed),
                step_size=float(args.step_size),
                start_key=start_key,
                nodes=nodes,
                edges=edges,
                world=world,
                out_dir=out_dir,
                no_gui=bool(args.no_gui),
                max_outer_iters=int(args.max_outer_iters),
                max_expansions=int(args.max_expansions),
                p0=(None if args.p0 is None else float(args.p0)),
                epsilon0=(None if args.epsilon0 is None else float(args.epsilon0)),
                allow_retract=bool(args.allow_retract),
            )

        index_row.update(result)
        index_path = _append_run_index(outputs_root, index_row)
        print(f"Run outputs saved under {out_dir} ; index updated at {index_path}")
    finally:
        timer.log_summary(logger)
        log_iris_planner_metrics(logger)


if __name__ == "__main__":
    main()

