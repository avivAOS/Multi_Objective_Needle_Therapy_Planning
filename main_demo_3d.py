"""
3D demo runner for the steerable-needle planner.

Builds a 3D world (wall obstacles + ball tumors), generates a lattice paths
graph, runs the (dimension-agnostic) IRIS graph planner on it, and renders the
returned best path as a GIF of the needle advancing through the volume and
healing tumors as it injects.

The graph search itself (`iris_on_graph.generate_iris_plan_on_graph`) is shared
with the 2D demo unchanged: it only needs a `Graph[NodeId]` plus a
`coverage_fn(node) -> Iterable[POI]`, so 3D simply supplies (x,y,z,ori) node
keys and (x,y,z) voxel POIs.

Examples:
    python main_demo_3d.py --no-gui
    python main_demo_3d.py --steps 11 --step-size 0.8 --no-gui
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import logging as _logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from graph_utils import Graph as PlannerGraph
from graph_utils import build_adjacency
from iris_on_graph import IrisPlanner, generate_iris_plan_on_graph, log_iris_planner_metrics
from random_path_on_graph import generate_random_path_on_graph
from pareto_dominance import EpsVec, Plan
from dp_on_graph import (
    MAX_PLANS_PER_NODE,
    DPStats,
    bfs_depth,
    build_dosage_masks,
    compute_dp,
    reconstruct_graph_path,
)
from main_demo import (
    DIM_3D,
    _append_run_index,
    _draw_frontier_3d,
    _draw_parallel_coords,
    _prepare_run_dir,
    _select_frontier_plans,
    save_frontier_snapshot,
)
from sim2d.run_logging import RunTimer, configure_run_logging

from sim3d import (
    DOSAGE_LEVELS,
    World3D,
    configure_demo_map_3d,
    direction_from_vector,
    generate_paths_graph_3d,
    make_dosage_fn_3d,
    make_inject_coverage_fn,
    render_world_3d,
)
from sim3d.path3d import (
    apply_step,
    init_world_for_path,
    path3d_from_iris_graph_path,
)

logger = logging.getLogger("main_demo_3d")


def build_demo_world() -> World3D:
    world = World3D()
    return world


def render_gif(
    *,
    world: World3D,
    path,
    out_dir: Path,
    no_gui: bool,
    pause: float = 0.4,
    rotate: bool = True,
    algo_label: str = "IRIS",
) -> None:
    import matplotlib.pyplot as plt

    if no_gui:
        matplotlib.use("Agg", force=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if not no_gui:
        plt.ion()

    init_world_for_path(world, path)
    frames = []
    n = len(path.steps)
    base_azim = -60.0

    def capture(step_idx: int, label: str) -> None:
        render_world_3d(ax, world, clear=True)
        if rotate and n > 0:
            ax.view_init(elev=22.0, azim=base_azim + 120.0 * (step_idx / max(1, n)))
        else:
            ax.view_init(elev=22.0, azim=base_azim)
        ax.set_title(f"{algo_label} 3D | {label}")
        fig.canvas.draw()
        if out_dir is not None:
            canvas = fig.canvas
            w, h = canvas.get_width_height()
            argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
            frames.append(argb[:, :, 1:4].copy())
        if not no_gui:
            plt.pause(pause)

    capture(0, f"start  (0/{n})")
    total_healed = 0
    for step in path.steps:
        healed = apply_step(world, step)
        if healed:
            total_healed += healed
        label = f"step {step.step_index + 1}/{n} | {step.action}"
        if step.is_inject:
            label += f" | healed +{healed or 0}"
        capture(step.step_index + 1, label)

    if frames:
        gif_path = out_dir / "simulation_3d.gif"
        try:
            import imageio.v3 as iio

            iio.imwrite(gif_path, frames, duration=pause or 0.5, loop=0)
            print(f"Saved GIF to {gif_path}")
        except ModuleNotFoundError:
            print(f"Warning: imageio not installed; skipping GIF ({gif_path}).")

    if not no_gui:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    # Coverage summary. Reported against the POI universe (coverable tumor
    # voxels) so it matches the planner's "POIs covered=.. (of N)" exactly.
    healed_total = int(np.count_nonzero(world.healed_mask)) if world.healed_mask is not None else 0
    logger.info(
        "3D run finished: POIs healed=%d/%d", healed_total, path.total_pois
    )


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a planner in a 3D needle environment.")
    p.add_argument(
        "--algo",
        choices=("iris", "random", "dp"),
        default="iris",
        help="Which planner to run (default: iris).",
    )
    p.add_argument("--steps", type=int, default=12, help="BFS depth for the paths graph.")
    p.add_argument("--step-size", type=float, default=1.0, help="Lattice step size (1.0 keeps integer coords).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-gui", action="store_true", help="Headless: save GIF, no window.")
    p.add_argument("--max-outer-iters", type=int, default=IrisPlanner.MAX_OUTER_ITERS)
    p.add_argument("--max-expansions", type=int, default=1000)
    p.add_argument("--p0", type=float, default=None)
    p.add_argument("--epsilon0", type=float, default=None)
    p.add_argument(
        "--retract",
        dest="allow_retract",
        action="store_true",
        help="Enable retract actions in IRIS (off by default; can be very slow).",
    )
    p.add_argument("--no-rotate", action="store_true", help="Disable camera rotation in the GIF.")
    p.add_argument("--max-injections", type=int, default=5, help="Max inject actions per IRIS run.")
    p.add_argument("--eps-len", type=float, default=0.5, help="DP: epsilon factor for path length.")
    p.add_argument("--eps-cov", type=float, default=0.5, help="DP: epsilon factor for coverage.")
    p.add_argument("--eps-dam", type=float, default=0.5, help="DP: epsilon factor for damage.")
    p.add_argument("--eps-ndose", type=float, default=0.5, help="DP: epsilon factor for dosage count.")
    p.add_argument(
        "--max-plans-per-node",
        type=int,
        default=MAX_PLANS_PER_NODE,
        help=f"DP: hard cap on plans kept per node after compression (default: {MAX_PLANS_PER_NODE}).",
    )
    p.add_argument(
        "--no-reorder-children",
        action="store_true",
        help=(
            "DP: disable merging each node's children strongest-first (default: "
            "enabled) -- see main_demo.py's --no-reorder-children for the full "
            "explanation."
        ),
    )
    p.add_argument("--start-x", type=float, default=None, help="Override start x (default: 0.0).")
    p.add_argument("--start-y", type=float, default=None, help="Override start y (default: 0.0).")
    p.add_argument("--start-z", type=float, default=None, help="Override start z (default: 0.0).")
    p.add_argument(
        "--start-orientation",
        type=int,
        default=None,
        help="Override start orientation: an index into DIRECTIONS_3D (0-25; "
        "default: 25, direction (1,1,1)).",
    )
    return p.parse_args(argv)


START_POSE = (0, 0, 0, direction_from_vector((1.0, 1.0, 1.0)))


def build_world_and_path(
    *,
    algo: str = "iris",
    steps: int = 12,
    step_size: float = 1.0,
    seed: int = 0,
    max_outer_iters: int = IrisPlanner.MAX_OUTER_ITERS,
    max_expansions: int = 1000,
    max_injections: int = 5,
    p0: Optional[float] = None,
    epsilon0: Optional[float] = None,
    allow_retract: bool = False,
    eps_len: float = 0.5,
    eps_cov: float = 0.5,
    eps_dam: float = 0.5,
    eps_ndose: float = 0.5,
    max_plans_per_node: int = MAX_PLANS_PER_NODE,
    start_pose: Optional[Tuple[float, float, float, int]] = None,
    reorder_children: bool = True,
    timer: Optional[RunTimer] = None,
    dp_stats: Optional[DPStats] = None,
):
    """
    Build the demo world, generate the paths graph, run a planner, and return
    (fresh_world, Path3D, dp_frontier). The returned world is freshly configured
    (no trail/healing applied yet) so callers can replay the path into it.
    `dp_frontier` is `None` for iris/random; for `algo="dp"` it's
    `{"plans": [...], "start_key": ..., "nodes": ...}` -- the full Pareto
    frontier, for callers that want the multi-solution dashboard instead of
    just the single chosen path.

    `timer`/`dp_stats`, if given, are populated exactly like `main_demo.py`'s
    `run_dp` (per-phase timing, per-node comparisons/frontier sizes, peak
    memory) -- unused by callers that don't care (e.g. `view_3d.py`, which
    never runs `algo="dp"` anyway).
    """
    start_x, start_y, start_z, start_ori = START_POSE if start_pose is None else start_pose

    timer = timer if timer is not None else RunTimer()

    world = build_demo_world()
    configure_demo_map_3d(world, start_x=start_x, start_y=start_y, start_z=start_z)

    logger.info("Generating 3D paths graph (steps=%d, step_size=%.2f)...", steps, step_size)
    print(f"Generating 3D paths graph (steps={steps}, step_size={step_size:.2f})...", flush=True)
    t0_graph = time.perf_counter()
    # Each node costs its own grid scan per representation -- skip whichever
    # this algo won't read (DP never reads inject_coverage; iris/random never
    # read dosage_coverage/dosage_damage).
    with timer.section("graph.generate"):
        nodes, edges, _depth = generate_paths_graph_3d(
            steps=int(steps),
            step_size=float(step_size),
            start_x=start_x,
            start_y=start_y,
            start_z=start_z,
            progress=True,
            start_orientation=start_ori,
            world=world,
            compute_dosage=(algo == "dp"),
            compute_inject_coverage=(algo != "dp"),
        )
    t_graph = time.perf_counter() - t0_graph
    print(f"Graph: nodes={len(nodes)} edges={len(edges)}", flush=True)
    if not nodes:
        raise SystemExit("Start pose is in collision; no graph generated.")

    start_key = (round(start_x, 4), round(start_y, 4), round(start_z, 4), start_ori)

    coverage_fn = make_inject_coverage_fn(nodes)

    t0_plan = time.perf_counter()
    dp_frontier: Optional[dict] = None
    if algo == "random":
        logger.info("Running random-walk planner...")
        graph_path = generate_random_path_on_graph(
            graph=PlannerGraph(
                nodes=tuple(nodes.keys()),
                adjacency=build_adjacency(edges, directed=True),
                directed=True,
            ),
            start_node=start_key,
            num_steps=int(steps),
            seed=int(seed),
            coverage_fn=coverage_fn,
        )
    elif algo == "dp":
        logger.info("Running DP planner...")
        dp_stats = dp_stats if dp_stats is not None else DPStats()
        graph = PlannerGraph(
            nodes=tuple(nodes.keys()),
            adjacency=build_adjacency(edges, directed=True),
            directed=True,
        )
        dosage_fn = make_dosage_fn_3d(nodes)
        depth_map = bfs_depth(graph, start_key)
        with timer.section("dp.masks_build"):
            coverage_masks, damage_masks = build_dosage_masks(
                graph=graph, dosage_fn=dosage_fn, dosage_levels=DOSAGE_LEVELS
            )
        eps_vec = EpsVec(
            length_tol=float(eps_len),
            coverage_tol=float(eps_cov),
            damage_tol=float(eps_dam),
            dose_count_tol=float(eps_ndose),
        )
        tracemalloc.start()
        with timer.section("dp.compute"):
            memo = compute_dp(
                graph=graph,
                start_node=start_key,
                depth=depth_map,
                dosage_levels=DOSAGE_LEVELS,
                coverage_masks=coverage_masks,
                damage_masks=damage_masks,
                eps_vec=eps_vec,
                max_plans_per_node=max_plans_per_node,
                progress=True,
                reorder_children=bool(reorder_children),
                stats=dp_stats,
            )
        _current_bytes, dp_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dp_stats.peak_memory_bytes = int(dp_peak_bytes)
        plans = memo[start_key]
        print(
            f"DP stats: {len(plans)} plans at start, "
            f"{dp_stats.total_comparisons()} comparisons, "
            f"peak frontier {dp_stats.peak_frontier_size()}, "
            f"peak memory {round(dp_peak_bytes / (1024 * 1024), 3)}MB",
            flush=True,
        )
        # Use Min Damage (the true min-damage treated plan) as the primary
        # output plan, mirroring the 2D demo's choice (main_demo.py's run_dp).
        with timer.section("dp.select_frontier"):
            selections, _treated_pool = _select_frontier_plans(plans)
            chosen_plan = next(p for name, p, _c in selections if name == "Min Damage")
            graph_path = reconstruct_graph_path(chosen_plan, start_node=start_key, seed=int(seed))
        dp_frontier = {"plans": plans, "start_key": start_key, "nodes": nodes}
    else:
        logger.info("Running IRIS planner...")
        adj = build_adjacency(edges, directed=True)
        graph = PlannerGraph(nodes=tuple(nodes.keys()), adjacency=adj, directed=True)
        graph_path = generate_iris_plan_on_graph(
            graph=graph,
            start_node=start_key,
            seed=int(seed),
            coverage_fn=coverage_fn,
            p0=p0,
            epsilon0=epsilon0,
            max_outer_iters=int(max_outer_iters),
            max_expansions=int(max_expansions),
            max_injections_per_run=int(max_injections),
            allow_retract=bool(allow_retract),
        )
    t_plan = time.perf_counter() - t0_plan

    path = path3d_from_iris_graph_path(
        graph_path=graph_path,
        seed=int(seed),
        step_size=float(step_size),
        nodes=nodes,
    )
    print(f"{algo} best path: {path.num_steps} steps")
    logger.info(
        "Timing summary — graph generation: %.2fs | %s planning: %.2fs | total: %.2fs",
        t_graph, algo, t_plan, t_graph + t_plan,
    )
    return world, path, dp_frontier


def _replay_world3d_to(world: World3D, path: object, idx: int):
    """
    Replay `path` from scratch up to step `idx` (-1 = start pose only), so it's
    correct no matter how the animation jumps between frames. Returns
    (last_step, cumulative healed voxels, cumulative damaged voxels).
    """
    if world.healed_mask is not None:
        world.healed_mask[:] = False
    if world.damaged_mask is not None:
        world.damaged_mask[:] = False

    init_world_for_path(world, path)
    last_step = None
    for s in path.steps[: idx + 1]:
        apply_step(world, s)
        last_step = s

    healed = int(world.healed_mask.sum()) if world.healed_mask is not None else 0
    damaged = int(world.damaged_mask.sum()) if world.damaged_mask is not None else 0
    return last_step, healed, damaged


def generate_pareto_dashboard_3d(
    plans: List[Plan],
    start_key,
    nodes: dict,
    world: World3D,
    *,
    seed: int = 0,
    step_size: float = 1.0,
    n_frames: int = 30,
    interval_ms: int = 200,
) -> Tuple["plt.Figure", FuncAnimation]:
    """
    3D analog of `main_demo.generate_pareto_dashboard`. Top row: three live
    3D sim panels (Max Coverage / Min Damage / Min Length), each replaying its
    plan in its own World3D copy. Bottom row: the same (dimension-agnostic)
    4D-frontier plots used by the 2D dashboard, since they operate purely on
    `Plan` objects. Returns `(fig, ani)` -- the caller must keep `ani` alive or
    playback is garbage-collected.

    3D voxel scatter re-rendering is heavier per frame than 2D's patch-based
    render, so this defaults to fewer frames and a slower interval than the 2D
    dashboard.
    """
    if not plans:
        raise ValueError("generate_pareto_dashboard_3d requires at least one plan")

    total_ppois = int(world.tumor_mask.sum()) if world.tumor_mask is not None else max(
        (p.coverage.bit_count() for p in plans), default=0
    )

    selections, _treated_pool = _select_frontier_plans(plans)

    # --- Build one independent (Path3D, World3D) replay per selected plan ---
    sim_panels = []  # (name, plan, color, Path3D, World3D)
    for name, plan, color in selections:
        graph_path = reconstruct_graph_path(plan, start_node=start_key, seed=int(seed))
        p3d = path3d_from_iris_graph_path(
            graph_path=graph_path, seed=int(seed), step_size=float(step_size), nodes=nodes
        )
        panel_world = copy.deepcopy(world)
        sim_panels.append((name, plan, color, p3d, panel_world))

    # Layout: top row = 3 animated 3D sims; bottom row = parallel coords + 3D frontier.
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0])
    map_axes = [fig.add_subplot(gs[0, c], projection="3d") for c in range(3)]

    # --- Bottom row: the 4D frontier (static, drawn once) ---
    ax_pc = fig.add_subplot(gs[1, 0])
    _draw_parallel_coords(ax_pc, plans, selections)
    ax_3d = fig.add_subplot(gs[1, 1:], projection="3d")
    _draw_frontier_3d(ax_3d, plans, selections)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)

    n_frames = max(int(n_frames), 1)
    hold_frames = 6
    total_frames = n_frames + hold_frames

    def update(frame_idx: int):
        frac = min(frame_idx, n_frames - 1) / max(n_frames - 1, 1)
        for (name, plan, color, p3d, panel_world), ax in zip(sim_panels, map_axes):
            n_steps = len(p3d.steps)
            idx = int(round(frac * (n_steps - 1))) if n_steps > 0 else -1
            last_step, healed, damaged = _replay_world3d_to(panel_world, p3d, idx)

            # render_world_3d always resets the view to the full world extent
            # (ax.cla() + set_axes_equal) -- save/restore the axis limits so
            # any zoom survives every animation tick (elev/azim already do,
            # since cla() doesn't reset them in this matplotlib version).
            if frame_idx > 0:
                xlim3d, ylim3d, zlim3d = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
            render_world_3d(ax, panel_world, clear=True)
            if frame_idx > 0:
                ax.set_xlim3d(xlim3d)
                ax.set_ylim3d(ylim3d)
                ax.set_zlim3d(zlim3d)

            action = last_step.action if last_step is not None else "start"
            shown = (idx + 1) if n_steps > 0 else 0
            ax.set_title(
                f"{name}  cov={plan.coverage.bit_count()}/{total_ppois} "
                f"dam={plan.damage.bit_count()} doses={plan.dose_count} len={plan.length:.0f}\n"
                f"step {shown}/{n_steps} [{action}]  healed={healed}  damaged={damaged}",
                fontsize=9, color=color,
            )
        return []

    ani = FuncAnimation(
        fig, update, frames=total_frames, interval=interval_ms, blit=False, repeat=True
    )
    return fig, ani


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    outputs_root = Path("outputs")
    timestamp_compact = time.strftime("%Y%m%d-%H%M%S")
    # Timestamped by default so reruns (even with identical params) never
    # silently overwrite a previous run's outputs -- pass --run-name
    # explicitly for a deterministic/reusable folder name instead.
    run_name = args.run_name or f"{timestamp_compact}_seed{args.seed}_steps{args.steps}"
    out_dir = _prepare_run_dir(outputs_root=outputs_root, algo_name=str(args.algo), dim=DIM_3D, run_name=run_name)

    # Logs go to the run.log file only; keep the terminal clean (console handler
    # is raised above CRITICAL so no log records reach stdout/stderr).
    log_path = configure_run_logging(out_dir, console_level=_logging.CRITICAL + 1)
    print(f"Logging to {log_path}")

    start_pose = None
    if (
        args.start_x is not None
        or args.start_y is not None
        or args.start_z is not None
        or args.start_orientation is not None
    ):
        sx = 0.0 if args.start_x is None else float(args.start_x)
        sy = 0.0 if args.start_y is None else float(args.start_y)
        sz = 0.0 if args.start_z is None else float(args.start_z)
        ori = (
            direction_from_vector((1.0, 1.0, 1.0))
            if args.start_orientation is None
            else int(args.start_orientation)
        )
        start_pose = (sx, sy, sz, ori)

    timer = RunTimer()
    dp_stats = DPStats()
    world, path, dp_frontier = build_world_and_path(
        algo=str(args.algo),
        steps=int(args.steps),
        step_size=float(args.step_size),
        seed=int(args.seed),
        max_outer_iters=int(args.max_outer_iters),
        max_expansions=int(args.max_expansions),
        max_injections=int(args.max_injections),
        p0=args.p0,
        epsilon0=args.epsilon0,
        allow_retract=bool(args.allow_retract),
        eps_len=float(args.eps_len),
        eps_cov=float(args.eps_cov),
        eps_dam=float(args.eps_dam),
        eps_ndose=float(args.eps_ndose),
        max_plans_per_node=int(args.max_plans_per_node),
        start_pose=start_pose,
        reorder_children=not bool(args.no_reorder_children),
        timer=timer,
        dp_stats=dp_stats,
    )

    # Persist the path so the interactive viewer can reload it.
    path_json = out_dir / "path.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(asdict(path), f, indent=2)

    if dp_frontier is not None:
        # The multi-solution Pareto dashboard supersedes the single-path
        # viewer/GIF for DP (mirrors main_demo.py's run_dp): simulation_3d.gif
        # is still written to disk headlessly, but the dashboard -- showing
        # all three picks plus the frontier -- is what's actually displayed.
        original_backend = matplotlib.get_backend()
        with timer.section("dp.replay"):
            render_gif(
                world=world,
                path=path,
                out_dir=out_dir,
                no_gui=True,
                rotate=not bool(args.no_rotate),
                algo_label=str(args.algo).upper(),
            )
        if not args.no_gui:
            matplotlib.use(original_backend, force=True)

        frontier_path = out_dir / "frontier_3d.pkl"
        save_frontier_snapshot(
            frontier_path, kind="3d", plans=dp_frontier["plans"],
            start_key=dp_frontier["start_key"], nodes=dp_frontier["nodes"], world=world,
            seed=int(args.seed), step_size=float(args.step_size),
        )
        print(
            f"Saved frontier to {frontier_path} -- reopen it anytime with "
            f"`python replay_dashboard.py {frontier_path}` (no recomputation needed).",
            flush=True,
        )

        print("Rendering Pareto dashboard (this can take a while)...", flush=True)
        with timer.section("dp.dashboard_render"):
            dashboard_fig, dashboard_ani = generate_pareto_dashboard_3d(
                dp_frontier["plans"], dp_frontier["start_key"], dp_frontier["nodes"], world,
                seed=int(args.seed), step_size=float(args.step_size),
            )
            dashboard_path = out_dir / "dashboard_3d.gif"
            dashboard_ani.save(dashboard_path, writer="pillow")
        print(f"Saved dashboard to {dashboard_path}", flush=True)
        if not args.no_gui:
            print("Showing dashboard window (close it to finish).", flush=True)
            plt.show(block=True)
        plt.close(dashboard_fig)

        # Written last, once every timer.section(...) has actually run --
        # mirrors main_demo.py's run_dp exactly (same DPStats/RunTimer shape).
        dp_stats_path = out_dir / "dp_stats.json"
        dp_stats_summary = {
            "algorithm": "dp",
            "graph": {"num_nodes": len(dp_frontier["nodes"]), "num_edges": None},
            "eps_vec": {
                "length_tol": float(args.eps_len),
                "coverage_tol": float(args.eps_cov),
                "damage_tol": float(args.eps_dam),
                "dose_count_tol": float(args.eps_ndose),
            },
            "max_plans_per_node": int(args.max_plans_per_node),
            "timing_seconds": timer.durations(),
            "frontier_size_at_start": len(dp_frontier["plans"]),
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
    else:
        if not args.no_gui:
            # Auto-launch the scrubbable viewer alongside the (fast, auto-playing)
            # GIF-capture window below, so there's an easy way to step through the
            # run at your own pace instead of just watching it fly by once.
            view_3d_path = Path(__file__).resolve().parent / "view_3d.py"
            subprocess.Popen([sys.executable, str(view_3d_path), "--path", str(path_json)])

        render_gif(
            world=world,
            path=path,
            out_dir=out_dir,
            no_gui=bool(args.no_gui),
            rotate=not bool(args.no_rotate),
            algo_label=str(args.algo).upper(),
        )
    if args.algo == "iris":
        log_iris_planner_metrics(logger)

    # Shared with the 2D runner's outputs/runs_index.csv, distinguished by
    # "dim" -- not a separate CSV. Frontier-derived columns (num_nodes,
    # chosen_*, poi_*, ...) are 2D-only for now since build_world_and_path
    # doesn't return that data; they're just left blank here (_append_run_index
    # already handles missing columns this way for algo-inapplicable fields).
    index_row: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "algo": args.algo,
        "dim": DIM_3D,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "step_size": float(args.step_size),
        "output_dir": str(out_dir),
    }
    if args.algo == "dp":
        index_row.update(
            eps_len=float(args.eps_len),
            eps_cov=float(args.eps_cov),
            eps_dam=float(args.eps_dam),
            eps_ndose=float(args.eps_ndose),
            max_plans_per_node=int(args.max_plans_per_node),
            dp_time_seconds=round(timer.durations().get("dp.compute", 0.0), 4),
            dp_total_comparisons=dp_stats.total_comparisons(),
            dp_peak_frontier_size=dp_stats.peak_frontier_size(),
            dp_peak_memory_mb=(
                round(dp_stats.peak_memory_bytes / (1024 * 1024), 3)
                if dp_stats.peak_memory_bytes is not None
                else ""
            ),
            dp_reordering_enabled=not bool(args.no_reorder_children),
        )
    elif args.algo == "iris":
        index_row.update(
            max_outer_iters=int(args.max_outer_iters),
            max_expansions=int(args.max_expansions),
            p0=("" if args.p0 is None else float(args.p0)),
            epsilon0=("" if args.epsilon0 is None else float(args.epsilon0)),
        )
    index_path = _append_run_index(outputs_root, index_row)
    print(f"Run outputs saved under {out_dir} ; index updated at {index_path}")


if __name__ == "__main__":
    main()
