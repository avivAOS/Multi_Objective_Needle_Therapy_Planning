"""
Random planner: random-walk the paths graph and translate visited nodes into the
2D replay format (`RandomPath`). Shared demo plumbing lives in `main_demo.py`.

`PathStep`/`RandomPath` are the shared 2D replay representation for all three
planners (random/IRIS/DP) -- only `random_path_from_paths_graph` itself is
random-specific; IRIS's and DP's own graph-path-to-`RandomPath` converters
live inline in `main_demo.py` as `_iris_randompath_from_graph_path`/
`_dp_randompath_from_graph_path`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from sim2d import Orientation
from sim2d.paths_graph import make_inject_coverage_fn
from graph_utils import graph_from_edges
from random_path_on_graph import generate_random_path_on_graph

logger = logging.getLogger(__name__)


@dataclass
class PathStep:
    step_index: int
    action: str  # "move" | "retract" | "inject"
    from_orientation: str
    to_orientation: str
    x: float
    y: float
    theta: float
    is_inject: bool = False
    inject_covered_points: Optional[List[List[float]]] = None
    inject_covered_points_cumulative: Optional[List[List[float]]] = None
    # Healthy-tissue (HPOI) damage points, populated for dosage-aware (DP)
    # inject steps; None for legacy binary-inject (IRIS/random) steps.
    inject_damaged_points: Optional[List[List[float]]] = None
    inject_damaged_points_cumulative: Optional[List[List[float]]] = None
    # Physical radius of the injection disk (dosage-aware DP steps only), so the
    # visualization can draw the true affected area; None for legacy steps.
    inject_radius: Optional[float] = None


@dataclass
class RandomPath:
    seed: int
    num_steps: int
    start_x: float
    start_y: float
    start_orientation: str
    step_size: float
    steps: List[PathStep]


def random_path_from_paths_graph(
    *,
    seed: int,
    num_steps: int,
    step_size: float,
    start_key: tuple[float, float, int],
    nodes: dict[tuple[float, float, int], dict],
    edges: set[tuple[tuple[float, float, int], tuple[float, float, int]]],
) -> RandomPath:
    """
    Generate a path by random-walking the existing paths_graph, then translate
    visited nodes back into 2D (x, y, theta) steps for replay.
    """
    g = graph_from_edges(nodes=nodes.keys(), edges=edges, directed=True)
    # We need a "post-action" node for each step so that retract steps can
    # correctly land on the previous pose+orientation. `generate_random_path_on_graph`
    # records the node *before* applying the step update, so we request one extra
    # step and use consecutive pairs (pre -> post).
    walk = generate_random_path_on_graph(
        graph=g,
        start_node=start_key,
        num_steps=int(num_steps) + 1,
        seed=int(seed),
        retract_probability=0.30,
        avoid_immediate_backtrack=True,
        dead_end_policy="terminate",
        coverage_fn=make_inject_coverage_fn(nodes),
    )

    if len(walk.steps) < 2:
        raise RuntimeError("Graph walk produced no steps.")
    if len(walk.steps) < int(num_steps) + 1:
        raise RuntimeError(
            f"Graph walk terminated early (requested {int(num_steps)+1} pre-steps, got {len(walk.steps)})."
        )

    first_node = nodes[start_key]
    start_x = float(first_node["x"])
    start_y = float(first_node["y"])
    start_orientation: Orientation = first_node["ori"]

    steps: List[PathStep] = []
    cumulative_covered: set[tuple[float, float]] = set()
    for i in range(int(num_steps)):
        pre = walk.steps[i]
        post = walk.steps[i + 1]

        pre_key = pre.node
        post_key = post.node
        action = str(getattr(pre, "action", "move"))

        pre_node = nodes[pre_key]
        post_node = nodes[post_key]
        from_ori: Orientation = pre_node["ori"]
        to_ori: Orientation = post_node["ori"]
        theta = to_ori.to_angle()

        inject_covered_points: Optional[List[List[float]]] = None
        inject_covered_points_cumulative: Optional[List[List[float]]] = None
        if action == "inject":
            # Coverage is stored on the graph nodes (computed during graph generation).
            covered_now = set(make_inject_coverage_fn(nodes)(pre_key) or ())
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

