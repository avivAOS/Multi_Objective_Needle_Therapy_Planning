from __future__ import annotations

import logging
import random
from typing import Callable, Optional, Sequence, Set, Tuple

from graph_utils import Graph, GraphPath, GraphPathStep, NodeId

logger = logging.getLogger(__name__)


def _fmt_seq(items: Sequence[NodeId], *, max_items: int = 12) -> str:
    if not items:
        return "[]"
    if len(items) <= int(max_items):
        return "[" + ", ".join(repr(x) for x in items) + "]"
    head = items[: int(max_items)]
    return "[" + ", ".join(repr(x) for x in head) + f", ... +{len(items) - int(max_items)}" + "]"


def format_covered_points(pts: Set[object], *, max_items: int = 8) -> str:
    """Format a covered-POI set for logging. POIs are opaque (2D `(x,y)` floats,
    3D `(ix,iy,iz)` voxel indices, ...) -- use `repr`, not a hardcoded 2-tuple
    unpack, so this works regardless of dimensionality."""
    if not pts:
        return "{}"
    items = sorted(pts, key=repr)
    if len(items) <= int(max_items):
        return "{" + ", ".join(repr(p) for p in items) + "}"
    head = items[: int(max_items)]
    return "{" + ", ".join(repr(p) for p in head) + f", ... +{len(items) - int(max_items)}" + "}"


def _choose_uniform(rng: random.Random, items: Sequence[NodeId]) -> NodeId:
    if not items:
        raise ValueError("Cannot choose from empty sequence.")
    return items[int(rng.randrange(0, len(items)))]


def generate_random_path_on_graph(
    *,
    graph: Graph[NodeId],
    start_node: NodeId,
    num_steps: int,
    seed: int,
    retract_probability: float = 0.30,
    avoid_immediate_backtrack: bool = True,
    dead_end_policy: str = "stay",
    inject_every_n_steps: Optional[int] = 5,
    coverage_fn: Optional[Callable[[NodeId], Set[Tuple[float, float]]]] = None,
) -> GraphPath[NodeId]:
    """
    Random walk over a graph (geometry-free). Can retract by stepping back through
    history even without a reverse edge. `dead_end_policy` when stuck with no
    neighbors and no retract: "stay" | "restart" (back to start) | "terminate".
    """
    if int(num_steps) < 0:
        raise ValueError("num_steps must be >= 0.")

    retract_probability = float(retract_probability)
    if not (0.0 <= retract_probability <= 1.0):
        raise ValueError("retract_probability must be in [0.0, 1.0].")

    dead_end_policy = str(dead_end_policy).strip().lower()
    if dead_end_policy not in {"stay", "restart", "terminate"}:
        raise ValueError('dead_end_policy must be one of {"stay", "restart", "terminate"}.')

    if inject_every_n_steps is not None:
        inject_every_n_steps = int(inject_every_n_steps)
        if inject_every_n_steps <= 0:
            raise ValueError("inject_every_n_steps must be > 0 or None.")

    node_set = set(graph.nodes)
    if start_node not in node_set:
        raise KeyError(f"start_node {start_node!r} not in graph.nodes.")

    # Normalize once so the rest of the function can treat it as always-callable.
    coverage_fn = coverage_fn or (lambda _node_id: set())

    rng = random.Random(int(seed))

    steps: list[GraphPathStep[NodeId]] = []
    history: list[NodeId] = [start_node]
    covered_union: set[Tuple[float, float]] = set()

    for i in range(int(num_steps)):
        current = history[-1]
        previous = history[-2] if len(history) >= 2 else None

        # Inject is a pure "do something here" action: it does NOT move and does
        # NOT change the retract history (matches the 2D implementation).
        if inject_every_n_steps is not None and (i + 1) % inject_every_n_steps == 0:
            covered = set(coverage_fn(current) or ())
            covered_union |= covered
            logger.info(
                "path.step i=%d action=inject at=%r prev=%r covered=%s total_covered=%d",
                i,
                current,
                previous,
                format_covered_points(covered),
                len(covered_union),
            )
            steps.append(
                GraphPathStep(
                    step_index=i,
                    action="inject",
                    node=current,
                    previous_node=previous,
                )
            )
            continue

        nbrs = list(graph.adjacency.get(current, ()))
        can_retract = len(history) >= 2

        # Dead-end handling: prefer retract if possible; otherwise follow policy.
        if not nbrs:
            if can_retract:
                logger.info(
                    "path.step i=%d action=retract dead_end at=%r prev=%r",
                    i,
                    current,
                    previous,
                )
                steps.append(
                    GraphPathStep(
                        step_index=i,
                        action="retract",
                        node=current,
                        previous_node=previous,
                    )
                )
                history.pop()
                continue
            if dead_end_policy == "stay":
                logger.info(
                    "path.step i=%d action=stay dead_end at=%r prev=%r",
                    i,
                    current,
                    previous,
                )
                steps.append(
                    GraphPathStep(
                        step_index=i,
                        action="stay",
                        node=current,
                        previous_node=previous,
                    )
                )
                continue
            if dead_end_policy == "restart":
                logger.info(
                    "path.step i=%d action=restart dead_end at=%r prev=%r -> start=%r",
                    i,
                    current,
                    previous,
                    start_node,
                )
                steps.append(
                    GraphPathStep(
                        step_index=i,
                        action="restart",
                        node=current,
                        previous_node=previous,
                    )
                )
                history = [start_node]
                continue
            logger.info(
                "path.step i=%d action=terminate dead_end at=%r prev=%r",
                i,
                current,
                previous,
            )
            break  # terminate

        # Optional retract even when forward moves exist.
        if can_retract and float(rng.random()) < retract_probability:
            logger.info(
                "path.step i=%d action=retract at=%r prev=%r nbrs=%s",
                i,
                current,
                previous,
                _fmt_seq(nbrs),
            )
            steps.append(
                GraphPathStep(
                    step_index=i,
                    action="retract",
                    node=current,
                    previous_node=previous,
                )
            )
            history.pop()
            continue

        if avoid_immediate_backtrack and previous is not None and len(nbrs) > 1:
            forward = [n for n in nbrs if n != previous]
            nxt = _choose_uniform(rng, forward or nbrs)
        else:
            nxt = _choose_uniform(rng, nbrs)

        logger.info(
            "path.step i=%d action=move at=%r prev=%r nbrs=%s next=%r",
            i,
            current,
            previous,
            _fmt_seq(nbrs),
            nxt,
        )
        steps.append(
            GraphPathStep(
                step_index=i,
                action="move",
                node=current,
                previous_node=previous,
                next_node=nxt,
            )
        )
        history.append(nxt)

    return GraphPath(
        seed=int(seed),
        start_node=start_node,
        num_steps=int(num_steps),
        steps=tuple(steps),
    )
