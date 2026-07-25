from __future__ import annotations

"""
Bottom-up dynamic-programming planner for the steerable-needle therapy tour.

Processes nodes deepest-first (memoized once each). Per node: make one base plan
per dosage level, then for each child expand every current plan with every child
plan (pay the round trip, gain its coverage/damage) and re-compress to the Pareto
frontier. Geometry-free -- takes a Graph, per-node dosage bitsets and a depth map,
so it serves 2D and 3D unchanged.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from graph_utils import Graph, GraphPath, GraphPathStep, NodeId
from pareto_dominance import (
    CompressionStats,
    EpsVec,
    InjectionChoice,
    SubtourMerge,
    Plan,
    insert_plan,
    make_local_plan,
    make_merge_plan,
    scale_eps_for_depth,
)

logger = logging.getLogger(__name__)

DosageLevel = int

# Maps each node to a dict of { dosage_level -> integer bitset }.
# Built once before the DP starts; one map for tumor coverage, one for healthy-tissue damage.
DosageMaskMap = Mapping[NodeId, Mapping[DosageLevel, int]]

# Every roadmap edge is one motion-primitive hop, so a move (or retract) costs 1.
EDGE_HOP_COST = 1.0

# Hard cap on plans kept per node. Runtime is ~quadratic in this cap. Raise for a
# richer frontier, lower for speed. See docs/DP_THERAPY_PLANNER.md's
# "Parameters -> time estimates" section for measured timings -- they vary too
# much by map/dimension/hardware to hardcode here.
MAX_PLANS_PER_NODE = 100000


@dataclass
class NodeStats:
    """Per-node instrumentation from one `_dp_at_node` call -- everything
    needed to answer "where did the time/space actually go" after a run."""
    depth: int
    num_children: int
    local_frontier_size: int = 0
    final_frontier_size: int = 0
    peak_frontier_size: int = 0
    candidates_built: int = 0            # merge candidates actually constructed
    candidates_skipped_no_coverage: int = 0  # merges skipped by the no-new-coverage prune
    comparisons: int = 0                 # eps_dominates calls across this node's compress() calls
    plans_inserted: int = 0
    plans_absorbed: int = 0
    plans_rejected: int = 0
    time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "num_children": self.num_children,
            "local_frontier_size": self.local_frontier_size,
            "final_frontier_size": self.final_frontier_size,
            "peak_frontier_size": self.peak_frontier_size,
            "candidates_built": self.candidates_built,
            "candidates_skipped_no_coverage": self.candidates_skipped_no_coverage,
            "comparisons": self.comparisons,
            "plans_inserted": self.plans_inserted,
            "plans_absorbed": self.plans_absorbed,
            "plans_rejected": self.plans_rejected,
            "time_seconds": self.time_seconds,
        }


@dataclass
class DPStats:
    """
    Aggregate + per-node instrumentation for one `compute_dp` call. Pass an
    instance in via `compute_dp(..., stats=...)` to have it populated; reading
    it afterward answers exactly what the profiling/bottleneck-analysis
    questions in docs/DP_THERAPY_PLANNER.md ask: per-node comparison counts,
    frontier sizes, and timing, plus the aggregate rollups below.

    Deliberately just numbers, not `Plan`-aware -- callers that also want a
    memory estimate should wrap their `compute_dp` call in `tracemalloc`
    (see `main_demo.py`'s `run_dp`) and set `peak_memory_bytes` themselves,
    since that measures real allocator activity instead of a hand-rolled
    object-size guess.
    """
    per_node: Dict[NodeId, NodeStats] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    reordering_enabled: bool = True
    peak_memory_bytes: Optional[int] = None

    def total_comparisons(self) -> int:
        return sum(n.comparisons for n in self.per_node.values())

    def total_candidates_built(self) -> int:
        return sum(n.candidates_built for n in self.per_node.values())

    def total_candidates_skipped(self) -> int:
        return sum(n.candidates_skipped_no_coverage for n in self.per_node.values())

    def total_plans_inserted(self) -> int:
        return sum(n.plans_inserted for n in self.per_node.values())

    def peak_frontier_size(self) -> int:
        return max((n.peak_frontier_size for n in self.per_node.values()), default=0)

    def mean_final_frontier_size(self) -> float:
        sizes = [n.final_frontier_size for n in self.per_node.values()]
        return (sum(sizes) / len(sizes)) if sizes else 0.0

    def total_final_frontier_size(self) -> int:
        """Sum of every node's surviving frontier -- the actual size of the
        final `memo` table (a direct proxy for its steady-state memory)."""
        return sum(n.final_frontier_size for n in self.per_node.values())

    def top_n_by(self, metric: str, n: int = 20) -> List[Tuple[NodeId, NodeStats]]:
        """The `n` nodes with the largest value of `metric` (any NodeStats
        field name, e.g. "time_seconds", "comparisons", "peak_frontier_size")
        -- the few worst offenders that actually explain the total runtime,
        per docs/DP_THERAPY_PLANNER.md's "the few shallow nodes near the
        start dominate runtime" observation."""
        return sorted(
            self.per_node.items(), key=lambda kv: getattr(kv[1], metric), reverse=True
        )[:n]

    def to_summary_dict(self) -> dict:
        """Aggregate-only view (no per-node breakdown) -- what gets written to
        dp_stats.json's top level; see `to_node_rows` for the full per-node table."""
        return {
            "reordering_enabled": self.reordering_enabled,
            "total_time_seconds": self.total_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": (
                round(self.peak_memory_bytes / (1024 * 1024), 3)
                if self.peak_memory_bytes is not None
                else None
            ),
            "num_nodes_processed": len(self.per_node),
            "total_comparisons": self.total_comparisons(),
            "total_candidates_built": self.total_candidates_built(),
            "total_candidates_skipped_no_coverage": self.total_candidates_skipped(),
            "total_plans_inserted": self.total_plans_inserted(),
            "peak_frontier_size": self.peak_frontier_size(),
            "mean_final_frontier_size": self.mean_final_frontier_size(),
            "total_final_frontier_size": self.total_final_frontier_size(),
            "top_10_nodes_by_time": [
                {"node": str(node), **ns.to_dict()} for node, ns in self.top_n_by("time_seconds", 10)
            ],
            "top_10_nodes_by_comparisons": [
                {"node": str(node), **ns.to_dict()} for node, ns in self.top_n_by("comparisons", 10)
            ],
        }

    def to_node_rows(self) -> List[dict]:
        """One row per node, for a CSV -- the full detail `to_summary_dict`
        only samples the top offenders from."""
        rows = []
        for node, ns in self.per_node.items():
            rows.append({"node": str(node), **ns.to_dict()})
        return rows


def bfs_depth(graph: Graph[NodeId], start_node: NodeId) -> Dict[NodeId, int]:
    """BFS edge-distance from `start_node` to every reachable node (the deepest-
    first processing order the DP needs). The real roadmap already has depth; this
    is for standalone/test use."""
    depth: Dict[NodeId, int] = {start_node: 0}
    queue = [start_node]
    i = 0
    while i < len(queue):
        u = queue[i]
        i += 1
        for v in graph.adjacency.get(u, ()):
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    return depth


def build_dosage_masks(
    *,
    graph: Graph[NodeId],
    dosage_fn: Callable[[NodeId, DosageLevel], Tuple[Iterable[object], Iterable[object]]],
    dosage_levels: Sequence[DosageLevel],
) -> Tuple[DosageMaskMap, DosageMaskMap]:
    """
    Turn a `dosage_fn(node, d) -> (covered_points, damaged_points)` callback into
    per-node, per-dosage integer bitsets (one global bit index per point, so unions
    compose). Returns (coverage_masks, damage_masks).
    """
    all_coverage_points: set = set()
    all_damage_points: set = set()
    raw: Dict[NodeId, Dict[DosageLevel, Tuple[set, set]]] = {}
    for n in graph.nodes:
        per_dose: Dict[DosageLevel, Tuple[set, set]] = {}
        for d in dosage_levels:
            covered, damaged = dosage_fn(n, d)
            coverage_set = set(covered)
            damage_set = set(damaged)
            per_dose[int(d)] = (coverage_set, damage_set)
            all_coverage_points |= coverage_set
            all_damage_points |= damage_set
        raw[n] = per_dose

    coverage_index = {p: i for i, p in enumerate(sorted(all_coverage_points))}
    damage_index = {p: i for i, p in enumerate(sorted(all_damage_points))}

    coverage_masks: Dict[NodeId, Dict[DosageLevel, int]] = {}
    damage_masks: Dict[NodeId, Dict[DosageLevel, int]] = {}
    for n in graph.nodes:
        coverage_by_dose: Dict[DosageLevel, int] = {}
        damage_by_dose: Dict[DosageLevel, int] = {}
        for d, (coverage_set, damage_set) in raw[n].items():
            coverage_mask = 0
            for p in coverage_set:
                coverage_mask |= 1 << coverage_index[p]
            damage_mask = 0
            for p in damage_set:
                damage_mask |= 1 << damage_index[p]
            coverage_by_dose[d] = coverage_mask
            damage_by_dose[d] = damage_mask
        coverage_masks[n] = coverage_by_dose
        damage_masks[n] = damage_by_dose

    return coverage_masks, damage_masks


def _trim_to_cap(
    plans: list[Plan], max_plans_per_node: Optional[int], *, node: NodeId, warn: bool = True
) -> list[Plan]:
    if max_plans_per_node is None or len(plans) <= max_plans_per_node:
        return plans
    if warn:
        logger.warning(
            "dp_on_graph: node %r has %d plans, exceeding max_plans_per_node=%d; "
            "truncating to the highest-coverage/lowest-damage/shortest plans. "
            "Consider tightening eps_vec.",
            node,
            len(plans),
            max_plans_per_node,
        )
    # Rank by most coverage, then least damage, shortest, fewest doses.
    ranked = sorted(plans, key=lambda p: (-p.coverage_count, p.damage_count, p.length, p.dose_count))
    return ranked[:max_plans_per_node]


def _compress_with_cap(
    plans: Iterable[Plan],
    eps_vec: EpsVec,
    max_plans_per_node: Optional[int],
    *,
    node: NodeId,
    stats: Optional[CompressionStats] = None,
) -> list[Plan]:
    """
    Pareto-compress `plans`, keeping at most `max_plans_per_node` survivors.
    Trimming back to the cap whenever the list exceeds 2x it keeps each insertion
    cheap -- otherwise a loose eps_vec lets the mid-run frontier blow up.
    Pass `None` for no cap (unbounded frontier).
    """
    survivors: list[Plan] = []
    trim_at = max(1, int(max_plans_per_node)) * 2 if max_plans_per_node is not None else None
    for p in plans:
        survivors = insert_plan(survivors, p, eps_vec, stats=stats)
        if trim_at is not None and len(survivors) > trim_at:
            survivors = _trim_to_cap(survivors, max_plans_per_node, node=node, warn=False)
    # Hitting the cap is normal on a real map, so don't warn (the progress
    # counter already surfaces frontier sizes).
    return _trim_to_cap(survivors, max_plans_per_node, node=node, warn=False)


def _forward_children(graph: Graph[NodeId], depth: Mapping[NodeId, int]) -> Dict[NodeId, list[NodeId]]:
    """
    Keep only out-edges to strictly-deeper nodes (depth[child] == depth[node]+1).

    The roadmap records an edge for every primitive move, including ones that land
    on an already-discovered shallower/equal-depth node. Those break the deepest-
    first ordering; dropping them recovers a clean layered DAG ("go deeper, then
    retract the way you came").
    """
    children: Dict[NodeId, list[NodeId]] = {}
    for u in graph.nodes:
        u_depth = depth.get(u)
        if u_depth is None:
            continue
        kids = [
            v
            for v in graph.adjacency.get(u, ())
            if depth.get(v) == u_depth + 1
        ]
        if kids:
            children[u] = kids
    return children


def _child_order_key(child: NodeId, memo: Dict[NodeId, list[Plan]]) -> tuple:
    """
    Rank a child by its best already-computed plan, using the exact same
    ranking `_trim_to_cap` uses for cap-truncation (most coverage, then least
    damage, shortest, fewest doses).

    `_dp_at_node` sorts its children by this key (best first) before merging
    them in one at a time. The final frontier is the same regardless of
    order -- eps-dominance defines a fixed final survivor set -- but the
    *work* to reach it isn't: `insert_plan` rejects a dominated newcomer
    immediately (cheap), while a newcomer that dominates an existing survivor
    causes that survivor's eviction plus an `absorb_discarded` call (more
    work). Merging strong children first means later, weaker children's
    candidates tend to be dominated on first contact instead of surviving
    into the frontier and being evicted once something better arrives.
    See docs/DP_THERAPY_PLANNER.md for the full reasoning and measurements.
    """
    child_plans = memo.get(child) or []
    if not child_plans:
        return (0, 0, 0.0, 0)
    best = min(child_plans, key=lambda p: (-p.coverage_count, p.damage_count, p.length, p.dose_count))
    return (-best.coverage_count, best.damage_count, best.length, best.dose_count)


def _dp_at_node(
    v: NodeId,
    *,
    children_of: Mapping[NodeId, Sequence[NodeId]],
    memo: Dict[NodeId, list[Plan]],
    dosage_levels: Sequence[DosageLevel],
    coverage_masks: DosageMaskMap,
    damage_masks: DosageMaskMap,
    eps_vec: EpsVec,
    max_plans_per_node: Optional[int],
    depth: int = 0,
    reorder_children: bool = True,
    stats: Optional[DPStats] = None,
) -> list[Plan]:
    node_t0 = time.perf_counter() if stats is not None else 0.0
    comp_stats = CompressionStats() if stats is not None else None
    candidates_built = 0
    candidates_skipped = 0

    coverage_by_dose = coverage_masks.get(v, {})
    damage_by_dose = damage_masks.get(v, {})
    local_plans = [
        make_local_plan(
            node=v,
            dosage=d,
            length=0.0,
            coverage=coverage_by_dose.get(int(d), 0),
            damage=damage_by_dose.get(int(d), 0),
        )
        for d in dosage_levels
    ]
    frontier = _compress_with_cap(local_plans, eps_vec, max_plans_per_node, node=v, stats=comp_stats)
    local_frontier_size = len(frontier)
    peak_frontier_size = local_frontier_size

    children = children_of.get(v, ())
    ordered_children = (
        sorted(children, key=lambda c: _child_order_key(c, memo))
        if reorder_children and len(children) > 1
        else children
    )

    for c in ordered_children:
        if c not in memo:
            # Shouldn't happen (children are strictly deeper); skip defensively.
            logger.warning("dp_on_graph: child %r of %r not yet computed; skipping.", c, v)
            continue
        child_plans = memo[c]

        def _candidates() -> Iterable[Plan]:
            nonlocal candidates_built, candidates_skipped
            yield from frontier  # baseline: skip this child
            for base in frontier:
                base_coverage = base.coverage
                for child_plan in child_plans:
                    # Skip merges that add no new coverage: the result is
                    # Pareto-dominated by `base` (already a candidate), so it can
                    # never survive -- but building+inserting it is expensive.
                    if child_plan.coverage | base_coverage == base_coverage:
                        if stats is not None:
                            candidates_skipped += 1
                        continue
                    if stats is not None:
                        candidates_built += 1
                    yield make_merge_plan(
                        node=v,
                        child=c,
                        base=base,
                        child_plan=child_plan,
                        round_trip_cost=2.0 * EDGE_HOP_COST,
                    )

        frontier = _compress_with_cap(
            _candidates(), eps_vec, max_plans_per_node, node=v, stats=comp_stats
        )
        if stats is not None:
            peak_frontier_size = max(peak_frontier_size, len(frontier))

    if stats is not None:
        stats.per_node[v] = NodeStats(
            depth=depth,
            num_children=len(children),
            local_frontier_size=local_frontier_size,
            final_frontier_size=len(frontier),
            peak_frontier_size=peak_frontier_size,
            candidates_built=candidates_built,
            candidates_skipped_no_coverage=candidates_skipped,
            comparisons=comp_stats.comparisons,
            plans_inserted=comp_stats.plans_inserted,
            plans_absorbed=comp_stats.plans_absorbed,
            plans_rejected=comp_stats.plans_rejected,
            time_seconds=time.perf_counter() - node_t0,
        )

    return frontier


def compute_dp(
    *,
    graph: Graph[NodeId],
    start_node: NodeId,
    depth: Mapping[NodeId, int],
    dosage_levels: Sequence[DosageLevel],
    coverage_masks: DosageMaskMap,
    damage_masks: DosageMaskMap,
    eps_vec: EpsVec,
    max_plans_per_node: Optional[int] = MAX_PLANS_PER_NODE,
    progress: bool = False,
    depth_scaled_eps: bool = False,
    reorder_children: bool = True,
    stats: Optional[DPStats] = None,
) -> Dict[NodeId, list[Plan]]:
    """
    Compute the Pareto frontier `memo[v]` for every reachable node, deepest-first
    so each node is finalized once before any parent needs it.

    `progress=True` prints a one-line counter (work is very uneven -- the few
    shallow nodes near the start dominate runtime).
    `depth_scaled_eps=True` treats `eps_vec` as a global target and shrinks it per
    node via `scale_eps_for_depth` (tighter, slower; default off).
    `reorder_children=True` (default) merges each node's children strongest-
    first (see `_child_order_key`) -- a pure scheduling optimization, same
    final frontier, less transient insert/absorb work. Pass `False` to force
    the original graph-adjacency order, e.g. for an apples-to-apples benchmark.
    `stats`, if given, is populated with per-node and aggregate instrumentation
    (comparisons, frontier sizes, timing) -- see `DPStats`.
    """
    dp_t0 = time.perf_counter() if stats is not None else 0.0
    if depth_scaled_eps:
        eps_vec = scale_eps_for_depth(eps_vec, max(depth.values(), default=1))
    if stats is not None:
        stats.reordering_enabled = reorder_children
    children_of = _forward_children(graph, depth)
    order = sorted(depth.keys(), key=lambda n: depth[n], reverse=True)
    memo: Dict[NodeId, list[Plan]] = {}
    n_total = len(order)
    for i, v in enumerate(order):
        memo[v] = _dp_at_node(
            v,
            children_of=children_of,
            memo=memo,
            dosage_levels=dosage_levels,
            coverage_masks=coverage_masks,
            damage_masks=damage_masks,
            eps_vec=eps_vec,
            max_plans_per_node=max_plans_per_node,
            depth=depth.get(v, 0),
            reorder_children=reorder_children,
            stats=stats,
        )
        if progress and (i % 25 == 0 or i == n_total - 1):
            pct = 100 * (i + 1) // max(n_total, 1)
            print(
                f"\rDP: {i + 1}/{n_total} nodes ({pct}%)  "
                f"frontier@node={len(memo[v])}      ",
                end="",
                flush=True,
            )
    if progress:
        print(flush=True)  # newline after the progress line
    if start_node not in memo:
        raise KeyError(f"start_node {start_node!r} not in depth map")
    if stats is not None:
        stats.total_time_seconds = time.perf_counter() - dp_t0
    return memo


def select_plan(plans: list[Plan]) -> Plan:
    """Default selection rule: maximize coverage, tie-break by minimum length."""
    if not plans:
        raise ValueError("select_plan called with an empty plan list")
    return min(plans, key=lambda p: (-p.coverage_count, p.length))


def reconstruct_graph_path(
    plan: Plan, *, start_node: NodeId, seed: int
) -> GraphPath[NodeId]:
    """
    Unfold `plan.origin` into a move/inject/retract step sequence: InjectionChoice
    emits an inject (if dosage > 0); SubtourMerge emits its base, a move to the
    child, the child's own steps, then a retract back.
    """
    steps: list[GraphPathStep[NodeId]] = []

    def emit(origin) -> None:
        if isinstance(origin, InjectionChoice):
            if origin.dosage > 0:
                steps.append(
                    GraphPathStep(
                        step_index=len(steps),
                        action="inject",
                        node=origin.node,
                        previous_node=None,
                        next_node=None,
                        dosage=int(origin.dosage),
                    )
                )
            return
        if isinstance(origin, SubtourMerge):
            emit(origin.base.origin)
            steps.append(
                GraphPathStep(
                    step_index=len(steps),
                    action="move",
                    node=origin.node,
                    previous_node=None,
                    next_node=origin.child,
                )
            )
            emit(origin.child_plan.origin)
            steps.append(
                GraphPathStep(
                    step_index=len(steps),
                    action="retract",
                    node=origin.child,
                    previous_node=origin.node,
                    next_node=None,
                )
            )
            return
        raise TypeError(f"Unknown PlanOrigin type: {type(origin)!r}")

    emit(plan.origin)
    return GraphPath(seed=int(seed), start_node=start_node, num_steps=len(steps), steps=tuple(steps))


def generate_dp_plan_on_graph(
    *,
    graph: Graph[NodeId],
    start_node: NodeId,
    seed: int,
    dosage_fn: Callable[[NodeId, DosageLevel], Tuple[Iterable[object], Iterable[object]]],
    dosage_levels: Sequence[DosageLevel],
    eps_vec: EpsVec,
    depth: Optional[Mapping[NodeId, int]] = None,
    max_plans_per_node: Optional[int] = MAX_PLANS_PER_NODE,
    select_plan_fn: Callable[[list[Plan]], Plan] = select_plan,
    depth_scaled_eps: bool = False,
    reorder_children: bool = True,
    stats: Optional[DPStats] = None,
) -> GraphPath[NodeId]:
    """
    End-to-end entry point: build dosage masks, run the DP, pick one
    representative plan from DP(start_node), and turn it into a `GraphPath`.
    """
    if start_node not in set(graph.nodes):
        raise KeyError(f"start_node {start_node!r} not in graph.nodes")

    depth_map = bfs_depth(graph, start_node) if depth is None else dict(depth)
    coverage_masks, damage_masks = build_dosage_masks(
        graph=graph, dosage_fn=dosage_fn, dosage_levels=dosage_levels
    )
    memo = compute_dp(
        graph=graph,
        start_node=start_node,
        depth=depth_map,
        dosage_levels=dosage_levels,
        coverage_masks=coverage_masks,
        damage_masks=damage_masks,
        eps_vec=eps_vec,
        max_plans_per_node=max_plans_per_node,
        depth_scaled_eps=depth_scaled_eps,
        reorder_children=reorder_children,
        stats=stats,
    )
    best = select_plan_fn(memo[start_node])
    logger.info(
        "DP planning complete: chosen plan length=%.3f coverage=%d damage=%d dose_count=%d "
        "(frontier size at start=%d)",
        best.length,
        best.coverage.bit_count(),
        best.damage.bit_count(),
        best.dose_count,
        len(memo[start_node]),
    )
    return reconstruct_graph_path(best, start_node=start_node, seed=seed)
