from __future__ import annotations

"""
Graph-only IRIS (inspection planning) implementation, per Fu et al.,
"Toward Asymptotically-Optimal Inspection Planning via Efficient Near-Optimal
Graph Search" (arXiv:1907.00506). See docs/IRIS_INSPECTION_PLANNER.md for the
full paper-to-code mapping.

We intentionally do NOT implement the roadmap-densification (RRT/RRG) part of
IRIS -- the roadmap graph is a given input, like `random_path_on_graph.py`.
"""

from dataclasses import dataclass
import logging
import time
from typing import Callable, ClassVar, Dict, Generic, Iterable, List, Literal, Optional, Tuple

from heapdict import heapdict

from graph_utils import Graph, GraphPath, GraphPathStep, NodeId

logger = logging.getLogger(__name__)

POI = object  # opaque, hashable set element -- (x,y) in 2D, (ix,iy,iz) voxel index in 3D, etc.

# Coverage as a bitset over a fixed POI index order (see `_build_node_pois`).
CoverageBits = int

# Set to True (e.g. from a test) to hard-fail on two distinct search states that
# hash-collide on (node, g, covered, injections_remaining, trail, last_action) --
# a debug assertion, not a production safety net.
DEBUG_ASSERT_UNIQUE_STATES = False


# ---------------------------------------------------------------------------
# Logging (sampled detail logs + always-on statistics)
# ---------------------------------------------------------------------------


class IrisLogger:
    """Metrics and sampled detail logging for IRIS planning runs."""

    LOG_SAMPLE_RATE: ClassVar[int] = 100
    _sample_counts: ClassVar[Dict[str, int]] = {}

    # Trail-dominance metrics (reset per planning run)
    trail_dominance_checks: ClassVar[int] = 0
    trail_dominance_pass: ClassVar[int] = 0
    trail_dominance_fail: ClassVar[int] = 0
    trail_dominance_fail_empty_trail: ClassVar[int] = 0
    trail_dominance_fail_head_mismatch: ClassVar[int] = 0
    trail_dominance_fail_no_anchor: ClassVar[int] = 0
    trail_dominance_fail_cost_exceeded: ClassVar[int] = 0
    trail_dominance_fail_retract_leq_forward: ClassVar[int] = 0

    # Timing (reset when a new plan generation starts)
    injection_delta_time: ClassVar[float] = 0.0
    dominates_time: ClassVar[float] = 0.0
    strict_dominates_time: ClassVar[float] = 0.0
    duplicate_state_collisions: ClassVar[int] = 0
    pap_subsumptions: ClassVar[int] = 0

    @classmethod
    def reset(cls) -> None:
        cls._sample_counts.clear()
        cls.injection_delta_time = 0.0
        cls.dominates_time = 0.0
        cls.strict_dominates_time = 0.0
        cls.duplicate_state_collisions = 0
        cls.pap_subsumptions = 0
        cls.trail_dominance_checks = 0
        cls.trail_dominance_pass = 0
        cls.trail_dominance_fail = 0
        cls.trail_dominance_fail_empty_trail = 0
        cls.trail_dominance_fail_head_mismatch = 0
        cls.trail_dominance_fail_no_anchor = 0
        cls.trail_dominance_fail_cost_exceeded = 0
        cls.trail_dominance_fail_retract_leq_forward = 0

    @classmethod
    def should_sample_log(cls, key: str) -> bool:
        count = cls._sample_counts.get(key, 0) + 1
        cls._sample_counts[key] = count
        return count == 1 or count % cls.LOG_SAMPLE_RATE == 0

    @classmethod
    def log_pap_subsumption(cls) -> None:
        cls.pap_subsumptions += 1

    @classmethod
    def record_trail_dominance_result(cls, *, passed: bool, reason: Optional[str] = None) -> None:
        cls.trail_dominance_checks += 1
        if passed:
            cls.trail_dominance_pass += 1
            return
        cls.trail_dominance_fail += 1
        if reason == "empty_trail":
            cls.trail_dominance_fail_empty_trail += 1
        elif reason == "head_mismatch":
            cls.trail_dominance_fail_head_mismatch += 1
        elif reason == "no_anchor":
            cls.trail_dominance_fail_no_anchor += 1
        elif reason == "cost_exceeded":
            cls.trail_dominance_fail_cost_exceeded += 1
        elif reason == "retract_leq_forward":
            cls.trail_dominance_fail_retract_leq_forward += 1

    @classmethod
    def log_planning_complete(
        cls,
        *,
        covered_pois: int,
        remaining_pois: int,
        total_pois: int,
        injections_remaining: int,
        injection_budget: int,
    ) -> None:
        logger.info(
            "IRIS planning complete: POIs covered=%s remaining=%s (of %s); "
            "injections remaining=%s (budget %s)",
            covered_pois,
            remaining_pois,
            total_pois,
            injections_remaining,
            injection_budget,
        )


def log_iris_planner_metrics(log: logging.Logger, *, prefix: str = "iris") -> None:
    """Log `IrisLogger` timing/trail-dominance counters (call after a planning run)."""
    log.info(
        "%s injection_delta_time=%.6fs dominates_time=%.6fs strict_dominates_time=%.6fs "
        "duplicate_state_collisions=%s pap_subsumptions=%s",
        prefix,
        IrisLogger.injection_delta_time,
        IrisLogger.dominates_time,
        IrisLogger.strict_dominates_time,
        IrisLogger.duplicate_state_collisions,
        IrisLogger.pap_subsumptions,
    )
    checks = int(IrisLogger.trail_dominance_checks)
    passed = int(IrisLogger.trail_dominance_pass)
    failed = int(IrisLogger.trail_dominance_fail)
    pct = (100.0 * float(passed) / float(checks)) if checks > 0 else 0.0
    log.info(
        "%s trail_dominance_checks=%s pass=%s fail=%s pass_pct=%.2f%% "
        "(TRAIL_DOMINANCE_EPSILON=%s fail_empty_trail=%s fail_head_mismatch=%s "
        "fail_no_anchor=%s fail_retract_leq_forward=%s fail_cost_exceeded=%s)",
        prefix,
        checks,
        passed,
        failed,
        pct,
        IrisPlanner.TRAIL_DOMINANCE_EPSILON,
        IrisLogger.trail_dominance_fail_empty_trail,
        IrisLogger.trail_dominance_fail_head_mismatch,
        IrisLogger.trail_dominance_fail_no_anchor,
        IrisLogger.trail_dominance_fail_retract_leq_forward,
        IrisLogger.trail_dominance_fail_cost_exceeded,
    )


def get_iris_planner_metrics() -> Dict[str, float]:
    """Return current `IrisLogger` timing/trail-dominance counters (for tests/logging)."""
    return {
        "injection_delta_time": IrisLogger.injection_delta_time,
        "dominates_time": IrisLogger.dominates_time,
        "strict_dominates_time": IrisLogger.strict_dominates_time,
        "duplicate_state_collisions": float(IrisLogger.duplicate_state_collisions),
        "pap_subsumptions": float(IrisLogger.pap_subsumptions),
        "trail_dominance_checks": float(IrisLogger.trail_dominance_checks),
        "trail_dominance_pass": float(IrisLogger.trail_dominance_pass),
        "trail_dominance_fail": float(IrisLogger.trail_dominance_fail),
        "trail_dominance_fail_empty_trail": float(IrisLogger.trail_dominance_fail_empty_trail),
        "trail_dominance_fail_head_mismatch": float(IrisLogger.trail_dominance_fail_head_mismatch),
        "trail_dominance_fail_no_anchor": float(IrisLogger.trail_dominance_fail_no_anchor),
        "trail_dominance_fail_cost_exceeded": float(IrisLogger.trail_dominance_fail_cost_exceeded),
        "trail_dominance_fail_retract_leq_forward": float(
            IrisLogger.trail_dominance_fail_retract_leq_forward
        ),
    }


class IrisPlanner:
    """
    Graph IRIS planner: tunable parameters, search, and class-level timing counters
    (updated during planning; reset at the start of each `generate_iris_plan_on_graph` call).
    """

    # Initial approximation parameters (epsilon >= 0, p in (0, 1])
    P0: ClassVar[float] = 0.80
    EPSILON0: ClassVar[float] = 2.0

    # Tightening factor f in (0,1]. Each outer iteration updates:
    #   p <- p + f*(1-p)        (increases toward 1)
    #   epsilon <- epsilon + f*(0-epsilon)   (decreases toward 0)
    TIGHTENING_F: ClassVar[float] = 0.05

    # Limits/budgets to keep the search practical.
    MAX_OUTER_ITERS: ClassVar[int] = 3  # set to 1 while debugging
    MAX_EXPANSIONS: ClassVar[int] = 1000000

    # Max inject actions per plan (coverage increases only on inject).
    MAX_INJECTIONS_PER_RUN: ClassVar[int] = 7

    # Trail-dominance tolerance X: hybrid rejoin cost must be <= (1+X) * retract-only cost on B.
    TRAIL_DOMINANCE_EPSILON: ClassVar[float] = 3.0

    INJECTION_COST: ClassVar[int] = 1
    RETRACT_COST: ClassVar[int] = 1
    MOVE_COST: ClassVar[int] = 1

    @staticmethod
    def _tighten(p: float, epsilon: float, f: float) -> Tuple[float, float]:
        """Tighten (p, epsilon) toward (1, 0) by fraction `f` for the next outer iteration."""
        f = float(f)
        if not (0.0 < f <= 1.0):
            raise ValueError("TIGHTENING_F must be in (0, 1].")
        p = float(p)
        epsilon = float(epsilon)
        p = p + f * (1.0 - p)
        epsilon = epsilon + f * (0.0 - epsilon)
        p = min(1.0, max(0.0, p))
        epsilon = max(0.0, epsilon)
        return p, epsilon

    @staticmethod
    def _build_node_pois(
        *,
        graph: Graph[NodeId],
        coverage_fn: Callable[[NodeId], Iterable[POI]],
    ) -> Tuple[Dict[NodeId, CoverageBits], CoverageBits]:
        """
        Build per-node coverage bitsets over the discrete POI universe.

        POIs are opaque, hashable set elements (2D: (x,y) tuples; 3D: (ix,iy,iz)
        voxel indices) -- no coercion happens here beyond set/sorted-tuple
        bookkeeping, so both survive as distinct universe members.

        Returns (node_masks, all_mask): node_masks[node] is the bitset of POI
        indices covered by injecting there; all_mask has every POI index set.
        """
        all_pois_set: set = set()
        per_node: Dict[NodeId, set] = {}
        for n in graph.nodes:
            pts = set(coverage_fn(n) or ())
            per_node[n] = pts
            all_pois_set |= pts

        if not all_pois_set:
            return {}, 0

        sorted_pois = tuple(sorted(all_pois_set, key=repr))
        poi_to_idx: Dict[POI, int] = {p: i for i, p in enumerate(sorted_pois)}
        all_mask = (1 << len(sorted_pois)) - 1

        node_masks: Dict[NodeId, CoverageBits] = {}
        for n in graph.nodes:
            mask = 0
            for p in per_node.get(n, ()):
                mask |= 1 << poi_to_idx[p]
            node_masks[n] = mask

        return node_masks, all_mask

    @classmethod
    def _trail_dominates(
        cls,
        *,
        trail_a: Tuple[NodeId, ...],
        trail_b: Tuple[NodeId, ...],
    ) -> bool:
        """
        Whether trail A dominates trail B for retract purposes: same head node,
        and for every node on B, retracting along A to the rightmost shared
        anchor and walking forward along B costs at most
        (1 + TRAIL_DOMINANCE_EPSILON) times retracting along B directly.

        Why this exists: dominance based only on (g, coverage,
        injections_remaining) is unsound once retract is allowed -- a state's
        future retract options depend entirely on its own trail, not just
        those three fields. This bounds how much worse A's retract-based
        access to B's history can be before A is disqualified from dominating B.
        """
        epsilon = float(cls.TRAIL_DOMINANCE_EPSILON)
        trail_len_a = len(trail_a)
        trail_len_b = len(trail_b)

        if trail_len_a == 0 or trail_len_b == 0:
            IrisLogger.record_trail_dominance_result(passed=False, reason="empty_trail")
            return False
        if trail_a[-1] != trail_b[-1]:
            IrisLogger.record_trail_dominance_result(passed=False, reason="head_mismatch")
            return False
        if trail_a == trail_b:
            IrisLogger.record_trail_dominance_result(passed=True)
            return True

        node_to_b_indices: Dict[NodeId, List[int]] = {}
        for idx_on_b, node in enumerate(trail_b):
            node_to_b_indices.setdefault(node, []).append(idx_on_b)

        retract_cost = cls.RETRACT_COST
        move_cost = cls.MOVE_COST
        cost_bound_scale = 1.0 + epsilon

        for target_idx_on_b in range(trail_len_b - 1):
            cost_b = (trail_len_b - 1 - target_idx_on_b) * retract_cost
            cost_bound = int(cost_bound_scale * float(cost_b))

            anchor_idx_on_a: Optional[int] = None
            anchor_idx_on_b: Optional[int] = None
            for candidate_idx_on_a in range(min(trail_len_a - 1, cost_bound), -1, -1):
                candidate_anchor_node = trail_a[candidate_idx_on_a]
                indices_on_b = node_to_b_indices.get(candidate_anchor_node)
                if not indices_on_b:
                    continue
                rightmost_occurrence_on_b = None
                for occurrence_idx_on_b in indices_on_b:
                    if occurrence_idx_on_b <= target_idx_on_b:
                        rightmost_occurrence_on_b = occurrence_idx_on_b
                    else:
                        break
                if rightmost_occurrence_on_b is None:
                    continue
                anchor_idx_on_a = candidate_idx_on_a
                anchor_idx_on_b = rightmost_occurrence_on_b
                break

            if anchor_idx_on_a is None or anchor_idx_on_b is None:
                IrisLogger.record_trail_dominance_result(passed=False, reason="no_anchor")
                return False

            retract_on_a_cost = (trail_len_a - 1 - anchor_idx_on_a) * retract_cost
            forward_on_b_cost = (target_idx_on_b - anchor_idx_on_b) * move_cost
            cost_hybrid = retract_on_a_cost + forward_on_b_cost

            if retract_on_a_cost * (1.0 + (epsilon / 2.0)) <= forward_on_b_cost:
                IrisLogger.record_trail_dominance_result(passed=False, reason="retract_leq_forward")
                return False
            if cost_hybrid > cost_bound:
                IrisLogger.record_trail_dominance_result(passed=False, reason="cost_exceeded")
                return False

        IrisLogger.record_trail_dominance_result(passed=True)
        return True

    @classmethod
    def _approx_dominates(
        cls,
        *,
        g_a: int,
        covered_a: CoverageBits,
        injections_remaining_a: int,
        trail_a: Tuple[NodeId, ...],
        g_b: int,
        covered_b: CoverageBits,
        injections_remaining_b: int,
        trail_b: Tuple[NodeId, ...],
        epsilon: float,
        p: float,
        check_trail_dominance: bool = True,
    ) -> bool:
        """
        (epsilon, p)-dominance on (g, coverage), plus a repo-specific extension:
        A dominates B only if A has at least as many injections remaining as B
        (keeps future "inject" actions feasible under a finite budget), and --
        when `check_trail_dominance` -- A must also trail-dominate B (required
        for soundness whenever retract is enabled; see `_trail_dominates`).
        """
        dominates_start = time.perf_counter()
        epsilon = float(epsilon)
        p = float(p)
        g_bound = (1.0 + epsilon) * float(g_b)

        if injections_remaining_a < injections_remaining_b:
            result = False
        elif g_a > g_bound:
            result = False
        else:
            union_cnt = (covered_a | covered_b).bit_count()
            covered_a_cnt = covered_a.bit_count()
            if union_cnt == 0:
                result = True
            else:
                result = float(covered_a_cnt) >= p * float(union_cnt)

        if result and check_trail_dominance:
            result = cls._trail_dominates(trail_a=trail_a, trail_b=trail_b)

        IrisLogger.dominates_time += time.perf_counter() - dominates_start
        return result

    @classmethod
    def _strict_dominates(
        cls,
        *,
        g_a: int,
        covered_a: CoverageBits,
        injections_remaining_a: int,
        trail_a: Tuple[NodeId, ...],
        g_b: int,
        covered_b: CoverageBits,
        injections_remaining_b: int,
        trail_b: Tuple[NodeId, ...],
        check_trail_dominance: bool = True,
    ) -> bool:
        """
        Strict dominance (minimize g, maximize coverage, maximize injections
        remaining, reject exact ties): `_approx_dominates` with epsilon=0 and
        p=1, which removes the approximation slack entirely.
        """
        strict_start = time.perf_counter()
        result = cls._approx_dominates(
            g_a=g_a,
            covered_a=covered_a,
            injections_remaining_a=injections_remaining_a,
            trail_a=trail_a,
            g_b=g_b,
            covered_b=covered_b,
            injections_remaining_b=injections_remaining_b,
            trail_b=trail_b,
            epsilon=0.0,
            p=1.0,
            check_trail_dominance=check_trail_dominance,
        )
        IrisLogger.strict_dominates_time += time.perf_counter() - strict_start
        return result

    @dataclass(frozen=True, eq=False)
    class _State(Generic[NodeId]):
        # `eq=False` forces identity-based hash/equality (the default frozen-dataclass
        # structural hash/eq would recurse through `trail` and the whole `parent` chain --
        # O(depth) per heapdict hash/compare instead of O(1), and would let two distinct
        # search branches that happen to be field-identical silently collide as one
        # heapdict key). Open/closed-list membership already uses `is` checks below, so
        # nothing relies on structural equality.
        node: NodeId
        g: int
        covered: CoverageBits
        # Potentially-achievable-path (PAP) summary: pap_g <= g, pap_covered ⊇ covered.
        # When a state is pruned by (epsilon,p)-dominance, its PAP is subsumed into the
        # survivor's PAP (min length, union coverage) instead of being discarded outright.
        pap_g: int
        pap_covered: CoverageBits
        injections_remaining: int
        # Trail used for retract: move endpoints only; inject does not change it.
        trail: Tuple[NodeId, ...]
        # Last primitive applied to reach this state (retract allowed only after inject/retract).
        last_action: Literal["start", "move", "inject", "retract"]
        # Primitive from parent to this state (root has None).
        from_parent: Optional[Literal["move", "inject", "retract"]]
        parent: Optional["IrisPlanner._State[NodeId]"]

    @staticmethod
    def _state_hash_key(
        s: "IrisPlanner._State[NodeId]",
    ) -> Tuple[NodeId, int, CoverageBits, int, Tuple[NodeId, ...], str]:
        """Search-position key for the opt-in duplicate-state debug assertion."""
        return (s.node, s.g, s.covered, s.injections_remaining, s.trail, s.last_action)

    @staticmethod
    def _reconstruct_graph_path(last: IrisPlanner._State[NodeId], *, seed: int, start_node: NodeId) -> GraphPath[NodeId]:
        """Walk a `_State` chain back to the root and emit it as move/inject/retract steps."""
        chain: List[IrisPlanner._State[NodeId]] = []
        s: Optional[IrisPlanner._State[NodeId]] = last
        while s is not None:
            chain.append(s)
            s = s.parent
        chain.reverse()

        if len(chain) <= 1:
            return GraphPath(seed=int(seed), start_node=start_node, num_steps=0, steps=tuple())

        steps: List[GraphPathStep[NodeId]] = []
        for k in range(len(chain) - 1):
            s0, s1 = chain[k], chain[k + 1]
            prev_n = chain[k - 1].node if k > 0 else None
            act = s1.from_parent
            if act is None:
                raise RuntimeError("IRIS state chain has non-root child with no from_parent.")
            if act == "inject":
                steps.append(
                    GraphPathStep(
                        step_index=len(steps),
                        action="inject",
                        node=s0.node,
                        previous_node=prev_n,
                        next_node=None,
                    )
                )
            elif act == "retract":
                steps.append(
                    GraphPathStep(
                        step_index=len(steps),
                        action="retract",
                        node=s0.node,
                        previous_node=s1.node,
                        next_node=None,
                    )
                )
            elif act == "move":
                steps.append(
                    GraphPathStep(
                        step_index=len(steps),
                        action="move",
                        node=s0.node,
                        previous_node=prev_n,
                        next_node=s1.node,
                    )
                )
            else:
                raise RuntimeError(f"Unknown action {act!r} in IRIS state chain.")

        return GraphPath(seed=int(seed), start_node=start_node, num_steps=len(steps), steps=tuple(steps))

    @classmethod
    def _iris_search_best_state(
        cls,
        *,
        graph: Graph[NodeId],
        start_node: NodeId,
        seed: int,
        node_masks: Dict[NodeId, CoverageBits],
        all_mask: CoverageBits,
        max_injections_per_run: int,
        p0: float,
        epsilon0: float,
        tightening_f: float,
        max_outer_iters: int,
        max_expansions: int,
        allow_retract: bool = False,
    ) -> IrisPlanner._State[NodeId]:
        """
        Best-first graph inspection search (near-optimal, (epsilon,p)-approximate).

        `allow_retract=False` disables retract actions entirely (and, since trail
        history is then irrelevant, trail-dominance checking too) -- useful when
        retract's added search breadth isn't worth the cost.
        """
        p = float(p0)
        epsilon = float(epsilon0)
        f = float(tightening_f)
        outer = int(max_outer_iters)
        expansions_budget = int(max_expansions)
        check_trail_dominance = bool(allow_retract)

        def h(_state: IrisPlanner._State[NodeId]) -> int:
            return 0

        def subsume_pap(
            dst: IrisPlanner._State[NodeId],
            src: IrisPlanner._State[NodeId],
        ) -> IrisPlanner._State[NodeId]:
            """Merge src's PAP into dst's (min length, union coverage), keeping dst's identity otherwise."""
            new_pap_g = min(int(dst.pap_g), int(src.pap_g))
            new_pap_cov = int(dst.pap_covered) | int(src.pap_covered)
            if new_pap_g == dst.pap_g and new_pap_cov == dst.pap_covered:
                return dst
            IrisLogger.log_pap_subsumption()
            return cls._State(
                node=dst.node,
                g=dst.g,
                covered=dst.covered,
                pap_g=new_pap_g,
                pap_covered=new_pap_cov,
                injections_remaining=dst.injections_remaining,
                trail=dst.trail,
                last_action=dst.last_action,
                from_parent=dst.from_parent,
                parent=dst.parent,
            )

        open_by_node: Dict[NodeId, List[IrisPlanner._State[NodeId]]] = {}
        closed_by_node: Dict[NodeId, List[IrisPlanner._State[NodeId]]] = {}
        open_pq: "heapdict" = heapdict()
        pq_seq = 0

        def _push_to_open_pq(s: IrisPlanner._State[NodeId]) -> None:
            nonlocal pq_seq
            pq_seq += 1
            open_pq[s] = (s.g + h(s), -s.covered.bit_count(), s.g, pq_seq)

        def _remove_from_open_pq(s: IrisPlanner._State[NodeId]) -> None:
            if s in open_pq:
                del open_pq[s]

        def _replace_in_open_pq(
            old_s: IrisPlanner._State[NodeId], new_s: IrisPlanner._State[NodeId]
        ) -> None:
            _remove_from_open_pq(old_s)
            _push_to_open_pq(new_s)

        def _remove_from_open(node: NodeId, s: IrisPlanner._State[NodeId]) -> bool:
            for i, ex in enumerate(open_by_node.get(node, ())):
                if ex is s:
                    open_by_node[node].pop(i)
                    return True
            return False

        def _is_in_open(node: NodeId, s: IrisPlanner._State[NodeId]) -> bool:
            for ex in open_by_node.get(node, ()):
                if ex is s:
                    return True
            return False

        def _strictly_dominated_by_closed(*, node: NodeId, s: IrisPlanner._State[NodeId]) -> bool:
            for ex in closed_by_node.get(node, ()):
                if cls._strict_dominates(
                    g_a=ex.g,
                    covered_a=ex.covered,
                    injections_remaining_a=ex.injections_remaining,
                    trail_a=ex.trail,
                    g_b=s.g,
                    covered_b=s.covered,
                    injections_remaining_b=s.injections_remaining,
                    trail_b=s.trail,
                    check_trail_dominance=check_trail_dominance,
                ):
                    return True
            return False

        def maybe_keep_open_state(
            *, s: IrisPlanner._State[NodeId], epsilon: float, p: float
        ) -> Optional[IrisPlanner._State[NodeId]]:
            """
            Insert `s` into the per-node OPEN list under (epsilon,p)-dominance
            pruning, subsuming each pruned state's PAP into the survivor.
            Returns the state to push to OPEN (`s` or a replaced copy), or
            None if `s` is dominated and should be discarded.
            """
            end_node = s.node
            open_list = open_by_node.setdefault(end_node, [])

            for i, ex in enumerate(open_list):
                if cls._approx_dominates(
                    g_a=ex.g,
                    covered_a=ex.covered,
                    injections_remaining_a=ex.injections_remaining,
                    trail_a=ex.trail,
                    g_b=s.g,
                    covered_b=s.covered,
                    injections_remaining_b=s.injections_remaining,
                    trail_b=s.trail,
                    epsilon=epsilon,
                    p=p,
                    check_trail_dominance=check_trail_dominance,
                ):
                    # subsume_pap() may return a *new* _State instance. If we replace the
                    # OPEN entry without re-pushing to the PQ under the new object, the
                    # updated state may never be popped/expanded (heapdict keys are the
                    # state objects themselves).
                    updated = subsume_pap(ex, s)
                    open_list[i] = updated
                    if updated is not ex:
                        _replace_in_open_pq(ex, updated)
                    return None

            new_list: List[IrisPlanner._State[NodeId]] = []
            s_acc = s
            for ex in open_list:
                if cls._approx_dominates(
                    g_a=s_acc.g,
                    covered_a=s_acc.covered,
                    injections_remaining_a=s_acc.injections_remaining,
                    trail_a=s_acc.trail,
                    g_b=ex.g,
                    covered_b=ex.covered,
                    injections_remaining_b=ex.injections_remaining,
                    trail_b=ex.trail,
                    epsilon=epsilon,
                    p=p,
                    check_trail_dominance=check_trail_dominance,
                ):
                    _remove_from_open_pq(ex)
                    s_acc = subsume_pap(s_acc, ex)
                    continue
                new_list.append(ex)
            new_list.append(s_acc)
            open_by_node[end_node] = new_list
            return s_acc

        all_states_hash: set = set()

        def _push_new_state(**kwargs: object) -> None:
            state = cls._State(**kwargs)  # type: ignore[arg-type]

            if _strictly_dominated_by_closed(node=state.node, s=state):
                return
            kept_state = maybe_keep_open_state(s=state, epsilon=epsilon, p=p)
            if kept_state is None:
                return

            if DEBUG_ASSERT_UNIQUE_STATES:
                key = cls._state_hash_key(kept_state)
                if key in all_states_hash:
                    raise RuntimeError(
                        "duplicate IRIS state (DEBUG_ASSERT_UNIQUE_STATES): "
                        f"node={kept_state.node} g={kept_state.g} trail={kept_state.trail} "
                        f"covered={kept_state.covered} last_action={kept_state.last_action}"
                    )
                all_states_hash.add(key)

            _push_to_open_pq(kept_state)

        start_state = cls._State(
            node=start_node,
            g=0,
            covered=0,
            pap_g=0,
            pap_covered=0,
            injections_remaining=int(max_injections_per_run),
            trail=(start_node,),
            last_action="start",
            from_parent=None,
            parent=None,
        )
        best_any: IrisPlanner._State[NodeId] = start_state

        for _ in range(int(outer)):
            open_pq = heapdict()
            open_by_node = {start_node: [start_state]}
            closed_by_node = {}
            pq_seq = 0
            all_states_hash = set()
            if DEBUG_ASSERT_UNIQUE_STATES:
                all_states_hash.add(cls._state_hash_key(start_state))
            _push_to_open_pq(start_state)

            expansions = 0
            best_full: Optional[IrisPlanner._State[NodeId]] = None
            best_cov_cnt = best_any.covered.bit_count()

            while open_pq and expansions < expansions_budget:
                cur, _priority = open_pq.popitem()

                # Ignore stale PQ entries that were pruned/replaced in the per-node OPEN list.
                if not _is_in_open(cur.node, cur):
                    continue

                _remove_from_open(cur.node, cur)
                closed_by_node.setdefault(cur.node, []).append(cur)

                expansions += 1

                cur_cov_cnt = cur.covered.bit_count()
                if cur_cov_cnt > best_cov_cnt or (cur_cov_cnt == best_cov_cnt and cur.g < best_any.g):
                    best_any = cur
                    best_cov_cnt = cur_cov_cnt

                if (cur.covered & all_mask) == all_mask:
                    best_full = cur
                    break

                # No further coverage is possible: inject is the only way to add POIs, and none left.
                if cur.injections_remaining <= 0:
                    continue

                # Inject at current node if it strictly increases coverage.
                injection_delta_start = time.perf_counter()
                injection_mask = node_masks.get(cur.node, 0)
                if injection_mask & ~cur.covered:
                    covered_after_injection = cur.covered | injection_mask
                    _push_new_state(
                        node=cur.node,
                        g=cur.g + cls.INJECTION_COST,
                        covered=covered_after_injection,
                        pap_g=cur.g + cls.INJECTION_COST,
                        pap_covered=covered_after_injection,
                        injections_remaining=cur.injections_remaining - 1,
                        trail=cur.trail,
                        last_action="inject",
                        from_parent="inject",
                        parent=cur,
                    )
                IrisLogger.injection_delta_time += time.perf_counter() - injection_delta_start

                # Retract (pop move trail): only right after inject/retract; cost 1.
                if (
                    allow_retract
                    and cur.last_action in ("inject", "retract")
                    and len(cur.trail) >= 2
                ):
                    new_trail = cur.trail[:-1]
                    back_node = new_trail[-1]
                    _push_new_state(
                        node=back_node,
                        g=cur.g + cls.RETRACT_COST,
                        covered=cur.covered,
                        pap_g=cur.g + cls.RETRACT_COST,
                        pap_covered=cur.covered,
                        injections_remaining=cur.injections_remaining,
                        trail=new_trail,
                        last_action="retract",
                        from_parent="retract",
                        parent=cur,
                    )

                # Move to neighbors (only while injections remain -- otherwise no future coverage).
                neighbors = graph.adjacency.get(cur.node, ())
                for next_node in neighbors or ():
                    _push_new_state(
                        node=next_node,
                        g=cur.g + cls.MOVE_COST,
                        covered=cur.covered,
                        pap_g=cur.g + cls.MOVE_COST,
                        pap_covered=cur.covered,
                        injections_remaining=cur.injections_remaining,
                        trail=cur.trail + (next_node,),
                        last_action="move",
                        from_parent="move",
                        parent=cur,
                    )

            if best_full is not None:
                best_any = best_full

            p, epsilon = cls._tighten(p, epsilon, f)

        return best_any

    @classmethod
    def generate_iris_plan_on_graph(
        cls,
        *,
        graph: Graph[NodeId],
        start_node: NodeId,
        seed: int,
        coverage_fn: Callable[[NodeId], Iterable[POI]],
        p0: Optional[float] = None,
        epsilon0: Optional[float] = None,
        tightening_f: Optional[float] = None,
        max_outer_iters: Optional[int] = None,
        max_expansions: Optional[int] = None,
        max_injections_per_run: Optional[int] = None,
        allow_retract: bool = False,
    ) -> GraphPath[NodeId]:
        """
        Compute a high-level inspection plan over a given graph.

        A POI is covered only when the planner injects at the current node
        (union of `coverage_fn(node)` for that inject); moves/retracts don't
        change coverage. Each inject consumes one of `max_injections_per_run`.
        Path cost `g` counts every move/inject/retract as 1; the heuristic is 0.

        Objective: maximize POIs covered, then minimize path length among plans
        tied on coverage. `allow_retract=False` disables retract actions (and
        trail-dominance checking) for a cheaper, less-thorough search.
        """
        if start_node not in set(graph.nodes):
            raise KeyError(f"start_node {start_node!r} not in graph.nodes")

        p = float(cls.P0 if p0 is None else p0)
        epsilon = float(cls.EPSILON0 if epsilon0 is None else epsilon0)
        f = float(cls.TIGHTENING_F if tightening_f is None else tightening_f)
        outer = int(cls.MAX_OUTER_ITERS if max_outer_iters is None else max_outer_iters)
        expansions_budget = int(cls.MAX_EXPANSIONS if max_expansions is None else max_expansions)
        injection_budget = int(cls.MAX_INJECTIONS_PER_RUN if max_injections_per_run is None else max_injections_per_run)

        if not (0.0 < p <= 1.0):
            raise ValueError("p0 must be in (0,1].")
        if epsilon < 0.0:
            raise ValueError("epsilon0 must be >= 0.")
        if outer <= 0:
            raise ValueError("max_outer_iters must be > 0.")
        if expansions_budget <= 0:
            raise ValueError("max_expansions must be > 0.")
        if injection_budget < 0:
            raise ValueError("max_injections_per_run must be >= 0.")

        IrisLogger.reset()

        node_masks, all_mask = cls._build_node_pois(graph=graph, coverage_fn=coverage_fn)

        if all_mask == 0:
            trivial_state = cls._State(
                node=start_node,
                g=0,
                covered=0,
                pap_g=0,
                pap_covered=0,
                injections_remaining=injection_budget,
                trail=(start_node,),
                last_action="start",
                from_parent=None,
                parent=None,
            )
            return cls._reconstruct_graph_path(trivial_state, seed=int(seed), start_node=start_node)

        best = cls._iris_search_best_state(
            graph=graph,
            start_node=start_node,
            seed=int(seed),
            node_masks=node_masks,
            all_mask=all_mask,
            max_injections_per_run=injection_budget,
            p0=p,
            epsilon0=epsilon,
            tightening_f=f,
            max_outer_iters=outer,
            max_expansions=expansions_budget,
            allow_retract=bool(allow_retract),
        )
        total_pois = all_mask.bit_count()
        covered_pois = best.covered.bit_count()
        remaining_pois = total_pois - covered_pois
        IrisLogger.log_planning_complete(
            covered_pois=covered_pois,
            remaining_pois=remaining_pois,
            total_pois=total_pois,
            injections_remaining=best.injections_remaining,
            injection_budget=injection_budget,
        )
        return cls._reconstruct_graph_path(best, seed=int(seed), start_node=start_node)


# Backward-compatible alias for callers that import the function from this module.
generate_iris_plan_on_graph = IrisPlanner.generate_iris_plan_on_graph
