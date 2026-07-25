from __future__ import annotations

"""
Epsilon-dominance and Pareto-frontier compression for the therapy planner.

Pure math over `Plan` values (no graph/world/geometry), so it works unchanged in
2D or 3D. Four objectives: length (min), coverage bitset (max), damage bitset
(min), dose_count (min). Each plan also carries "bound" fields -- the optimistic
best-case over itself plus every plan absorbed into it (see `absorb_discarded`).
"""

from dataclasses import dataclass
from typing import Optional, Union

DosageLevel = int


@dataclass
class CompressionStats:
    """
    Optional counters threaded through `insert_plan`/`compress` to measure how
    much dominance-comparison work actually happened. Comparisons don't
    correlate 1:1 with survivor-list size -- `insert_plan` early-exits the
    scan once an incoming plan is absorbed -- so this is the only way to know
    the true count without re-deriving it from wall-clock timing.
    """
    comparisons: int = 0
    plans_inserted: int = 0   # every plan ever fed to insert_plan (~ construction churn)
    plans_absorbed: int = 0   # existing survivors evicted by an incoming plan
    plans_rejected: int = 0   # incoming plans discarded as dominated on arrival


@dataclass(frozen=True, slots=True)
class EpsVec:
    """Per-objective tolerance as a fraction (0.0 = exact Pareto, 0.5 = within 50%)."""
    length_tol: float
    coverage_tol: float
    damage_tol: float
    dose_count_tol: float


def scale_eps_for_depth(eps: EpsVec, depth: int) -> EpsVec:
    """
    Shrink a global target tolerance to the per-node tolerance eps' =
    (1+eps)^(1/depth)-1, so per-node error (which compounds as (1+eps)^depth in a
    depth-`depth` DP) sums back to the global (1+eps) at the root. Only a clean
    end-to-end bound when the per-node cap is not also truncating the frontier.
    """
    depth = max(1, int(depth))
    if depth == 1:
        return eps
    rescale = lambda tol: (1.0 + tol) ** (1.0 / depth) - 1.0
    return EpsVec(
        length_tol=rescale(eps.length_tol),
        coverage_tol=rescale(eps.coverage_tol),
        damage_tol=rescale(eps.damage_tol),
        dose_count_tol=rescale(eps.dose_count_tol),
    )


@dataclass(frozen=True, slots=True)
class InjectionChoice:
    """Leaf origin: chose `dosage` at `node` (0 = no injection), visiting no children."""
    node: object
    dosage: DosageLevel


@dataclass(frozen=True, slots=True)
class SubtourMerge:
    """Origin: extend `base` at `node` by going to `child`, running `child_plan`, retracting."""
    node: object
    child: object
    base: "Plan"
    child_plan: "Plan"


PlanOrigin = Union[InjectionChoice, SubtourMerge]


@dataclass(frozen=True, slots=True)
class Plan:
    """
    A candidate plan: four objectives, their optimistic `bound_*` counterparts,
    and an `origin` tree for path reconstruction. Bounds start equal to the actual
    values and widen as other plans are absorbed (see `absorb_discarded`).
    """
    node: object

    length: float
    coverage: int    # tumor-coverage bitset
    damage: int      # healthy-tissue-damage bitset
    dose_count: int

    bound_length: float
    bound_coverage: int
    bound_damage: int
    bound_dose_count: int

    # Popcounts of the actual coverage/damage bitsets, cached to avoid recomputing
    # them on every dominance check (absorb_discarded carries them over unchanged).
    coverage_count: int
    damage_count: int

    origin: PlanOrigin


def make_local_plan(
    *, node: object, dosage: DosageLevel, length: float, coverage: int, damage: int
) -> Plan:
    """Fresh leaf plan for a single dosage choice at `node`."""
    dose_count = 1 if int(dosage) > 0 else 0
    return Plan(
        node=node,
        length=length,
        coverage=coverage,
        damage=damage,
        dose_count=dose_count,
        bound_length=length,
        bound_coverage=coverage,
        bound_damage=damage,
        bound_dose_count=dose_count,
        coverage_count=coverage.bit_count(),
        damage_count=damage.bit_count(),
        origin=InjectionChoice(node=node, dosage=int(dosage)),
    )


def make_merge_plan(
    *, node: object, child: object, base: Plan, child_plan: Plan, round_trip_cost: float
) -> Plan:
    """Plan that extends `base` at `node` by visiting `child`'s sub-tour."""
    length = base.length + child_plan.length + round_trip_cost
    coverage = base.coverage | child_plan.coverage
    damage = base.damage | child_plan.damage
    dose_count = base.dose_count + child_plan.dose_count
    return Plan(
        node=node,
        length=length,
        coverage=coverage,
        damage=damage,
        dose_count=dose_count,
        bound_length=length,
        bound_coverage=coverage,
        bound_damage=damage,
        bound_dose_count=dose_count,
        coverage_count=coverage.bit_count(),
        damage_count=damage.bit_count(),
        origin=SubtourMerge(node=node, child=child, base=base, child_plan=child_plan),
    )


def _best_case_bound(a: Plan, b: Plan) -> tuple[float, int, int, int]:
    """Joint optimistic bound over a and b: min length, union coverage, intersection
    damage, min dose_count (on the bound fields, so absorbed potential is included)."""
    return (
        min(a.bound_length, b.bound_length),
        a.bound_coverage | b.bound_coverage,
        a.bound_damage & b.bound_damage,
        min(a.bound_dose_count, b.bound_dose_count),
    )


def eps_dominates(a: Plan, b: Plan, eps: EpsVec) -> bool:
    """
    True if `a` is within tolerance of the joint best-case of (a, b) on all four
    objectives -- i.e. `b` is not worth keeping as a separate frontier entry.

    Cheap scalar checks (length, dose) run first so a failure skips the wide-bitset
    OR/AND/popcount. A zero joint best on coverage/damage/dose forces `a` to match
    exactly (no tolerance can be granted around zero).
    """
    best_length = min(a.bound_length, b.bound_length)
    if a.length > (1.0 + eps.length_tol) * best_length:
        return False

    best_dose_count = min(a.bound_dose_count, b.bound_dose_count)
    if best_dose_count == 0:
        if a.dose_count != 0:
            return False
    elif a.dose_count > (1.0 + eps.dose_count_tol) * best_dose_count:
        return False

    best_coverage_count = (a.bound_coverage | b.bound_coverage).bit_count()
    if best_coverage_count > 0 and a.coverage_count < best_coverage_count / (1.0 + eps.coverage_tol):
        return False

    best_damage_count = (a.bound_damage & b.bound_damage).bit_count()
    if a.damage_count > (1.0 + eps.damage_tol) * best_damage_count:
        return False

    return True


def absorb_discarded(survivor: Plan, discarded: Plan) -> Plan:
    """Return `survivor` with its bounds widened to the joint best-case of (survivor,
    discarded). Actual values, counts and origin are unchanged -- it is the same plan,
    now also accounting for the pruned plan's potential in future dominance checks."""
    best_length, best_coverage, best_damage, best_dose_count = _best_case_bound(survivor, discarded)
    return Plan(
        node=survivor.node,
        length=survivor.length,
        coverage=survivor.coverage,
        damage=survivor.damage,
        dose_count=survivor.dose_count,
        bound_length=best_length,
        bound_coverage=best_coverage,
        bound_damage=best_damage,
        bound_dose_count=best_dose_count,
        coverage_count=survivor.coverage_count,
        damage_count=survivor.damage_count,
        origin=survivor.origin,
    )


def insert_plan(
    survivors: list[Plan], p: Plan, eps: EpsVec, stats: Optional[CompressionStats] = None
) -> list[Plan]:
    """
    Insert `p` into the eps-non-dominated `survivors`. Per existing survivor `s`:
    `s` dominates `p` -> absorb `p` into `s`, done; `p` dominates `s` -> drop `s`,
    fold its potential into `p`; neither -> keep `s`. Unabsorbed `p` is appended.

    `stats`, if given, is updated with exactly the `eps_dominates` calls this
    invocation makes (1 if `p` is absorbed on the first check against a given
    survivor, 2 if not) -- purely additive bookkeeping, no effect on the result.
    """
    p_absorbed = False
    result: list[Plan] = []
    for s in survivors:
        if p_absorbed:
            result.append(s)
        elif eps_dominates(s, p, eps):
            if stats is not None:
                stats.comparisons += 1
                stats.plans_rejected += 1
            result.append(absorb_discarded(s, p))
            p_absorbed = True
        elif eps_dominates(p, s, eps):
            if stats is not None:
                stats.comparisons += 2
                stats.plans_absorbed += 1
            p = absorb_discarded(p, s)
        else:
            if stats is not None:
                stats.comparisons += 2
            result.append(s)
    if not p_absorbed:
        result.append(p)
    if stats is not None:
        stats.plans_inserted += 1
    return result


def compress(plans: list[Plan], eps: EpsVec, stats: Optional[CompressionStats] = None) -> list[Plan]:
    """Reduce `plans` to an eps-non-dominated frontier (unbounded size). For a
    size-bounded version see `_compress_with_cap` in dp_on_graph."""
    survivors: list[Plan] = []
    for p in plans:
        survivors = insert_plan(survivors, p, eps, stats=stats)
    return survivors
