from __future__ import annotations

from collections import deque
import logging
import time
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, Mapping, Set, Tuple, TypeVar

import numpy as np

from .primitives3d import (
    DEFAULT_MAX_TURN_DEG,
    allowed_primitives_3d,
    apply_primitive_3d,
)
from .world3d import World3D

logger = logging.getLogger(__name__)

NodeId = TypeVar("NodeId", bound=Hashable)

# A 3D node key: (x, y, z, orientation_index), with x/y/z quantized.
Node3D = Tuple[float, float, float, int]
# A POI is an opaque, hashable set element: the integer voxel index (ix, iy, iz).
POI3D = Tuple[int, int, int]

DosageLevel = int

# Dosage 0 = no therapy. Higher levels treat more tumor volume but also damage
# more healthy tissue (see `dosage_to_radius_3d`).
DOSAGE_LEVELS: Tuple[DosageLevel, ...] = (0, 1, 2, 3)


def dosage_to_radius_3d(dose: DosageLevel) -> Tuple[float, float]:
    """
    Dosage -> (coverage_radius, damage_radius) in 3D. One injection sphere of
    radius r(dose) affects everything inside it, so both radii equal r(dose) --
    bigger doses reach more tumor AND more healthy tissue. dose == 0 = no
    injection.

    r(dose) = 1.25*dose is anchored to `World3D.tumor_injection_radius=2.5`:
    dose=2 reproduces that existing single-dose radius, so this isn't an
    unconstrained guess -- but it's still a starting point flagged for
    empirical retuning once a rendered 3D map is visually inspectable (2D's
    cell size / world extent are roughly 20x finer than 3D's, so the formulas
    aren't directly portable).
    """
    dose = int(dose)
    if dose <= 0:
        return (0.0, 0.0)
    radius = 1.25 * dose
    return (radius, radius)


for _dose in range(1, max(DOSAGE_LEVELS) + 1):
    _cov_prev, _dam_prev = dosage_to_radius_3d(_dose - 1)
    _cov_cur, _dam_cur = dosage_to_radius_3d(_dose)
    assert _cov_cur >= _cov_prev and _dam_cur >= _dam_prev, (
        "dosage_to_radius_3d must be non-decreasing in both radii"
    )
del _dose, _cov_prev, _dam_prev, _cov_cur, _dam_cur


def inject_coverage(node: Mapping[str, object]) -> Set[POI3D]:
    """Return the POI set covered by injecting at this node."""
    cov = node.get("inject_coverage", frozenset())
    return set(cov)  # type: ignore[arg-type]


def make_inject_coverage_fn(
    nodes: Mapping[NodeId, Mapping[str, object]],
) -> Callable[[NodeId], FrozenSet[POI3D]]:
    """Return a lookup function: node_id -> covered POI set (voxel indices)."""

    def _coverage(node_id: NodeId) -> FrozenSet[POI3D]:
        return nodes[node_id]["inject_coverage"]  # type: ignore[return-value]

    return _coverage


def compute_dosage_coverage_and_damage_for_xyz(
    *,
    x: float,
    y: float,
    z: float,
    world: World3D,
    dosage_levels: Iterable[DosageLevel] = DOSAGE_LEVELS,
) -> Dict[DosageLevel, Tuple[FrozenSet[POI3D], FrozenSet[POI3D]]]:
    """
    For each dosage level, return (tumor voxels covered, healthy voxels damaged)
    as voxel-index POI sets.

    Slices `tumor_mask`/`healthy_mask` to a bounding box around (x, y, z) sized
    to the largest requested dose radius (`World3D.window_bounds`) before doing
    the distance/nonzero work, instead of scanning the whole grid (tens of
    thousands of cells) for every node -- `np.nonzero` on the full grid was the
    single largest cost in graph generation, since it always touches every
    cell regardless of how few actually fall within the injection radius.
    """
    world.ensure_tumor_grid()
    x_centers, y_centers, z_centers = world._x_centers, world._y_centers, world._z_centers
    tumor_mask, healthy_mask = world.tumor_mask, world.healthy_mask

    result: Dict[DosageLevel, Tuple[FrozenSet[POI3D], FrozenSet[POI3D]]] = {}
    if x_centers is None or y_centers is None or z_centers is None or tumor_mask is None:
        return {int(d): (frozenset(), frozenset()) for d in dosage_levels}

    radii = {int(d): dosage_to_radius_3d(d)[0] for d in dosage_levels}
    max_r = max(radii.values(), default=0.0)
    if max_r <= 0.0:
        return {int(d): (frozenset(), frozenset()) for d in dosage_levels}

    ix_lo, ix_hi, iy_lo, iy_hi, iz_lo, iz_hi = world.window_bounds(x, y, z, max_r)
    dx = x_centers[ix_lo:ix_hi, None, None] - x
    dy = y_centers[None, iy_lo:iy_hi, None] - y
    dz = z_centers[None, None, iz_lo:iz_hi] - z
    dist2 = dx * dx + dy * dy + dz * dz

    tumor_window = tumor_mask[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
    healthy_window = (
        healthy_mask[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] if healthy_mask is not None else None
    )

    for dose in dosage_levels:
        r = radii[int(dose)]
        if r <= 0.0:
            result[int(dose)] = (frozenset(), frozenset())
            continue
        within = dist2 <= (r * r)

        coverage_mask = within & tumor_window
        ix, iy, iz = np.nonzero(coverage_mask)
        covered: FrozenSet[POI3D] = frozenset(
            zip((ix + ix_lo).tolist(), (iy + iy_lo).tolist(), (iz + iz_lo).tolist())
        )

        if healthy_window is not None:
            damage_mask = within & healthy_window
            ixd, iyd, izd = np.nonzero(damage_mask)
            damaged: FrozenSet[POI3D] = frozenset(
                zip((ixd + ix_lo).tolist(), (iyd + iy_lo).tolist(), (izd + iz_lo).tolist())
            )
        else:
            damaged = frozenset()

        result[int(dose)] = (covered, damaged)

    return result


def dosage_coverage_3d(node: Mapping[str, object], dose: DosageLevel) -> FrozenSet[POI3D]:
    """Tumor voxels covered by `dose` at this node."""
    by_dose = node.get("dosage_coverage", {})
    return frozenset(by_dose.get(int(dose), frozenset()))  # type: ignore[union-attr]


def dosage_damage_3d(node: Mapping[str, object], dose: DosageLevel) -> FrozenSet[POI3D]:
    """Healthy voxels damaged by `dose` at this node."""
    by_dose = node.get("dosage_damage", {})
    return frozenset(by_dose.get(int(dose), frozenset()))  # type: ignore[union-attr]


def make_dosage_fn_3d(
    nodes: Mapping[NodeId, Mapping[str, object]],
) -> Callable[[NodeId, DosageLevel], Tuple[FrozenSet[POI3D], FrozenSet[POI3D]]]:
    """Return a lookup function: (node_id, dosage) -> (covered, damaged) voxel sets."""

    def _dosage(node_id: NodeId, dose: DosageLevel) -> Tuple[FrozenSet[POI3D], FrozenSet[POI3D]]:
        node = nodes[node_id]
        return dosage_coverage_3d(node, dose), dosage_damage_3d(node, dose)

    return _dosage


def _build_node_dict(
    *,
    x: float,
    y: float,
    z: float,
    ori: int,
    world: World3D,
    compute_dosage: bool = True,
    compute_inject_coverage: bool = True,
) -> dict:
    """
    A node's data dict: position/orientation, the single-radius `inject_coverage`
    (used by iris/random planners), and the dosage-indexed `dosage_coverage` /
    `dosage_damage` maps (used by the DP planner).

    Each representation costs its own grid scan per node, so callers that only
    run one planner family can skip the other via `compute_dosage`/
    `compute_inject_coverage` -- e.g. DP never reads `inject_coverage`, and
    IRIS/random never read `dosage_coverage`/`dosage_damage`.
    """
    node = {"x": float(x), "y": float(y), "z": float(z), "ori": int(ori)}

    if compute_dosage:
        by_dose = compute_dosage_coverage_and_damage_for_xyz(x=x, y=y, z=z, world=world)
        node["dosage_coverage"] = {dose: cov for dose, (cov, _dam) in by_dose.items()}
        node["dosage_damage"] = {dose: dam for dose, (_cov, dam) in by_dose.items()}
    else:
        node["dosage_coverage"] = {}
        node["dosage_damage"] = {}

    node["inject_coverage"] = world.covered_pois_at(x, y, z) if compute_inject_coverage else frozenset()
    return node


def generate_paths_graph_3d(
    *,
    steps: int,
    step_size: float,
    start_x: float,
    start_y: float,
    start_z: float,
    start_orientation: int,
    world: World3D,
    max_turn_deg: float = DEFAULT_MAX_TURN_DEG,
    progress: bool = False,
    compute_dosage: bool = True,
    compute_inject_coverage: bool = True,
) -> tuple[Dict[Node3D, dict], Set[tuple[Node3D, Node3D]], Dict[Node3D, int]]:
    """
    BFS the reachable 3D lattice up to depth `steps` using turn-limited
    one-step primitives. Returns (nodes, edges, discovered_depth).

    Nodes are (x, y, z, orientation) with x/y/z quantized to 4 decimals.

    The 3D lattice's branching factor makes node count grow roughly
    exponentially in `steps` (unlike 2D's much more constrained roadmap), so
    `progress=True` prints a running node count -- without it, a large `steps`
    can run for a very long time with no visible output at all.

    `compute_dosage`/`compute_inject_coverage` pass through to
    `_build_node_dict`: skip whichever representation the caller's planner
    won't read, since each costs its own per-node grid scan.
    """

    def quantize(x: float, y: float, z: float) -> Tuple[float, float, float]:
        return (round(float(x), 4), round(float(y), 4), round(float(z), 4))

    start_pt = np.array([start_x, start_y, start_z], dtype=float)
    if world.in_collision(start_pt):
        return {}, set(), {}

    start_key: Node3D = (*quantize(start_x, start_y, start_z), int(start_orientation))
    nodes: Dict[Node3D, dict] = {
        start_key: _build_node_dict(
            x=float(start_x), y=float(start_y), z=float(start_z), ori=int(start_orientation), world=world,
            compute_dosage=compute_dosage, compute_inject_coverage=compute_inject_coverage,
        )
    }
    edges: Set[tuple[Node3D, Node3D]] = set()
    depth_of: Dict[Node3D, int] = {start_key: 0}
    q: deque[Node3D] = deque([start_key])

    processed = 0
    last_print = time.perf_counter()
    while q:
        u_key = q.popleft()
        processed += 1
        if progress and time.perf_counter() - last_print >= 1.0:
            print(
                f"\rGraph3D: expanded={processed}  discovered={len(nodes)}  "
                f"queue={len(q)}      ",
                end="",
                flush=True,
            )
            last_print = time.perf_counter()
        depth = depth_of[u_key]
        if depth >= int(steps):
            continue
        u = nodes[u_key]
        ux, uy, uz, uori = float(u["x"]), float(u["y"]), float(u["z"]), int(u["ori"])

        for primitive in allowed_primitives_3d(uori, max_turn_deg):
            x_new, y_new, z_new, v_ori = apply_primitive_3d(
                ux, uy, uz, uori, primitive, step_size=step_size
            )
            v_pt = np.array([x_new, y_new, z_new], dtype=float)
            if world.in_collision(v_pt):
                continue
            v_key: Node3D = (*quantize(x_new, y_new, z_new), int(v_ori))
            edges.add((u_key, v_key))
            if v_key not in depth_of:
                depth_of[v_key] = depth + 1
                nodes[v_key] = _build_node_dict(
                    x=float(x_new), y=float(y_new), z=float(z_new), ori=int(v_ori), world=world,
                    compute_dosage=compute_dosage, compute_inject_coverage=compute_inject_coverage,
                )
                q.append(v_key)

    if progress:
        print(flush=True)  # newline after the progress line
    logger.info("graph3d: nodes=%d edges=%d", len(nodes), len(edges))
    return nodes, edges, depth_of
