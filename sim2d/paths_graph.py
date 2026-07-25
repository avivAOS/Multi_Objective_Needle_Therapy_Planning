from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
from functools import lru_cache
from typing import Callable, Dict, Iterable, Mapping, Optional, Set, Tuple, TypeVar, Hashable

import matplotlib.pyplot as plt
import numpy as np

from .primitives import Orientation, allowed_primitives, apply_primitive
from .demo_map import configure_demo_map
from .world import World2D

logger = logging.getLogger(__name__)

NodeId = TypeVar("NodeId", bound=Hashable)

DosageLevel = int

# Dosage 0 = no therapy. Higher levels treat more tumor area but also damage
# more healthy tissue (see `dosage_to_radius`).
DOSAGE_LEVELS: Tuple[DosageLevel, ...] = (0, 1, 2, 3)


def dosage_to_radius(dose: DosageLevel) -> Tuple[float, float]:
    """
    Dosage -> (coverage_radius, damage_radius). One disk per injection of radius
    r(dose) = 0.5 + 0.25*dose affects everything inside it, so both radii equal
    r(dose); bigger doses reach more tumor AND more healthy tissue. dose == 0 =
    no injection.
    """
    dose = int(dose)
    if dose <= 0:
        return (0.0, 0.0)
    radius = 0.5 + 0.25 * dose
    return (radius, radius)


for _d in range(1, max(DOSAGE_LEVELS) + 1):
    _cov_prev, _dam_prev = dosage_to_radius(_d - 1)
    _cov_cur, _dam_cur = dosage_to_radius(_d)
    assert _cov_cur >= _cov_prev and _dam_cur >= _dam_prev, (
        "dosage_to_radius must be non-decreasing in both radii"
    )
del _d, _cov_prev, _dam_prev, _cov_cur, _dam_cur


def _fmt_points(
    pts: Set[Tuple[float, float]],
    *,
    max_items: int = 8,
) -> str:
    if not pts:
        return "{}"
    items = sorted(pts)
    if len(items) <= int(max_items):
        return "{" + ", ".join(f"({x:.3g},{y:.3g})" for x, y in items) + "}"
    head = items[: int(max_items)]
    return (
        "{"
        + ", ".join(f"({x:.3g},{y:.3g})" for x, y in head)
        + f", ... +{len(items) - int(max_items)}"
        + "}"
    )


def inject_coverage(node: Mapping[str, object]) -> Set[Tuple[float, float]]:
    """Tumor cells covered by an injection at this node (precomputed at graph build)."""
    cov = node.get("inject_coverage", set())
    # Accept either a set/frozenset of tuples or a list/tuple of tuples.
    return set(cov)  # type: ignore[arg-type]


def make_inject_coverage_fn(
    nodes: Mapping[NodeId, Mapping[str, object]],
) -> Callable[[NodeId], Set[Tuple[float, float]]]:
    """Lookup function node_id -> covered tumor cells, so path code stays geometry-free."""

    def _coverage(node_id: NodeId) -> Set[Tuple[float, float]]:
        return inject_coverage(nodes[node_id])

    return _coverage


@lru_cache(maxsize=32)
def _offsets_for_radius(
    nx: int, ny: int, cell_size_x: float, cell_size_y: float, r: float
) -> Tuple[Tuple[int, int], ...]:
    # Precompute integer offsets whose physical distance is within r.
    rx = int(np.ceil(r / cell_size_x))
    ry = int(np.ceil(r / cell_size_y))
    rr = float(r) * float(r)
    offsets: list[Tuple[int, int]] = []
    for diy in range(-ry, ry + 1):
        dy = float(diy) * float(cell_size_y)
        dy2 = dy * dy
        for dix in range(-rx, rx + 1):
            dx = float(dix) * float(cell_size_x)
            if dx * dx + dy2 <= rr:
                offsets.append((dix, diy))
    return tuple(offsets)


def _scan_mask_within_radius(
    *, x: float, y: float, world: World2D, mask: Optional[np.ndarray], radius: float
) -> Set[Tuple[float, float]]:
    """
    Grid-cell centers where `mask` is True within `radius` of (x, y). Used for both
    coverage and damage scans. Only checks precomputed in-radius offsets around the
    nearest cell -- O(k), not O(grid). Cells occluded by an obstacle (a wall between
    (x, y) and the cell) are excluded, since the injection radius shouldn't reach
    through walls.
    """
    world.ensure_tumor_grid()
    x_centers = world._x_centers
    y_centers = world._y_centers
    grid_x_limits = world._tumor_grid_x_limits
    grid_y_limits = world._tumor_grid_y_limits
    if x_centers is None or y_centers is None or mask is None:
        return set()
    if grid_x_limits is None or grid_y_limits is None:
        return set()
    if radius <= 0.0:
        return set()

    r = float(radius)
    nx = int(x_centers.shape[0])
    ny = int(y_centers.shape[0])
    x_min, x_max = grid_x_limits
    y_min, y_max = grid_y_limits
    cell_size_x = (float(x_max) - float(x_min)) / float(nx)
    cell_size_y = (float(y_max) - float(y_min)) / float(ny)
    if cell_size_x <= 0.0 or cell_size_y <= 0.0:
        return set()

    # Map world coordinate to nearest cell index (clamped).
    ix0 = int(np.floor((float(x) - float(x_min)) / cell_size_x))
    iy0 = int(np.floor((float(y) - float(y_min)) / cell_size_y))
    ix0 = max(0, min(nx - 1, ix0))
    iy0 = max(0, min(ny - 1, iy0))

    offsets = _offsets_for_radius(nx, ny, cell_size_x, cell_size_y, r)

    hit: Set[Tuple[float, float]] = set()
    for dix, diy in offsets:
        ix = ix0 + int(dix)
        iy = iy0 + int(diy)
        if 0 <= ix < nx and 0 <= iy < ny:
            if bool(mask[iy, ix]):
                cx = float(x_centers[ix])
                cy = float(y_centers[iy])
                if not world.line_of_sight_blocked((x, y), (cx, cy)):
                    hit.add((cx, cy))
    return hit


def _compute_inject_coverage_for_xy(
    *, x: float, y: float, world: World2D, radius: Optional[float] = None
) -> Set[Tuple[float, float]]:
    r = float(world.tumor_injection_radius) if radius is None else float(radius)
    return _scan_mask_within_radius(x=x, y=y, world=world, mask=world.tumor_mask, radius=r)


def _compute_damage_for_xy(
    *, x: float, y: float, world: World2D, radius: float
) -> Set[Tuple[float, float]]:
    return _scan_mask_within_radius(x=x, y=y, world=world, mask=world.healthy_mask, radius=radius)


def compute_dosage_coverage_and_damage_for_xy(
    *,
    x: float,
    y: float,
    world: World2D,
    dosage_levels: Iterable[DosageLevel] = DOSAGE_LEVELS,
) -> Dict[DosageLevel, Tuple[Set[Tuple[float, float]], Set[Tuple[float, float]]]]:
    """For each dosage level, return (tumor cells covered, healthy cells damaged)."""
    result: Dict[DosageLevel, Tuple[Set[Tuple[float, float]], Set[Tuple[float, float]]]] = {}
    for dose in dosage_levels:
        cov_r, dam_r = dosage_to_radius(dose)
        covered = (
            _compute_inject_coverage_for_xy(x=x, y=y, world=world, radius=cov_r)
            if cov_r > 0.0
            else set()
        )
        damaged = (
            _compute_damage_for_xy(x=x, y=y, world=world, radius=dam_r)
            if dam_r > 0.0
            else set()
        )
        result[int(dose)] = (covered, damaged)
    return result


def dosage_coverage(node: Mapping[str, object], dose: DosageLevel) -> Set[Tuple[float, float]]:
    """Tumor cells covered by `dose` at this node (see `inject_coverage`)."""
    by_dose = node.get("dosage_coverage", {})
    return set(by_dose.get(int(dose), set()))  # type: ignore[union-attr]


def dosage_damage(node: Mapping[str, object], dose: DosageLevel) -> Set[Tuple[float, float]]:
    """Healthy cells damaged by `dose` at this node."""
    by_dose = node.get("dosage_damage", {})
    return set(by_dose.get(int(dose), set()))  # type: ignore[union-attr]


def make_dosage_fn(
    nodes: Mapping[NodeId, Mapping[str, object]],
) -> Callable[[NodeId, DosageLevel], Tuple[Set[Tuple[float, float]], Set[Tuple[float, float]]]]:
    """Return a lookup function: (node_id, dosage) -> (covered, damaged) cell sets."""

    def _dosage(node_id: NodeId, dose: DosageLevel) -> Tuple[Set[Tuple[float, float]], Set[Tuple[float, float]]]:
        node = nodes[node_id]
        return dosage_coverage(node, dose), dosage_damage(node, dose)

    return _dosage


def _build_node_dict(
    *, x: float, y: float, ori: Orientation, world: World2D
) -> dict:
    """
    A node's data dict: position/orientation, the single-radius `inject_coverage`
    (used by iris/random planners), and the dosage-indexed `dosage_coverage` /
    `dosage_damage` maps (used by the DP planner).
    """
    legacy_cov = _compute_inject_coverage_for_xy(x=x, y=y, world=world)
    by_dose = compute_dosage_coverage_and_damage_for_xy(x=x, y=y, world=world)
    return {
        "x": float(x),
        "y": float(y),
        "ori": ori,
        "inject_coverage": legacy_cov,
        "dosage_coverage": {dose: cov for dose, (cov, _dam) in by_dose.items()},
        "dosage_damage": {dose: dam for dose, (_cov, dam) in by_dose.items()},
    }


def generate_paths_graph(
    *,
    steps: int,
    step_size: float,
    start_x: float,
    start_y: float,
    start_orientation: Orientation,
    world: World2D | None = None,
) -> tuple[
    Dict[tuple[float, float, int], dict],
    Set[tuple[tuple[float, float, int], tuple[float, float, int]]],
    Dict[tuple[float, float, int], int],
]:
    """
    BFS roadmap to depth `steps` via one-step primitives. Nodes are
    (x, y, orientation) with x/y quantized to 4 decimals; an edge u->v exists when
    v is reachable from u by one `allowed_primitives` move. Returns (nodes, edges,
    depth).
    """
    if world is None:
        world = World2D()
        configure_demo_map(world, start_x=start_x, start_y=start_y)

    def quantize_xy(x: float, y: float) -> tuple[float, float]:
        return (round(float(x), 4), round(float(y), 4))

    start_theta = start_orientation.to_angle()
    start_state = np.array([float(start_x), float(start_y), start_theta], dtype=float)
    if world.in_collision(start_state):
        return {}, set(), {}

    start_key = (*quantize_xy(start_x, start_y), int(start_orientation))
    start_node = _build_node_dict(x=float(start_x), y=float(start_y), ori=start_orientation, world=world)
    nodes: Dict[tuple[float, float, int], dict] = {start_key: start_node}
    logger.info(
        "graph.node_add key=%s xy=(%.4f,%.4f) theta=%.4f cov=%s",
        start_key,
        float(start_x),
        float(start_y),
        float(start_theta),
        _fmt_points(start_node["inject_coverage"]),
    )
    edges: Set[tuple[tuple[float, float, int], tuple[float, float, int]]] = set()
    discovered_depth: Dict[tuple[float, float, int], int] = {start_key: 0}
    expanded_depth: Dict[tuple[float, float, int], int] = {}

    q: deque[tuple[float, float, int]] = deque([start_key])

    while q:
        u_key = q.popleft()
        depth = discovered_depth[u_key]
        if u_key not in expanded_depth:
            expanded_depth[u_key] = depth
        if depth >= int(steps):
            continue

        u = nodes[u_key]
        u_x = float(u["x"])
        u_y = float(u["y"])
        u_ori: Orientation = u["ori"]

        for primitive in allowed_primitives(u_ori):
            x_new, y_new, theta_new = apply_primitive(
                u_x,
                u_y,
                u_ori,
                primitive,
                step_size=step_size,
            )
            v_ori = primitive.new_orientation
            v_state = np.array([x_new, y_new, theta_new], dtype=float)
            if world.in_collision(v_state):
                continue

            v_key = (*quantize_xy(x_new, y_new), int(v_ori))
            edges.add((u_key, v_key))

            if v_key not in discovered_depth:
                discovered_depth[v_key] = depth + 1
                v_node = _build_node_dict(x=float(x_new), y=float(y_new), ori=v_ori, world=world)
                nodes[v_key] = v_node
                logger.info(
                    "graph.node_add key=%s xy=(%.4f,%.4f) theta=%.4f cov=%s",
                    v_key,
                    float(x_new),
                    float(y_new),
                    float(theta_new),
                    _fmt_points(v_node["inject_coverage"]),
                )
                q.append(v_key)

    return nodes, edges, expanded_depth


def plot_paths_graph(
    *,
    nodes: Dict[tuple[float, float, int], dict],
    edges: Set[tuple[tuple[float, float, int], tuple[float, float, int]]],
    distances: Dict[tuple[float, float, int], int],
    step_size: float,
    start_x: float,
    start_y: float,
    start_orientation: Orientation,
    steps: int,
    output_path: Path,
    no_gui: bool,
    world: World2D,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    world.render(ax=ax, clear=True, show_robot=False, show_trail=False, show_goal=False)

    for u_key, v_key in edges:
        u = nodes.get(u_key)
        v = nodes.get(v_key)
        if u is None or v is None:
            continue
        ax.plot(
            [u["x"], v["x"]],
            [u["y"], v["y"]],
            color="0.6",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )

    near_intensity = 0.15
    far_intensity = 0.75
    alpha = 0.9

    max_d = max(distances.values(), default=0)
    node_items: list[tuple[int, tuple[float, float, int], dict]] = [
        (distances.get(k, max_d), k, v) for k, v in nodes.items()
    ]
    node_items.sort(key=lambda t: t[0], reverse=True)

    for d, _k, v in node_items:
        t = 0.0 if max_d <= 0 else float(d) / float(max_d)
        t = max(0.0, min(1.0, t))
        intensity = (1.0 - t) * near_intensity + t * far_intensity
        z = 2 + (max_d - d)
        ax.scatter(
            [v["x"]],
            [v["y"]],
            s=30,
            color=(intensity, intensity, intensity, alpha),
            zorder=z,
        )

    if nodes:
        ax.scatter([start_x], [start_y], s=120, marker="*", color="k", zorder=3)
    ax.set_title(f"Paths graph (step_size={step_size})")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if not no_gui:
        plt.show()
    plt.close(fig)

