from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np


def _build_directions() -> Tuple[Tuple[int, int, int], ...]:
    """All 26 integer lattice directions (each component in {-1,0,1}, not all zero)."""
    dirs: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                dirs.append((dx, dy, dz))
    return tuple(dirs)


# 26 discrete movement directions on a 3D lattice. The orientation of the needle
# is simply an index into this table (the 3D analog of the 8-way `Orientation`
# enum used by the 2D simulator).
DIRECTIONS_3D: Tuple[Tuple[int, int, int], ...] = _build_directions()
NUM_ORIENTATIONS: int = len(DIRECTIONS_3D)  # 26

# Default maximum heading change per step (degrees). 46° lets an axial heading
# turn onto any of its four 45° face-diagonal neighbours (the 3D analog of the
# 2D "turn ±45°" rule) while keeping the needle from doubling back.
DEFAULT_MAX_TURN_DEG: float = 46.0


def direction_vector(ori: int) -> Tuple[int, int, int]:
    """Return the integer lattice direction for an orientation index."""
    return DIRECTIONS_3D[int(ori) % NUM_ORIENTATIONS]


def _unit(vec: Tuple[float, float, float]) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def direction_from_vector(vec: Tuple[float, float, float]) -> int:
    """Quantize an arbitrary direction vector to the nearest lattice orientation."""
    u = _unit(vec)
    if not np.any(u):
        return 0
    best_i = 0
    best_dot = -2.0
    for i, d in enumerate(DIRECTIONS_3D):
        dot = float(np.dot(u, _unit(d)))
        if dot > best_dot:
            best_dot = dot
            best_i = i
    return best_i


def orientation_to_heading(ori: int) -> Tuple[float, float, float]:
    """Return the normalized heading (unit vector) for an orientation index."""
    return tuple(_unit(direction_vector(ori)).tolist())  # type: ignore[return-value]


@dataclass(frozen=True)
class MotionPrimitive3D:
    """
    One-step 3D motion primitive on a lattice.

    For a normal move the needle changes its heading to `new_orientation` and
    advances `step_size` along the corresponding lattice direction.
    `is_inject` triggers tumor healing without moving; `is_retract` steps one
    pose back along the traversal history.
    """

    new_orientation: int
    is_inject: bool = False
    is_retract: bool = False


@lru_cache(maxsize=None)
def allowed_orientations(current_ori: int, max_turn_deg: float) -> Tuple[int, ...]:
    """
    Return orientation indices reachable from `current_ori` within `max_turn_deg`.

    This is the 3D generalization of `allowed_primitives` in the 2D simulator:
    instead of {-45, 0, +45} on a circle, we keep every lattice direction whose
    angle to the current heading is at most `max_turn_deg` (a forward cone).
    """
    cur = _unit(direction_vector(current_ori))
    cos_thresh = float(np.cos(np.deg2rad(max_turn_deg)))
    out: List[int] = []
    for i in range(NUM_ORIENTATIONS):
        dot = float(np.dot(cur, _unit(DIRECTIONS_3D[i])))
        if dot >= cos_thresh - 1e-9:
            out.append(i)
    return tuple(out)


def allowed_primitives_3d(
    current_ori: int, max_turn_deg: float = DEFAULT_MAX_TURN_DEG
) -> List[MotionPrimitive3D]:
    """Return the move primitives allowed from the current orientation."""
    return [
        MotionPrimitive3D(new_orientation=o, is_inject=False)
        for o in allowed_orientations(int(current_ori), float(max_turn_deg))
    ]


def inject_primitive_3d(orientation: int) -> MotionPrimitive3D:
    """Create an inject primitive (needle does not move)."""
    return MotionPrimitive3D(new_orientation=int(orientation), is_inject=True)


def retract_primitive_3d(orientation: int) -> MotionPrimitive3D:
    """Create a retract primitive (pose comes from history, not from orientation)."""
    return MotionPrimitive3D(new_orientation=int(orientation), is_inject=False, is_retract=True)


def apply_primitive_3d(
    x: float,
    y: float,
    z: float,
    orientation: int,
    primitive: MotionPrimitive3D,
    step_size: float = 1.0,
    *,
    pose_history: List[tuple[float, float, float, int]] | None = None,
) -> Tuple[float, float, float, int]:
    """
    Apply a 3D motion primitive, returning (x_new, y_new, z_new, orientation_new).

    Like the 2D version, lattice motion is *not* normalized: an axial step moves
    `step_size` along one axis and a diagonal step moves `step_size` along each
    active axis. This keeps every reachable pose on a regular lattice so nodes
    merge cleanly during graph generation.
    """
    if primitive.is_inject:
        return float(x), float(y), float(z), int(orientation)

    if primitive.is_retract:
        if not pose_history or len(pose_history) < 2:
            return float(x), float(y), float(z), int(orientation)
        x_prev, y_prev, z_prev, ori_prev = pose_history[-2]
        return float(x_prev), float(y_prev), float(z_prev), int(ori_prev)

    new_ori = int(primitive.new_orientation)
    dx, dy, dz = direction_vector(new_ori)
    x_new = x + step_size * float(dx)
    y_new = y + step_size * float(dy)
    z_new = z + step_size * float(dz)
    return float(x_new), float(y_new), float(z_new), new_ori